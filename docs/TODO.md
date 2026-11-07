# Plan — #915 follow-up: `--silk` for training (not just MiniLM embedding)

Issue track: GitHub issue yet to be filed. Parent implementation: PR #978 (`khurram/915`).

## Prior art (the decision made easy)

NivaraClassifier already proves this device-training shape end to end:

- `Program.cs` constructs `GpuTextLinearClassifier` / `GpuTextConvClassifier` inside a `try/catch`,
  falls back to CPU with a single `Silk/OpenCL unavailable (...)` line, and only uses the GPU
  result after `ExportCpuLayout()`.
- It treats training and evaluation as a layout round-trip: weights leave the device as CPU
  `Module` weights (`SetWeights`, `SetWeightsCpuLayout`) and inference afterwards runs on the
  CPU model.
- `--auto-silk` is heuristic, not unconditional: always for conv, only for linear when the batch
  is at least 128. For rag-learn there is no batched token tensors, only 4–36 parameters per query,
  so the heuristic must be more conservative (opt-in in phase 1, evidence-based auto-selection after
  benchmark evidence).
- `Epochs` report `Epoch / Loss / Elapsed` and write human-readable progress per epoch. Rag-learn's
  `EpochReport` already exposes the same fields.

Reusable pieces from `Nivara.Silk`:

- `SilkRuntime` allocation/upload/download/dispatch/dispose.
- kernels already cited (`Gemm*`, `ColSum`, `Add`, `AdamUpdate`), and the host-side softmax
  convention already established by `GpuTextLinearClassifier`.
- pattern for downloading trained weights: `ExportCpuLayout()` for rag-learn becomes
  `ExportWeights()` + an adapter into `FusionWeights` / gated weights.

## Problem

`--rag-learn` already honors `--silk`, but only for MiniLM embedding during index construction
(`MultiViewIndex.BuildAsync` → `MiniLMEmbedder.Create(minilmDir, ctx.UseSilk)`). The heavy CPU
work is not there: embedding the ~3,386 deduplicated view texts is ~2 minutes on CPU and is
cached, so it is paid once. The trainable part itself is tiny — 4 to 1,540 parameters — and
runs sub-second once labelled.

So today `--silk` changes the candidate scores but the optimizer, the fusion graph, and the
gate all run on Nivara's CPU-only `ReverseGradTensor`. The follow-up makes the *whole*
`--rag-learn` pipeline device-native when `--silk` is requested, and keeps a safe CPU
fallback path.

Original direction to respect: no `src/Nivara` changes (#915 scope was sample-only). All OpenCL
work belongs in `samples/NivaraChat/Helpers/` or, if reused, in `Nivara.Silk`; the CPU
AutoDiff baseline in `RetrievalTrainer.cs` remains the reference implementation.

## Ground truth in the repo

- `Nivara.Silk.SilkRuntime` owns context/queue/buffers and has `Ensure`, `Download`, `Upload`,
  `Allocate`, `Run`. It also owns `GpuTextLinearClassifier`, which is the established pattern for
  device-resident forward/backward + fused Adam (see #900).
- `SilkKernels.Gemm`, `GemmNt`, `GemmTn`, `GemmRelu`, `GemmBias`, `GemmEpilogue`, `ColSum`,
  `Add`, `AdamUpdate`, `Fill`, `Silu`, plus pairwise/listwise-relevant losses in
  `SilkLoss.cs`.
- `MiniLMEmbedder.Create(..., useSilk)` already exposes the device name and cleans up its
  runtime; the fusion trainer should mirror the embedded and disposable lifecycle.
- The CPU reference (`RetrievalTrainer.cs`) uses `ReverseGradOperations.MatMul`,
  `Multiply`, `Divide`, `LogSoftmax`, `Sum`, `Subtract`, and `Adam`.
- Derivation needed for device backward: a fixed, hand-written forward/backward rather than
  `ReverseGradTensor`, because Nivara's AutoDiff has no OpenCL tensor backend. This is the
  reusable split that #900 (`GpuTextLinearClassifier`) already established.

## Design choice — three options

### Option A: Device-native fusion trainer (recommended, hold the line on parity)

Host/plan asymmetry is unavoidable: the candidate pool and its scores are CPU arrays.
The plan is a hybrid that is device-authoritative about *weights* and *evaluation*, while the
prediction/gradient math is moved to the device as a closed kernel:

1. At trainer construction, upload the candidate scores and presence masks to device buffers.
   Candidate pool is per-query but up to ~100 rows; one buffer per reusable shape is enough
   with the upload loop at `(query-count, C)` total.
2. For each epoch, for each query:
   * download nothing during the forward: launch a `fusion_forward` kernel over C rows;
   * (CPU) compute the listwise softmax gradient from the downloaded `f` and the constant
     `2^grade` target, upload `g = softmax(f) - target`;
   * launch a `fusion_weight_grad` reduction to accumulate `dTheta` across all C candidates;
   * launch `adam_update` over the 4 weights.
3. Gate stats path (36 params): build the 8-feature vector on CPU, upload it (8 floats),
   reuse `Gemm`/`GemmRelu` for `W·x + b`, and a hand-written `fusion_*` for the rest.
4. Gate embed path (1,540 params): gate matmul `Gemm(W, x, logits, 4, 384 , 1)` then
   fusion path; reuse `Gemm*`.
5. Listwise loss gradient is explicit; pairwise loss deferred to a later milestone.
6. Drawn forward must be identity to the CPU result, measured within `1e-6` per row.

### Option B: Slim `ReverseGradTensor` wrapper over Silk

Rejected for now. The AutoDiff tensor API has shape, grad node, null masks and nullable
`NivaraColumn<T>` plumbing. Reusing it with a mostly-CPU throughput path invalidates the single
benefit the gate is held against (numerical parity). A later core upstream issue may expose a
device `IExecutingTensorBackend`, but that is out of the `--rag-learn` sample's scope.

### Option C: Embed compute on CPU and Silk only for launch overhead

Rejected; worse wall clock, no honesty gain.

## Staged implementation

### Phase 0 — Benchmark and freeze the bar

- Document current CPU numbers from the smoke runs (init parity `≈1e-7`, 24 tests, 22
  `MultiViewScoring` tests). Generate the optimizer step count and learn/eval wall time from
  the report. Before touching Silk, a `--rag-learn --profile` (not in scope; just tee timings)
  captures the CPU time split.
- Acceptance: 4/36/1540 parameters train to the same `p < 0.05` outcome (or the same
  win/loss below that significance) as CPU within 10 % of validation NDCG.
  Significance must not flip because of floating-point differences in the forward path.

### Phase 1 — `SilkFusionTrainer` skeleton + `FusionModel` (4 weights, listwise only)

- New type `Nivara.Samples`- or sample-local helper, probably `samples/NivaraChat/Helpers/SilkRetrievalTrainer.cs`.
- Constructor holds:
  * `SilkRuntime runtime`
  * max candidate count `maxC` (from CLI, default max candidate pool, or `--candidates`)
  * `theta[4]`, host mirror + device buffer
  * `momentum/found` buffers sized `4` each, uploaded zero-initialized
  * `S[rows*4]`, `P[rows*4]`, `f[rows]`, `g[rows]`, `target[rows]`, plus reusable upload space
- Uploads **masked cosine score frames**: `present.Scores[i,k] = score[i,k]`, `present`.
- `Forward query`: upload matrices (or reuse one batch array), run `fusion_forward` kernel:
  * output `f[i] = (Σ_k p[i,k] s[i,k] w[k]) / max(epsilon, Σ_k p[i,k] w[k])`.
  * the epsilon rule is not currently used in `MaskedFusion`, because `Divide` demands
    `b != 0`. The kernel must instead throw — not substitute — so parity is preserved.
- Download logits `f`, host compute `dg[i] = softmax(f)[i] - target[i]`.
- Reduction launch `fusion_wgrad`: `dθ[k] = Σ_i dg[i] * p[i,k] * (s[i,k] - f[i]) / d[i]`,
  where `d[i] = Σ_k p[i,k] w[k]`.
- `AdamUpdate` launch over 4 values, plus host-side mirror refresh (`theta`).
- `ExportWeights()` returns `softmax (theta)`; map back to `FusionWeights` by De Quantization
  or by extension.

### Phase 2 — add kernels and launchers

- `SilkKernels`: `FusionForward` and `FusionWgrad` constants plus host-side source strings.
- `SilkLauncher.FusionForward(runtime, S, P, W, F, rows)` and
  `SilkLauncher.FusionWgrad(runtime, S, P, f, g, W, dTheta, dW, rows, views)` where `dW`
  is the `legacy d` array or `d[i]` computed up-stream. Either precompute `d` and upload, or
  have the kernel recompute in a device loop.
- Gate: reuse `Gemm(..., 4?)` for `W·x` and `AddBias` for the bias. Host softmax tiny.
- Adam moment: use `adam_update` kernel as all text-classifier paths do; initialize with
  zero, per-run.

### Phase 3 — listwise/pairwise choices

- Keep listwise as default. Pairwise needs a second kernel (selector matrix @ fused), defer
  unless requested. Pairwise numbers rely on a well-defined `±1` matrix; still doable via
  a normalized one-row accumulation `pairwise_grad` kernel, but the complexity vs. payoff
  is high for sub-second training. Keep CPU fallback for `--loss pairwise` even when
  `--silk` is on.

### Phase 4 — integration in `RagLearnMode`

- `Load(labels etc.)`, index construction with Silk as existing behavior.
- `ModeContext.UseSilk == true` selects `SilkFusionTrainer` for the 4-value and stats
  gates, otherwise:

  - CPU `FusionModel`.
- Embed-gate: `SilkFusionTrainer` uses `Gemm(..., ..., 1)` path but reuses the same
  regression test inputs.
- Report line `TrainingBackend: CPU|Silk/OpenCL` per model, with device name from
  `runtime.DeviceName`.
- `--loss pairwise` falls back with a printed note, or gracefully delegates to CPU trainer
  while `--silk` remains in effect for embeddings.
- `--silk` meaning stays singular: request `--silk` to use the device **everywhere it can**,
  with documented fallbacks.

### Phase 5 — tests

- `tests/Nivara.Tests/Samples/SilkRetrievalTrainerTests` with synthetic 4-view 3-row input,
  asserting forward and `dθ` match `RetrievalTrainerTests` within `1e-6` (entirely on CPU-vs
  device parity).
- Golden parity: a synthetic run through `SilkTrainer` produces the same `WeightsFor` values
  as `FusionModel` before learning.
- Existing `RetrievalTrainerTests` must remain unchanged.
- Any new CLI flag tests, CLI family.

### Phase 6 — documentation

- `docs/SILK.md` section for `rag-learn`, plus a micro-benchmark showing CPU vs Silk on
  100 queries, run per instructions (division of human-run).
- `samples/NivaraChat/915.md` section 6 updated: full Silk training path now exists, list
  kernels used and benchmark methodology.
- README changes: when the feature lands, revise the comment "training stays CPU" and the
  new `915.md` "Silk training is TBD" notes; remove fallback copy from section 6 when no
  longer needed.

## Compatibility and fallback contract

- `--silk`: embeddings on device; if `SilkRuntime` fails to construct or `MiniLMEmbedder
  .Create` throws, fall back to CPU and print one line.
- If `SilkFusionTrainer` construction fails mid-run, do not switch silently to CPU for the
   same query set — abort the training step, print device reason, report prior nothing; suggest
   re-running without `--silk`.
- No "fork" state: the best weights only live on the device buffer until exported on exit;
  `Exporter` downloads and hands plain `float[]` to the report.
- Ensure disposal (`Dispose()` chain) matches `GpuTextLinearClassifier`'s ownership, because
  the trainer will own its buffers while the index runtime owns its.

## Known risks

1. **Parity numerical drift.** The CPU path uses `TensorPrimitives.SoftMax` over spans of
   locally created memory. The OpenCL kernel may reorder floating-point reductions. First
   test should be a row-wise `fused_` reference, not on full learning. Target `1e-6`, not
   bit-exact, because host `Adam` may accumulate order differently.
2. **Tiny throughput.** A per-query reduction over <=100 rows gains no wall-clock margin over
   the CPU. This plan is a correctness/consistency plan, not a performance plan. If the user
   expects a headline training speedup it cannot deliver without a combined GEMM over all
   candidate rows; `batch` by query would magnify pool sizes but change numerical order.
3. **Backward derivation errors.** The 4-value and stats-gate gradients must be derived
   and unit-tested against `RetrievalTrainer` before use in production.
4. **Pairwise parity** is much harder to derive at listwise-first quality; defer it.
5. **Device memory** is trivial for 4-1540 weights + 100×4 scores. No dedicated memory-gate
   beyond `SilkRuntime` allocations.

## Estimated footprint

- ~ 700 LoC new in sample + ~150 LoC kernel source.
- 1 new kernel family (`fusion_*`) plus `SilkLauncher` dispatch.
- 2-4 new tests, plus golden parity case in existing suite.
- No new `src/Nivara` changes, preserving sample-only scope.

## Exit criteria

1. `--rag-learn --silk` produces the same report and the same `Initialize-parity` line, and
   the weights agree with the CPU run within `1e-5` on a single seeded fixture query.
2. Device path degrades to CPU path with a visible line when no GPU is available.
3. Init and regress do not weaken the existing `RetrievalTrainerTests` or `DocsCoverageTests`.
4. `docs/SILK.md` and `samples/NivaraChat/915.md` sections updated such that "training stays
   CPU" is removed after merge.

## Open questions for human

- Should the GPU trainer use device-fixed candidate array layout or per-query allocation?
- Should the model's stats-gate learned feature (mean/max per query) be CPU by default and
  device only for the fused cotangent?
- What significance impact, if any, is acceptable on the 40-query fixture (expected