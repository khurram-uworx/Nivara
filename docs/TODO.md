# Plan: #956 full device backward for SilkDistilBertTrainer (#953 follow-up)

## Problem

`NivaraFineTuning --mode train --silk` accepts `--silk`, documents it as supported, and then
silently runs the CPU AutoDiff path: `RunTrain` (`samples/NivaraFineTuning/Program.cs`) never reads
`cli.Silk`. PR #957 (issue #953) added two backward kernels (`gelu_backward`,
`layer_norm_backward`) and the CLI accept, and deferred the trainer to #956.

The deferred trainer must run forward, backward and the optimizer on the Silk/OpenCL device, and
materialize the trained weights into a CPU `DistilBertForSequenceClassification` so
`ModelSerializer.Save` keeps working.

## Current state (verified against `main` @ fbfe77bc)

Already present and gated:

- Backward GEMMs: `gemm_row4_nt` (C = A·Bᵀ), `gemm_row4_tn` (C = Aᵀ·B), `gemm_row4_add`
  (C = A·Bᵀ + residual) — one shared accumulation macro, byte-identical across variants.
- `col_sum` (bias / LayerNorm-parameter reductions), `fill`, `add`, `relu_backward`,
  `gelu_backward`, `layer_norm_backward`, `embedding_backward_input` (deterministic scatter:
  one work item per embedding column), `adam_update`.
- Forward path: `SilkBertEncoderRunner` (weights through `WeightLayout.Transposed`/`QkvPacked`,
  cap workspace, per-layer reuse of `x`/`qkv`/`attn`/`h`/`f1`).

Genuinely missing:

1. `attention_backward` (dQ/dK/dV for `batched_attention` with the padding mask and band).
2. A forward that **keeps** per-layer activations (the inference runner reuses one set of buffers).
3. The composed reverse chain, and a `gelu` elementwise kernel so the GELU **pre-activation** is
   available to `gelu_backward` (`gemm_row4_gelu` fuses bias+GELU and stores only the output).
4. A row-major embedding scatter: `embedding_backward_input` addresses
   `dX[b*h*seq + e*seq + s]`, which fits the pooled classifiers, not a `[rows, hidden]` activation.
5. Decoupled weight decay on the device: the CPU path trains with `AdamW(lr, wd: 0.01)`;
   `adam_update` is plain Adam.
6. Materialization of device weights back into HF checkpoint layout for `LoadWeights`/`ModelSerializer`.

Not needed (filed prescription does not hold):

- **No device checkpoint format.** Nothing in the repo defines or consumes one; the only reader of a
  trained model is `ModelSerializer.Save` on a CPU module. Scope is `ExportCheckpoint()` (HF-keyed,
  CPU layout) + `model.LoadWeights(...)`.

## Decisions (cleared with the human before execution)

1. **AdamW fidelity:** extend `adam_update` with a `weightDecay` argument. The two Silk classifiers
   pass `0`; the decay is behind `if (wd != 0)`, so their byte-exact host-formula test still holds.
2. **Correctness gate:** device gradient parity against reverse-mode AutoDiff on a tiny model
   (2 layers, hidden 32, 4 heads, vocab 64, seqLen 8, batch 2) at 5e-4 — the
   `GpuTextLinearClassifierTests` pattern — plus a human-run E2E smoke. The issue's stated
   acceptance (25 examples / 1 epoch) cannot distinguish a correct backward from a broken one,
   and the classifier head is randomly initialized with no seed, so two runs do not even start
   from the same weights.
3. **Checkpointing:** drop the device checkpoint; materialize into the CPU module instead.
4. **Fallback:** `--silk train` fails loudly (exit 1) when the device cannot be initialized. A
   silent CPU fallback would make a green result say nothing about the device path.

## Proposed changes

### A. `attention_backward` (`SilkKernels.Source` + `SilkLauncher` + `KernelNames`)

Two deterministic launches, both reusing the forward's `attn_score` / `attn_keep` / band logic and
re-deriving the softmax (recompute is ~2.4 GFLOP per batch over six layers — negligible against the
6 GEMM layers, and it leaves the forward kernel untouched):

- `attention_backward_dq`: one work item per `(b, h, qPos)`. Computes
  `delta = Σ_j p_ij (dO_i · V_j)` and `dS_ij = p_ij (dO_i·V_j − delta)`, then
  `dQ_i = scale · Σ_j dS_ij K_j`. Fully masked query rows write zeros (no `0/0` NaN).
- `attention_backward_dkv`: one work item per `(b, h, keyPos)`, looping `qPos` ascending:
  `dV_j = Σ_i p_ij dO_i`, `dK_j = scale · Σ_i dS_ij Q_i`.

One work item per output element, ascending inner loop → bit-reproducible CPU replay, which is what
`SilkBackwardKernelTests` asserts.

### B. Glue kernels

- `gelu` (elementwise, `y = gelu_exact(x)`) so the training forward can run
  `gemm_row4_bias` → `gelu` and keep the pre-activation. A test asserts the split form is
  **bit-identical** to `gemm_row4_gelu` at the model's shapes.
- `embedding_backward_rows` (row-major `[rows, hidden]`, one work item per column `e`, ascending
  rows) for word and position embedding gradients.
- `scatter_rows` (CLS-gather backward: zero the row-gradient buffer, write `batch` rows at
  `b * seqLen`).

### C. `adam_update` weight decay

Add a `weightDecay` argument (`w ← w − lr·(update + wd·w)`, guarded), update the three call sites
(two classifiers pass `0`), and extend the existing host-formula test for a non-zero `wd`.

### D. Shared weight upload + `SilkDistilBertTrainer` (`src/Nivara.Extensions/Silk/`)

- Extract the runner's `UploadLayer`/`UploadPlain`/`UploadTransposed` into one internal weight-set
  type; `SilkBertEncoderRunner` delegates to it (no behaviour change, SILK.md §15.5 keeps
  `WeightLayout` as the only layout math).
- `SilkDistilBertTrainer` owns activations, gradients and moments. Per-layer storage:
  `xIn`, packed `qkv`, `attnCtx`, `oOut`, `hPostLn1`, `f1pre`, `f1post`, `fc2Out`
  (≈ 24 MB/layer at the 1024-row cap). Pre-norm sums are recomputed with `add`, not stored.
- Ctor takes `(SilkRuntime, BertConfig, tensors, BertGpuNaming, maxBatch, maxSeqLen)`; workspace
  caps come from the arguments, not from the runner's private constants.
- Public surface: `DeviceName`, `Forward(...)` (logits), `Train(...)` returning
  `IReadOnlyList<SilkTrainEpoch>`, `ExportCheckpoint()`. `Backward`/`Step`/gradient download stay
  `internal` for the parity tests (IVT to `Nivara.Tests` exists).

### E. Backward chain (per batch, reverse)

head → CLS-gather scatter → 6 × (LN2 → FFN + GELU → LN1 → attention → qkv) → embedding LN →
word/pos embedding scatter. Weight gradients are written directly in device layout
(`gemm_row4_tn` with the same packed q/k/v column blocks the forward consumed, so no re-packing);
bias and LayerNorm-parameter gradients via `col_sum` scaled by `1/rows` (mean over the batch).

### F. CLI wiring (`samples/NivaraFineTuning/Program.cs`)

- `RunTrain` branches to `RunTrainSilk`: same tokenization/batching, device epoch loop, per-batch
  loss, per-epoch dev evaluation through the materialized CPU module, final
  `ModelSerializer.Save`.
- Remove the stale "accepted … training currently runs on the CPU AutoDiff path" comment and the
  matching help text; document the device path.
- Fail with exit 1 (message names the device error) when OpenCL is unavailable.

### G. Docs

- `docs/SILK.md`: new subsection for the fine-tuning trainer, kernel-set table rows for the new
  kernels, and the "NivaraFineTuning … training is CPU-only" line removed.
- `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` counts updated in the same commit as any new
  symbol citation.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Targeted GPU tests (`--filter "Category=Gpu"`) — asked before running
- Docs namespace suite because `docs/` changes — asked before running
- Human-run E2E: `NivaraFineTuning --mode train --silk --max-examples 25 --epochs 1`
  (requires `data/distilbert` and `data/sst2`, absent from this checkout)

## Planned commits

1. `feat(silk): add attention_backward kernel` (+ launcher, catalog, CPU-replay test).
2. `feat(silk): add gelu, embedding_backward_rows and scatter_rows kernels`.
3. `feat(silk): add decoupled weight decay to adam_update`.
4. `refactor(silk): share the BERT weight upload between runner and trainer` + trainer forward and
   activation storage; forward-parity test.
5. `feat(silk): add the DistilBERT backward chain`; gradient-parity test.
6. `feat(silk): add device AdamW step and the SilkDistilBertTrainer training loop`.
7. `feat(silk): export trainer weights to the CPU checkpoint layout`.
8. `feat(samples): run --mode train --silk on the device`.
9. `docs: document the Silk fine-tuning trainer`.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` (translation unit + `KernelNames`), `SilkLauncher.cs`.
- `src/Nivara.Extensions/Silk/SilkBertEncoderRunner.cs` (weight upload extracted; forward unchanged).
- new `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`.
- `src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs`, `GpuTextConvClassifier.cs` (one new
  argument to `AdamUpdate`).
- `samples/NivaraFineTuning/Program.cs`.
- `tests/Nivara.Tests/Gpu/SilkBackwardKernelTests.cs`, `SilkKernelsCatalogTests.cs`, new
  `SilkDistilBertTrainerTests.cs`.
- `docs/SILK.md`, `tests/Nivara.Tests/Docs/DocsCoveragePins.cs`.
- Covered by: `Category=Gpu` suites, `SilkKernelsCatalogTests` (device-free), the Docs namespace
  suite, `dotnet build`.

## GitHub issues log

- [ ] #956 — full device backward for `SilkDistilBertTrainer` (this plan)
