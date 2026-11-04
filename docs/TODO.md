# Plan: Silk classifier — device-side backward kernels and parity coverage (#900)

Branch: `khurram/900` (fresh off `main`; the previous `khurram/900` landed as PR #903 —
full-device conv forward + host-gradient training — and was merged).

## Problem

After #903 the classifier paths are half-device:

- `GpuTextConvClassifier` runs the whole forward on the device but computes gradients and
  runs Adam on the host, which requires downloading `convOut`, `h`, and the replicated
  conv input every batch (≈15 MB/batch at the sample's default shapes) and uploading
  updated weights back.
- `GpuTextLinearClassifier` trains with host backward + host SGD, a hard-coded shuffle
  seed (`12345`), returns `void`, and allocates+uploads `embBuf` every batch **without
  ever binding it to a kernel** (dead).
- No backward kernels exist anywhere in `SilkKernels`.
- The issue's parity item is only half-met: conv parity is pinned against an inline
  AutoDiff mirror in the fixture, not the CPU `ConvTextClassifierModel` the issue names
  (it lives in an Exe project the test project does not reference), and the linear
  classifier has no fixture at all.

## Design decisions (cleared with the human, 2026-11-04)

1. **Embedding stays host-resident** (gather/mean-pool/scatter on host). The dead
   `embBuf` is deleted. A device-resident table with a CAS-scatter backward is a
   measured follow-up, not part of #900 (the scatter is ~0.5M adds next to ~1 GFLOP of
   conv backward; and it would make the gradient non-deterministic — see issues log).
2. **Device Adam** (`adam_update` kernel) consumes device-resident gradients. Weights
   become device-authoritative during training; host mirror arrays are refreshed from
   the device at the end of `Train` so `Export*` stays correct.
3. **Parity definition**: fixed-weights forward parity + fixed-batch gradient parity
   (device vs reverse-mode autograd) + a loss-decreasing/accuracy smoke test. Not
   training-trajectory equality — different summation orders, per the conv fixture's
   documented tolerance rationale.
4. **Linear `Train` aligns with conv**: returns `IReadOnlyList<SilkTrainEpoch>`, takes a
   `seed` parameter, trains with Adam.
5. **`ConvTextClassifierModel` moves to `Nivara.Samples`** (namespace
   `Nivara.AutoDiff.Nn`, next to `TextClassifierModel`) so the conv fixture can use the
   real CPU model as its oracle instead of the inline duplicate. Its `Dropout` is
   identity in `Eval` mode, matching the device path.
6. **No softmax/cross-entropy kernel**: logits are B×2 floats, already downloaded for
   the loss report; `dZ` is computed on the host and uploaded. The `1/n` batch scale
   folds into `dZ`, so no separate gradient-scaling pass anywhere downstream.
7. **`mean_pool_backward` is not a kernel**: it is the NT GEMM `dPooled = dH·W1ᵀ` plus a
   `1/seqLen` factor applied per occurrence at the host scatter.
8. **GEMM backward reuses the Row4 accumulate discipline**, not operand transposition:
   new NT/TN kernels transpose the *tile load*, keeping the same ascending-K
   `mul + add` body and the file-scope `#pragma OPENCL FP_CONTRACT OFF`. Byte-identical
   host replay therefore still holds, and no transposed weight copies are needed.

## Changes

### Phase 1 — backward kernels, launcher, catalog, kernel-level parity

New f32 kernels in `SilkKernels` (single translation unit; `KernelNames` entries;
`SilkLauncher` wrappers; `docs/SILK.md` §11.1 + §11.3):

- `gemm_row4_nt` — `C[M,N] = A[M,K]·B[N,K]ᵀ`, B row-major `[N,K]` (tile loads transpose B).
  Covers `dH = dZ·W2ᵀ`, `dPooled = dH·W1ᵀ`, `dA(branch) = dH·W1ᵀ(branch)`.
- `gemm_row4_tn` — `C[M,N] = A[K,M]ᵀ·B[K,N]`, A row-major `[K,M]` (tile loads transpose A).
  Covers `dW2 = hᵀ·dZ`, `dW1 = pooledᵀ·dH`, `dW1(branch) = convOut(branch)ᵀ·dH`.
- `relu_backward` — `y = act > 0 ? d : 0` (mask from the post-activation, which both
  classifiers already keep device-resident: ReLU(x)>0 ⇔ x>0).
- `col_sum` — `y[j] = scale·Σ_i x[i,j]` (bias gradients for FC1/FC2).
- `conv2d_backward_weight` — `dW[oc,ic,kh,kw] = Σ_{n,oh,ow} dA[n,oc,oh,ow]·X[n,ic,ih,iw]`
  (im2col-free correlation; same NCHW/stride/pad/groups geometry args as the forward
  `conv2d`).
- `conv2d_backward_input` — `dX[n,ic,ih,iw] = Σ_{oc,kh,kw} dA[n,oc,oh,ow]·W[oc,ic,kh,kw]`
  (accumulates, so the caller zeroes first; per-branch launches sum across branches).
- `conv2d_backward_bias` — `db[oc] = Σ_{n,oh,ow} dA[n,oc,oh,ow]`.
- `adam_update` — elementwise fused m/v/param step; `bc1 = 1-β1^t`, `bc2 = 1-β2^t`
  precomputed host-side. β1=0.9, β2=0.999, ε=1e-8 — identical constants to the conv
  classifier's host `AdamState`.

Not added (deliberate): softmax/cross-entropy kernel (decision 6), mean-pool backward
kernel (decision 7), embedding scatter kernel (decision 1).

### Phase 2 — linear classifier: device backward + device Adam

`GpuTextLinearClassifier`:

- Delete `embBuf` (allocated, uploaded, never bound — dead).
- Persistent `Ensure` buffers for `pooled/h/logits` (and backward scratch) instead of
  per-call `Allocate`/`Release` in `Forward`.
- `Backward(logits, tokens, labels, batch)` (internal): host dZ (`1/n`-scaled) → upload →
  `col_sum`(db2) → `gemm_row4_tn`(dW2 = hᵀ·dZ) → `gemm_row4_nt`(dH = dZ·W2ᵀ) →
  `relu_backward`(dH, mask = h) → `col_sum`(db1) → `gemm_row4_tn`(dW1 = pooledᵀ·dH) →
  `gemm_row4_nt`(dPooled = dH·W1ᵀ) → download `dPooled` `[B,E]` → host embedding scatter
  (per-occurrence `1/max(1,seqLen)`).
- `adam_update` for fc1/fc2 (m/v device-resident); host Adam for the host-resident
  embedding only.
- `Train` → `IReadOnlyList<SilkTrainEpoch>`, `seed` parameter (shuffle), per-epoch mean
  CE loss. `SilkTrainEpoch` promoted from a nested type of `GpuTextConvClassifier` to a
  top-level `Nivara.Silk` type so both classifiers return the same record.
- `SetWeightsCpuLayout` / `ExportCpuLayout` adapters mirroring the conv classifier's
  (FC weights arrive/leave in CPU `[out,in]` layout).
- `ThrowIfDisposed` guards on public members; disposal is idempotent.

### Phase 3 — conv classifier: device backward + device Adam

`GpuTextConvClassifier`:

- Remove the `hostConvOut` / `hostHidden` / `hostInput` downloads; `ForwardBatch` stops
  returning activation copies (`collectBackwardState` goes away).
- Device backward per batch: upload `dZ` → `col_sum`(db2) → `gemm_row4_tn`(dW2 = hᵀ·dZ) →
  `gemm_row4_nt`(dH) → `relu_backward`(dH, mask = h) → `col_sum`(db1) → per branch:
  `gemm_row4_tn`(dW1 branch block, into a writable subview of the dW1 buffer) +
  `gemm_row4_nt`(dA branch block, into a writable subview of the dA workspace) →
  `relu_backward`(dA, mask = convOut) → per branch: `conv2d_backward_weight` +
  `conv2d_backward_bias` → zero + per branch `conv2d_backward_input` (accumulates the
  three branches into `dX`) → download `dX` → host embedding scatter.
- Host `AdamState` and the packed `gradConv`/`gradFc1`/`gradFc2` layout go away;
  per-parameter gradient buffers (one per weight tensor) + `adam_update` per tensor.
- Host weight mirrors refreshed from the device after `Train`.

### Phase 4 — sample model move + sample hardening

- Move `samples/NivaraClassifier/ConvTextClassifierModel.cs` to
  `samples/Nivara.Samples/ConvTextClassifierModel.cs`, namespace `Nivara.AutoDiff.Nn`
  (same as `TextClassifierModel`). `NivaraClassifier/Program.cs` already opens that
  namespace, so unqualified uses keep resolving.
- `GpuTextConvClassifierTests` switches its oracle from the inline `ReferenceModel` to
  `ConvTextClassifierModel<float>` (with `Eval()` — Dropout is identity), seeding it the
  same way; the inline duplicate is deleted.
- `NivaraClassifier/Program.cs`: Silk train/predict paths dispose their GPU objects
  (the interactive `predict` closure keeps the GPU alive, so disposal moves to the end
  of prediction); Silk paths fall back to the CPU model with a console note when no
  OpenCL GPU is available (acceptance criterion 3); help/console text updated
  ("host-gradient scaffold" / "host-companion" wording is stale once backward is
  device-side).

### Phase 5 — docs, pins, measurement

- `docs/SILK.md`: §11.1 kernel table + `KernelNames` block; §11.3 byte-identity note
  extended to the NT/TN variants (same accumulation body, transposed tile loads).
- `docs/AUTODIFF.md`/`README.md` touch-ups only where wording is now stale.
- Docs coverage: re-run `Nivara.Tests.Docs`; revise `DocsCoveragePins` arithmetic if
  citations change (per the pin doc-comments, not by blind bumping).
- Measurement stays human-run (#901): read `docs/QWEN.md` § "Benchmark commands" if
  benchmark figures are needed; do not run benchmark commands as the agent.

## Blast radius

- `src/Nivara.Extensions/Silk/`: `SilkKernels.cs` (+~250 lines OpenCL C, 8 kernels),
  `SilkLauncher.cs` (+8 wrappers), `GpuTextLinearClassifier.cs` (rewrite of Train),
  `GpuTextConvClassifier.cs` (rewrite of Backward/ApplyAdam), new `SilkTrainEpoch.cs`
  + `SilkLoss.cs`. Callers (code-memory): `samples/NivaraClassifier/Program.cs`,
  `tests/Nivara.Tests/Gpu/GpuTextConvClassifierTests.cs`; `GpuTrainingSeam.cs`
  (`ITextClassifierGpuTrainer`) remains unimplemented (deferred, #883).
- `samples/Nivara.Samples/`: + `ConvTextClassifierModel.cs` (moved).
- `samples/NivaraClassifier/`: Program.cs wiring/help only.
- `tests/Nivara.Tests/Gpu/`: + `SilkBackwardKernelTests.cs`, + `GpuTextLinearClassifierTests.cs`,
  rewrite of `GpuTextConvClassifierTests.cs`; `SilkKernelsCatalogTests.cs` auto-covers
  the new `KernelNames` (device-free).
- `docs/SILK.md` (+ pins if citations change). No `src/Nivara` (core) changes.

## Verification

- `dotnet build Nivara.slnx`
- Targeted GPU tests (device-gated, run autonomously):
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Gpu"`
- Docs suite (SILK.md changes): `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Full suite only with human confirmation (per AGENTS.md).
- Benchmarks: human-run (#901) — the agent asks, never runs them.

Kernel parity expectations (test plan):

- `gemm_row4_nt` / `gemm_row4_tn` / `col_sum` / `relu_backward` / the three
  `conv2d_backward_*` kernels: **byte-exact** against a CPU reference replaying the same
  ascending order (file-scope `FP_CONTRACT OFF`; only mul/add/reduce involved).
- `adam_update`: device-vs-device bitwise determinism (two runs identical) +
  tolerance vs the host formula (≈1e-5 rel) — div/sqrt are not guaranteed correctly
  rounded without the `-cl-fp32-correctly-rounded-divide-sqrt` build option, which
  `SilkRuntime.BuildProgram` does not pass.

## Planned commits

1. `docs: plan Silk device backward kernels in TODO.md (#900)` — this file.
2. `feat(silk): add backward kernels (gemm nt/tn, relu backward, col sum, conv2d backward, adam)` — kernels + launcher + `KernelNames` + catalog + SILK.md §11.
3. `test(silk): kernel-level parity for the Silk backward kernels` — `SilkBackwardKernelTests`.
4. `feat(silk): linear classifier device backward with device Adam` — Phase 2 code.
5. `test: pin GpuTextLinearClassifier against the CPU TextClassifierModel` — Phase 2 tests.
6. `refactor(samples): move ConvTextClassifierModel to Nivara.Samples` — Phase 4 move + fixture oracle swap.
7. `feat(silk): conv classifier device backward with device Adam` — Phase 3 code.
8. `test: pin conv classifier device gradients against reverse autograd` — fixture rewrite.
9. `feat(samples): OpenCL fallback and disposal for the classifier CLI` — Phase 4 sample hardening.
10. `docs: Silk backward kernel notes` — remaining doc updates + pins.
11. `chore: remove TODO.md — plan executed` — after G2 clears.

Steps 3/5/8 may fold into their feature commits when the feature and its fixture land
together — the unit of work is "code + the tests that pin it".

## GitHub issues log

- [ ] #901 — measurement/benchmarks for the classifier paths (human-run; referenced, not created here).
- [ ] #NNN — device-resident embedding table with CAS scatter backward (created during execution).
