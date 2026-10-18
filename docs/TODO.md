# Plan: Issue #458 — CPU transformer gap (attribution, small-M GEMM)

## Problem

The Nivara CPU transformer path is ~6.1-6.5x slower than PyTorch at the BERT-scale forwards
documented in `samples/NivaraInference/README.md`. Two sub-gaps are open:

1. A *non-GEMM* cost: at Laya shapes (S=512, d=1024, 28 layers) the in-tree GEMM sustains
   66-80 GMAC/s while end-to-end is ~26 GMAC/s equivalent — ~3x of the gap is outside
   GEMM. The mechanism is unmeasured; `ModernBertEncoder` runs `GradKernels`,
   `LayerNormKernel`, `AttentionKernels`, and `TensorPrimitives` directly, so none of it is
   captured by `AutoDiffDiagnostics`.
2. A *small-M GEMM* inefficiency: at S=128 the in-tree path is 6-10x behind the probe-local
   `Blocked` leg on `distilbert qkv/o` and `fc1`, but the probe that measured it took
   best-of-5 per leg sequentially, which the README already flags as the #482 flakiness
   pattern. Single rows swing up to 6.5x run-to-run.

The GEMM *ceiling* at S=512 was measured and cleared (2414 ms projected, `AutoDiff`); the
`Blocked` leg did not win. Any fix must not regress that result.

## Selected scope

- Phase 1 — per-op attribution for the CPU inference path (instrument + report).
- Phase 2 — re-stabilise the small-M GEMM comparison (interleaved, report medians).
- Phase 3 — evaluate managed `MultiplyCore` fixes against **both** M=128 and M=512.

Deliberately deferred (tracked, not implemented here):
- Native BLAS leg / provider seam (see `docs/BLAS-MKL.md` §5-§7, issues #133/#444).
  Only worth doing once Phase 2's medians and Phase 3's managed ceiling are on record.

## Phase 1 — Per-op attribution

### Changes planned

1. Add `InferenceDiagnostics` mirroring `AutoDiffDiagnostics`:
   - file: `src/Nivara/Extensions/AI/InferenceDiagnostics.cs` (or `src/Nivara/AutoDiff/InferenceDiagnostics.cs`).
   - `Enabled` (default false), `Measure/Measure<T,TResult>` wrapping `DiagnosticsTracker`.
   - Gate by `Enabled` so disabled runs take the fast path (no lock acquisition).
2. Instrument `ModernBertEncoder`'s forward/inference layer loop with blocks:
   - attn norm (LayerNorm), Q/K/V projections, RoPE, QK^T, scale/apply-bounds, softmax,
     PV matmul, o-projection + residual, MLP norm, GeGLU (act/gate + down), final residual.
   - Each block records `operationType = "CPU-Encoder-..."`, `inputLength` = elements touched,
     kernel label via `KernelSelector.DetermineKernelType<T>(length)` where meaningful, else
     the fixed label the block genuinely represents.
3. Keep overhead visible: report instrumented total alongside uninstrumented forward and an
   explicit absent-ops list. Do not present shares as a % of the uninstrumented forward.
4. Expose from `--laya-forward`'s diagnostics pass.
5. Tests: no-op when disabled (zero tracker entries), plus any doc-snippet gates if added.

### Blast radius
- `ModernBertEncoder` forward only; no change to inference outputs or alloc contract.
- `DiagnosticsTracker` reuse; no new storage.
- Consumers: `LayaForwardProbe`, `NivaraInference` sample (diagnostics surface, off by default).

## Phase 2 — Re-stabilise small-M GEMM

### Changes planned

1. Rework `--cpu-gemm` small-M reporting to the `--mask` protocol:
   - 5 warmups per leg per shape, 30 interleaved rounds, first-measured leg alternates
   - each round, report per-leg median tick counts, ranges, win counts.
2. Add probe-local `AutoDiffNoRentCopy` leg (mirrors `MultiplyCore` but reads A in place)
   alongside `AutoDiff`, `BTranspose`, `Blocked`.
3. Report roll-up (citable) separately from per-row medians (with ranges and win counts).
4. Update `tests/Nivara.PerformanceTests/README.md` and `docs/BLAS-MKL.md` §4/§6 with the new
   protocol and the now-trusted numbers; revise the "6-10x ±6.5x swing" language.
5. No core test deltas; correctness is already gated (`maxAbs <= 1e-3`).

### Blast radius
- `tests/Nivara.PerformanceTests/CpuGemmProbe.cs` only (probe).
- Docs: README + BLAS-MKL.

## Phase 3 — Managed GEMM evaluation

Experiments run as probe-local legs first; only the winner touches `src/Nivara/Tensors/TensorsHelper.cs`.

1. **Serial threshold**: force `ShouldParallelize = false` for M=128; derive an M-based cut.
2. **Remove `RentCopy`**: read A in place (as `Blocked` does), confirm DP-truth parity.
3. **`clearArray` cost**: measure returning pooled buffers without clearing.
4. **Adaptive blocked path**: use blocked accumulator loop when M is below a documented
   cutoff; keep Dot-per-element loop above it.
5. Patch `TensorsHelper.MultiplyCore` for the winning variant; run perf gate and tensor tests.
6. Update `docs/BLAS-MKL.md`. If managed closes the small-M gap, the native bar rises.

### Exit criteria (both required)
- M=128 roll-up improves materially vs current `AutoDiff`, no correctness regression.
- S=512 roll-up stays within noise.

### Blast radius
- `TensorsHelper.MultiplyCore`, `MultiplyRowFloat/Double/Scalar`, `MultiplyCoreFloat/Double/Generic`,
  `ShouldParallelize`, `RentCopy`.
- Callers via `GradKernels.MatMul{,TransposedB}`, `Linear<T>`, BERT/ModernBERT/Laya inference,
  attention. Correctness gate is mandatory before commit.

## Verification

- Instrumented diagnostics run (`--laya-forward`) with absent-ops listed.
- `--cpu-gemm` interleaved small-M run; review medians + win counts.
- Phase 3: probe variants first, then targeted perf gate (`--compare`) and targeted tests
  `TensorsHelperTests`, `GradKernelsTests`. Full suite only with human confirmation.

## Outcome summary

- Phase 1 landed: `InferenceDiagnostics` now records the fused CPU inference path.
  Representative Laya pass (valid=512, real checkpoint, 28 layers, 226 ops):
  CPU-Encoder-GeGLU 54.5%, CPU-Encoder-Attention 24.3%, CPU-Encoder-QKV 14.0%,
  CPU-Encoder-OProj 5.2%; instrumented total 4968.6 ms vs forward 5461 ms.
  Norms/RoPE/residual are each under 1%.
- Phase 2 landed: `--cpu-gemm` now uses 5 warmups plus 20 interleaved rounds and reports
  medians, and adds a `NoRentCopy` leg. The `NoRentCopy` leg does not improve on
  `AutoDiff` (within run-to-run noise), so the hypothesised `RentCopy`/`clearArray`
  cost is not the 6x small-M gap; that gap is not reproduced under the interleaved protocol
  (single-row swings are run-to-run noise). `AutoDiff` vs `Blocked` is ~1% at both S=128
  and S=512, so there is no shape-regime case for porting the blocked kernel into core.
- Phase 3 (managed `MultiplyCore`): exit criteria not met (no material gain while preserving
  S=512). No edit to `TensorsHelper.cs`. `ShouldParallelize` and pooled-buffer handling
  stay as they are.
- Phase 4 (native BLAS, `docs/BLAS-MKL.md` §5-§7) remains deferred per the decision rule;
  the managed legs already sit within ~1% of each other, so native would now need to beat the
  interleaved `AutoDiff` median, not a stale best-of-5 row.

## Planned commits

Actual landed commits on `khurram/458`:
1. `docs: plan #458 CPU gap attribution and small-M GEMM in TODO.md`
2. `feat: add InferenceDiagnostics for fused CPU inference path`
3. `feat: instrument fused ModernBERT inference path for per-op attribution`
4. `perf: interleave --cpu-gemm small-M measurement and add no-rent-copy leg`
5. `test: cover InferenceDiagnostics disabled path`

## GitHub issues log

- [x] #458 — CPU transformer path gap (parent, this issue)
- (create additional tracked issues here as deferred work is discovered during execution)
