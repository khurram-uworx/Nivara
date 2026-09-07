# Plan: Qwen-fast P0-2 — eliminate the redundant weight copy in single-row matmul

Branch: `khurram/qwen-perf` (off `main`). Feeds the O(qwen-fast) effort; executes
item **P0-2** from `docs/QWEN-PERF.md`. Harness-first: measure before and after so
progress is objective (README-methodology: idle machine, `--runs 3` medians).

## Problem

Every decode matmul in the Qwen path is `aRows == 1` against weights stored
row-major `[out, in]` (`bTransposed: true`). `TensorsHelper.MultiplyCore*` rents
a buffer and **copies the entire weight matrix** (`b.CopyTo(bT)`,
`src/Nivara/Tensors/TensorsHelper.cs:150-151`) before dotting. For `aRows == 1`
and `bTransposed` the copy is a pure identity copy — the weight rows *already are*
the rows we dot against.

Traffic cost per decoded token (F32): LM head `[1,896]·[151936,896]ᵀ` = a fresh
**~544 MB allocation + copy every token** (ArrayPool's poolable max is ~1M
elements < 136M → fresh array, GC churn), FFN ≈ **1.25 GB/token** of redundant
copies, attention QKV+O ≈ 172 MB/token. Effectively doubles decode memory
traffic (per-token weight reads ~2 GB → ~1 GB once the copies are gone).

Root cause: the transposed-B preparation in `MultiplyCoreFloat` /
`MultiplyCoreDouble` / `MultiplyCoreGeneric` never considers that
`aRows == 1 && bTransposed` needs no preparation at all.

## Proposed changes

### A. Kernel fast path — `src/Nivara/Tensors/TensorsHelper.cs`

In `MultiplyCore<T>`, after the existing argument validation and **before** the
`typeof(T)` dispatch, add:

```csharp
if (bTransposed && aRows == 1)
{
    // Single-row mat-vec (BLAS2 GEMV): weights are row-major [out, in], so each
    // output column is a dot against one weight row. No rent/copy — the copy
    // was a pure identity at aRows == 1 and doubled decode memory traffic
    // (LM head alone: ~544 MB fresh rent + copy per token). Reuses the same
    // per-type Dot as the row kernels, so numerics stay bit-identical.
    // Deliberately TensorPrimitives-based, not hand-rolled intrinsics —
    // learn.microsoft.com/dotnet/standard/simd: "reach for the existing
    // higher-level APIs first". Swap target for this whole kernel remains
    // Tensor.MatrixMultiply once the BCL ships it (dotnet/runtime#95863,
    // BLAS epic dotnet/runtime#93286).
    var aRow = a.Slice(0, aCols);
    Span<T> res = result.AsSpan(0, bCols);
    for (int j = 0; j < bCols; j++)
        res[j] = DotFor<T>(aRow, b.Slice(j * aCols, aCols));
    return;
}
```

with a private helper mirroring the existing row-kernel dispatch:

```csharp
static T DotFor<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y) where T : struct, INumber<T>
    => typeof(T) == typeof(float) || typeof(T) == typeof(double)
        ? TensorPrimitives.Dot(x, y)   // matches MultiplyRowFloat / MultiplyRowDouble
        : WidenPrimitives.Dot(x, y);   // matches MultiplyRowScalar (widens Half/BF16 → float)
```

Bit-identical by construction: same `Dot` implementation, same elements, same
accumulation order as the multi-row paths. `ShouldParallelize` already gates at
`aRows >= 4`, so single-row never parallelized — no behavior change there.
Every caller passes a fresh result array, so reading `b` in place is safe
(document the invariant in the comment). All `GradKernels.MatMulTransposedB`
callers (reverse-mode `MatMulTransposedB` op, forward-mode ops, `Linear<T>`,
LM head, FFN) inherit the fast path automatically.

### B. Harness scenarios — `tests/Nivara.PerformanceTests/Program.cs`

Add `RunQwenDecodeMatMulScenarios()` registered from `RegisterScenarios()`.
Qwen2.5-0.5B shapes (hidden 896, vocab 151,936, FFN 4864, GQA heads 14/2):

| Scenario | Shape (aRows=1, aCols @ bCols) | What it isolates |
|---|---|---|
| `Qwen Decode LM head [1x896 @ 151936x896]` | 896 @ 151,936 | Raw `GradKernels.MatMulTransposedB`, preallocated result. B/op pre-fix ≈ 544 MB/op → post-fix ≈ 0. **The P0-2 signal.** |
| `Qwen Decode FFN [1x896 @ 4864x896] x3` | 896 @ 4864 | Three FFN matmuls per token |
| `Qwen Decode QKV [1x896 @ 2688x896]` | 896 @ 2688 | Fused-width QKV projection |
| `Qwen Decode Linear<float> [1x896 -> 2688]` | op-level | `Linear<T>.Forward` single-row — captures per-op boxing allocs (neighbors P1-#4) |

Preallocated result buffers (fixes the alignment-noise caveat from the SIMD
guidance); small iteration counts for the LM-head row. Follow the `Run(...)`
pattern. Record and commit `qwen-fast-baseline.json` (see §Verification).

### C. E2E synthetic weights — `samples/NivaraInference`

Add `--synthetic-weights` for qwen: fabricate a tensor dict keyed by the same
names `LlamaForCausalLM.LoadModel` expects, with Qwen shapes from `LlamaConfig`
and deterministic pseudo-random values — so `qwen benchmark: KV-cache decode`
(and `RunBenchmark`) runs **without the model file** (timing is shape-driven).
Correctness stays gated on the real fixtures (`qwen_tool_logits_py.bin`, etc.).
Flag must be rejected for non-qwen models and must not affect real-weight loads.

### D. Unit tests — `tests/Nivara.Tests`

- `TensorsHelperTests`:
  - Extend `CheckMatMulShapes<T>` with a `bTransposed: true` variant against a
    new `ReferenceMatMulTransposedB`, including single-row shapes `(1,1,1)`,
    `(1,5,1)`, `(1,64,48)`, `(1,128,256)`.
  - **Bit-exact** parity (float/double, 0.0 tolerance): fast path vs. explicit
    `TensorPrimitives.Dot` row loop (same implementation — required because Dot
    may call arch-specific code; tolerance tests cover cross-arch safety).
  - Cross-path consistency: `aRows == 2` transposed run, row 0 must bit-equal
    the `aRows == 1` fast path.
  - **Alloc regression guard**: `float`, preallocated result,
    `GC.GetAllocatedBytesForCurrentThread` delta ≈ 0 across the fast-path call
    (locks "no 544 MB rent" against future regressions).
- `WidenPrimitivesPhase1Tests`: transposed single-row `Half`/`BFloat16` parity
  (guards the `WidenPrimitives.Dot` dispatch).
- AutoDiff level: `MatMulTransposedB` op `aRows == 1` vs. row 0 of `aRows == 2`
  (tolerance).

### E. Docs

- `CHANGELOG.md`: P0-2 entry.
- `tests/Nivara.PerformanceTests/README.md`: Results table (Prev/Current/Δ%,
  B/op, gen0/op) + scenario rows.
- `docs/QWEN-PERF.md`: append a status note marking P0-2 implemented with the
  measured numbers (review doc stays otherwise untouched) — confirm with human
  before editing.

## Verification

1. `dotnet build Nivara.slnx` after each commit (ask before running).
2. **Baseline** (before the kernel change, harness committed):
   `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json qwen-fast-baseline.json --runs 3`
   → expect LM-head B/op ≈ 5.4e8. Commit the JSON beside `baseline-v140.json`.
3. **Post-fix**: `--compare qwen-fast-baseline.json --runs 3` → LM-head B/op ≈ 0,
   ops/s up; keep the no-regression gate green (≤ 1.01× B/op, ≥ 90% ops/s).
4. **E2E**: `dotnet run --project samples/NivaraInference -c Release -- qwen --synthetic-weights benchmark`
   → KV-cached vs full-forward ms/token before/after.
5. `dotnet test` — only after human confirmation (AGENTS.md: ask first).

## Planned commits

1. `docs: plan Qwen-fast P0-2 + measurement harness in TODO.md`
2. `perf: add Qwen decode single-row matmul scenarios to PerformanceTests`
3. `perf: add NivaraInference qwen --synthetic-weights decode benchmark mode`
4. `perf: record qwen-fast baseline (tests/Nivara.PerformanceTests/qwen-fast-baseline.json)`
5. `core: kill redundant weight copy in single-row transposed-B matmul`
6. `test: pin single-row transposed-B matmul parity + alloc-free fast path`
7. `docs: record qwen-fast P0-2 results (perf README, CHANGELOG)`

## Blast radius

- `TensorsHelper.MultiplyCore<T>` — internal; reachable only via
  `GradKernels.MatMulTransposedB` / `MatMul` (reverse + forward AutoDiff ops),
  the null-aware `TensorsHelper.Multiply`, and tests/harness. All call sites
  allocate a fresh result array (no aliasing). Numerics must be bit-identical
  for float/double; `WidenPrimitives.Dot` dispatch preserves Half/BF16. No
  public API change. Downstream consumers of the single-row path: Qwen,
  SmolLM, MiniLM, DistilBERT samples + `Nivara.Samples` LLM pipelines.
- `tests/Nivara.PerformanceTests` — additive scenario rows only; gate behavior
  unchanged until baselines are re-recorded.
- `samples/NivaraInference` — additive `--synthetic-weights` flag only.
- Unit tests — additive.

## MS Learn verification (G1, folded in)

Performed during planning; no decisions surfaced, no plan changes needed:

- **`TensorPrimitives.Dot<T>`** is the documented BLAS1 `dot` equivalent —
  generic over numeric types, computes with **no temporary storage**, SIMD /
  architecture-accelerated ("may call into the underlying C runtime or employ
  architecture-specific instructions; exact results may differ between
  OS/arch"). -> fast path must reuse it (bit-exact tests compare Dot-vs-Dot;
  cross-arch safety via tolerance tests, which the plan does).
  https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives.dot
- **No `Tensor.MatrixMultiply` shipped.** The `System.Numerics.Tensors` API
  catalog at the repo's pinned version (11.0.0-preview.7.26381.103) exposes only
  *element-wise* `Tensor.Multiply`; dense matmul is still tracking
  dotnet/runtime#95863 (BLAS epic #93286). The code's "swap target" note stays
  accurate — no BCL matmul to switch to.
- **"Reach for existing higher-level APIs first… don't hand-roll what's already
  optimized and tested"** (Use SIMD and hardware intrinsics in .NET) — the fast
  path deliberately routes through the same `TensorPrimitives.Dot` /
  `WidenPrimitives.Dot` the row kernels use; no hand-rolled intrinsics.
- **"Benchmark to confirm the win… benchmark the input sizes your callers
  actually use"** (same guide) — the harness measures at real Qwen shapes; B/op
  is the allocation-driven signal; preallocated result buffers neutralize the
  documented randomized-alignment noise; `--runs 3` medians on an idle machine
  are the documented control.
- Single-row mat-vec is BLAS2 **GEMV**, memory-bound: the optimum reads each
  weight exactly once; the fast path achieves it (previously ~2× due to copy).

## GitHub issues log

- (none at plan time; created during execution as deferred work is found)
- Related tracked work referenced: #384 (qkvBias), #387/#391 (BF16 SIMD),
  #388 (fused BF16→F32 read), #390 (GGUF backend).

## Open items

- [ ] Confirm whether `docs/QWEN-PERF.md` gets a status note (see §E) before
      editing it.
- [ ] Ask before running `dotnet test` / the `--runs 3` baseline harness.