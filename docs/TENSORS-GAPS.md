# Tensor Gaps — What Remains

> **⚠️ Pre-decision draft.** Many items here propose tensor math APIs and
> optimizations that the project has since decided *not* to own. See
> [`TENSORS.md`](TENSORS.md) for the current direction. This document needs a
> full revision to remove items that conflict with the new scope.

This document catalogs future tensor performance work, API follow-ups, testing
gaps, and deferred design decisions. Items are grouped by severity and impact.

Current bugs, correctness risks, and immediate technical debt are tracked in
`KNOWN-ISSUES.md`.

---

## Performance Gaps

### P1. `RowNorms` uses column-major iteration; no batch TensorPrimitives kernel

**Where:** `NivaraFrame.RowNorms<T>()`.

**Status:** Row norms iterate columns for each row with a per-row `T[buffer]`.
This is correct but not SIMD-optimized despite `TensorPrimitives.Norm` being
available. The issue is extracting a row-span from column-major storage.

**Recommendation:** Use row-major re-layout (frame → `T[rows*cols]` → `Tensor<T>`,
then batch `TensorPrimitives.Norm` over individual row slices). Add a pooled
transpose buffer for large frames.

---

### P2. `TopKDescending` has no threshold-based optimization

**Where:** `NivaraSeries<T>.TopKDescending(int count)`.

**Status:** Sorts all values and takes top `count`. For small `count` relative
to series length, a partial-sort (heap-based) approach would be faster.

**Recommendation:** If `count <= series.Length / 10`, use a bounded min-heap.
Otherwise use the current full-sort approach. Benchmark first.

---

## API & Ergonomics Gaps

### E1. No row-wise batch cosine similarity on `NivaraFrame`

**Where:** `NivaraFrame`.

**Status:** `CosineSimilarity` operates column-wise. Row-wise similarity
requires writing a manual loop (as shown in `EXAMPLES.md` section 6).
That loop currently requires manual feature-column extraction, per-row temporary
vectors in simple implementations, and caller-owned null handling.

**Recommendation:** Add `RowCosineSimilarity(NivaraSeries<T> query,
IColumn? labels = null)` using the same row-iteration pattern as `RowNorms`.
It should propagate nulls through the result mask and avoid per-row allocations
where possible.

---

### E2. No `Normalize` fluent helper on `NivaraSeries`

**Where:** `NivaraSeries<T>`.

**Status:** Users must manually call `series.Norm()` and then apply the
reciprocal scale themselves. There is no direct series-level `Divide` helper.

**Recommendation:** Add `NivaraSeries<T>.Normalize()` returning a new normalized
series. Null positions in the input produce null in the result.

---

### E3. `DotProduct` / `Norm` naming inconsistency

**Where:** `TensorExtensions.DotProduct(NivaraSeries)` vs
`NivaraFrame.Dot(NivaraSeries)`.

**Status:** Frame-level uses `Dot`, series-extension uses `DotProduct`.

**Recommendation:** Either rename to `Dot` everywhere or provide both names
with one delegating. Phase 4 deferred this as non-critical.

---

### E4. No nullable-preserving tensor conversion type

**Where:** `ToTensor()`, `ToTensor(T defaultValue)`.

**Status:** Nulls must be replaced with a sentinel value. There is no
result type that preserves nulls alongside tensor data.

**Recommendation:** Defer until there is a clear use case that requires it.
A `(Tensor<T> Data, Tensor<bool>? NullMask)` result type is one option.

---

### E5. Frame tensor conversion metadata is limited

**Where:** `TensorInterop.FromTensor<T>(Tensor<T>, string[]?)`.

**Status:** Frame creation from a 2D tensor is exposed through
`TensorInterop.FromTensor<T>(Tensor<T>, string[]?)`, with generated column
names when none are supplied. The API does not carry row labels, nullable
metadata, or an explicit conversion result object.

**Recommendation:** If ML workflows need richer metadata, add an explicit
frame conversion overload or result type, such as
`FromTensor<T>(Tensor<T> matrix, string[]? columnNames = null,
object[]? rowLabels = null)`. Keep the `[rows, columns]` convention explicit.

---

### E6. No `Axis` enum for row vs column orientation

**Where:** Frame-level batch operations.

**Status:** Column-wise and row-wise operations are separate named methods
(`Dot`, `CosineSimilarity`, `ColumnNorms`, and `RowNorms`). No axis parameter.

**Recommendation:** Keep named methods for clarity. Add an `Axis` enum only
if the method count becomes unwieldy.

---

### E7. Collection expressions not supported for series/columns

**Status:** `NivaraSeries<float> s = [1f, 2f, 3f]` is not supported.

**Recommendation:** Add `[CollectionBuilder]` support only if it can preserve
explicit null policy. The likely shape is a `NivaraSeriesBuilder` with an
accessible static method that accepts a final `ReadOnlySpan<T>` parameter.

---

### E8. Shape-oriented frame constructors are not available

**Where:** `NivaraFrame`.

**Status:** Tensor and frame conversion APIs exist, but there are no ergonomic
constructors for common embedding-table or matrix-ingestion shapes.

**Recommendation:** Consider explicit ingestion helpers such as
`NivaraFrame.FromRows<T>(IEnumerable<(string Label, T[] Vector)> rows)` and
`NivaraFrame.FromMatrix<T>(Tensor<T> matrix, string[]? rowLabels,
string[]? columnNames)`. Keep these scoped to data ingestion so Nivara does not
blur into a general tensor matrix library.

---

### E9. `TopKDescending` has no named result type

**Where:** `NivaraSeries<T>.TopKDescending(int count)`.

**Status:** Ranking results are currently represented without a dedicated
result type.

**Recommendation:** If ranking APIs grow, introduce a named result type such as
`RankedValue<T>` carrying position, label, score, and null state. This would
preserve Nivara's explicit null semantics better than tuple-only results.

---

### E10. Series scalar tensor reducers have throw-on-null semantics

**Where:** `TensorExtensions.DotProduct<T>()`, `TensorExtensions.Norm<T>()`,
`TensorExtensions.SumTensor<T>()`.

**Status:** Scalar tensor reducers return `T`, so null-containing series throw
instead of returning a nullable or masked result. Frame-level batch operations
such as `Dot`, `CosineSimilarity`, `ColumnNorms`, and `RowNorms` can preserve
nulls in their result masks.

**Recommendation:** Keep the current explicit throw behavior unless a concrete
workflow needs nullable scalar reducer results. If needed, add separate methods
or result types rather than changing the existing scalar-returning contract.

---

## Testing Gaps

### T1. No allocation-count benchmarks for tensor paths

**Status:** No benchmarks measure allocation pressure for repeated
`ToTensor()`, `FlattenTo`, or batch operation calls.

**Recommendation:** Add benchmarks for:
- Repeated column flattening and tensor conversion paths.
- `BatchDot` — allocation per column vs per batch.
- `RowNorms` — allocation per row vs column-major overhead.

---

## Deferred Design Items

### D1. Broadcasting

Adding implicit broadcasting to arithmetic or comparison operations risks
silent misalignment bugs. Nivara should remain explicit about shapes until
a concrete user workflow demands broadcasting with clear semantics.

### D2. Operator overloading (+, *, etc.)

Arithmetic operators would need to handle null masks, label alignment, and
type promotion. Named methods (`Add`, `Multiply`) are safer until the
semantics are stable and benchmarked.

### D3. BLAS-level matrix multiplication

The current `MatrixMultiply` uses a triple-nested loop. A production-quality
implementation would use `TensorPrimitives.Dot` per row×column pair, batched
BLAS calls, or hardware-accelerated paths. This belongs in `Nivara.Extensions`
if added.

### D4. NaN-as-null semantics

Semantically separate from Nivara's explicit boolean null masks. Not
compatible with the current design. If users need NaN-based skipping in
float/double workloads, it should be an explicit opt-in policy, not the
default.

---

## Summary

| Area | Critical | Important | Nice-to-have |
|------|----------|-----------|-------------|
| Correctness | | | |
| Performance | | P1, P2 | |
| API/Ergonomics | | E1, E2, E3 | E4, E5, E6, E7, E8, E9, E10 |
| Testing | | T1 | |
| Design | | | D1, D2, D3, D4 |
