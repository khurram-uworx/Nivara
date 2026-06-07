# Tensor Gaps — What Remains After Phase 4

This document catalogs gaps, deferred items, and design decisions that remain
after Phases 0–4 of the tensor improvements plan (`TENSORS-IMP-PLAN.md`).
Items are grouped by severity and impact.

---

## Correctness Gaps

### G1. Legacy series-level DotProduct and Norm may still throw on nulls

**Where:** `TensorExtensions.DotProduct<T>()`, `TensorExtensions.Norm<T>()`
(currently individual-vector helpers, not yet null-propagating).

**Status:** These are series-level (not frame-level) helpers. The frame-level
`Dot`, `CosineSimilarity`, `ColumnNorms`, and `RowNorms` on `NivaraFrame`
propagate nulls via mask-OR. The individual helpers may still throw on nulls.

**Recommendation:** Audit and decide policy for individual (non-frame) tensor helpers.

---

### G2. Flattened cache can become stale after mutation

**Where:** `TensorStorage.GetFlattenedSpan()` caches the flattened buffer. If
internal mutation occurs after the cache is populated (e.g., via internal
`TensorSpan` access), the cached buffer is stale.

**Status:** The `AsTensorSpanIfNoNulls()` method now returns `ReadOnlyTensorSpan<T>`,
which is a safe read-only view. However, the flattened cache could theoretically
be invalidated by direct storage mutation.

**Recommendation:** Add a `_flattenedVersion` counter, incremented on mutation
and checked on cache read, and add an `InvalidateFlattenedCache()` internal
method to document and enforce the invalidation contract.

---

### G3. `NullMask?` slicing needs consistent `.Length > 0` checks

**Where:** Various `NivaraColumn.cs` paths.

**Status:** `ReadOnlyMemory<bool>?` has `HasValue == true` even when the memory
is empty (length 0). Some slicing paths check `.Length > 0`, some don't.

**Recommendation:** Audit all `nullMask` slicing sites and add consistent
`.Length > 0` guards.

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

### P3. BufferPool not used in frame-level row-wise temporary buffers

**Where:** `RowNorms` and any future row-wise batch ops.

**Status:** Row-wise temporary buffers use `new T[ColumnCount]` inside each
row's inner scope. For large column counts, pooling would help.

**Recommendation:** Apply `ArrayPool<T>.Shared.Rent` for row buffers when
`ColumnCount >= 1024`.

---

### P4. `ColumnStorageFactory.Create<T>(ReadOnlySpan<T>, ReadOnlyMemory<bool>?)`
always copies the span

**Where:** `ColumnStorageFactory.Create<T>()`.

**Status:** The overload always copies `values.ToArray()` to create storage. For
callers that already own a buffer (e.g., rented arrays), this is a double copy.

**Recommendation:** Add an internal overload accepting `T[]` with ownership
semantics to avoid the copy when the caller can relinquish ownership.

---

## API & Ergonomics Gaps

### E1. No row-wise batch cosine similarity on `NivaraFrame`

**Where:** `NivaraFrame`.

**Status:** `CosineSimilarity` operates column-wise. Row-wise similarity
requires writing a manual loop (as shown in `EXAMPLES.md` section 6).

**Recommendation:** Add `RowCosineSimilarity(NivaraSeries<T> query,
IColumn? labels = null)` using the same row-iteration pattern as `RowNorms`.

---

### E2. No `Normalize` fluent helper on `NivaraSeries`

**Where:** `NivaraSeries<T>`.

**Status:** Users must manually call `series.Norm()` and then `series.Divide(norm)`.

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

### E5. No `FromTensor` on `NivaraFrame`

**Where:** `TensorInterop.FromTensor` supports series and column. Frame
creation from a 2D tensor is not exposed.

**Recommendation:** Add `NivaraFrame.FromTensor<T>(Tensor<T> matrix,
string[]? rowNames = null, string[]? columnNames = null)` as an explicit
`[rows, columns]` conversion.

---

### E6. No `Axis` enum for row vs column orientation

**Where:** Frame-level batch operations.

**Status:** Column-wise and row-wise operations are separate named methods
(`Dot` vs row-wise not yet existent). No axis parameter.

**Recommendation:** Keep named methods for clarity. Add an `Axis` enum only
if the method count becomes unwieldy.

---

### E7. Collection expressions not supported for series/columns

**Details:** See `docs/TENSORS-SUGGESTIONS.md` for design notes. Requires
`[CollectionBuilder]` attribute and a builder type.

---

## Testing Gaps

### T1. No allocation-count benchmarks for tensor paths

**Status:** No benchmarks measure allocation pressure for repeated
`ToTensor()`, `FlattenTo`, or batch operation calls.

**Recommendation:** Add benchmarks for:
- Repeated `ColumnToTensor` — should allocate only on first call (flattened cache).
- `BatchDot` — allocation per column vs per batch.
- `RowNorms` — allocation per row vs column-major overhead.

---

### T2. No property-based tests for null propagation

**Status:** Tests are hand-written with fixed arrays. Property-based tests
(using NUnit parameterization, not FsCheck) would cover edge cases.

**Recommendation:** Add parameterized null-propagation tests:
- Random null positions in left operand.
- Random null positions in right operand.
- All-null operands.
- No-null operands.
- Single-element operands.

---

### T3. Diagnostics recording not validated in tests

**Status:** `OperationDiagnostics` is recorded but no tests assert that
the diagnostics contain the expected operation name and kernel type.

**Recommendation:** Add `Assert.That(tracker.LastOperation, Is.EqualTo("FrameDot"))`
style assertions to the ranking integration tests.

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
| Correctness | G1, G2, G3 | | |
| Performance | | P1, P2, P3, P4 | |
| API/Ergonomics | | E1, E2, E3 | E4, E5, E6, E7 |
| Testing | | T1, T2, T3 | |
| Design | | | D1, D2, D3, D4 |
