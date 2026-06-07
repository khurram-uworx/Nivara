# Known Issues & Technical Debt

This file documents real bugs, correctness risks, and technical debt in the
current codebase. It is distinct from `TENSORS-GAPS.md`, which catalogs
forward-looking feature gaps and deferred design items.

---

## 1. `ArrayPool.Return` omits `clearArray: true`

**Severity:** Low

**Where:** `src/Nivara/NivaraColumn.cs` — all 9 `ArrayPool<T>.Shared.Return` calls.

**What:**
All pool returns use the default `clearArray: false`. For the current numeric
types (`int`, `float`, `double`, `long`, etc.) this is safe because no
sensitive data is retained. However, if `T` is ever a reference type or
struct with references, the returned array will still reference those objects,
preventing GC collection and potentially leaking sensitive data.

**When to fix:**
- If `T` can be a reference type in any of the pooled paths.
- If the arrays contain sensitive PII or security-critical data.
- If memory-pressure profiling shows that pool retention of large arrays
  delays collection of referenced objects.

**Fix:**
Add `clearArray: true` to `ArrayPool<T>.Shared.Return` calls, or add a
conditional that clears only for reference types:

```csharp
ArrayPool<T>.Shared.Return(pooledDataBuffer, clearArray: !typeof(T).IsValueType);
```

---

## 2. Flattened cache can become stale after mutation

**Severity:** Low (no known code path triggers this today)

**Where:** `src/Nivara/Storage/TensorStorage.cs` — `GetFlattenedSpan()`

**What:**
The flattened buffer is cached after the first call. If internal mutation
occurs after the cache is populated, the cached buffer is stale. Currently,
`NivaraColumn` is immutable at the public API boundary, so no production
code path mutates storage after construction. However, internal paths
could violate this in the future.

**Mitigation:**
- `AsTensorSpanIfNoNulls()` returns `ReadOnlyTensorSpan<T>`, preventing
  mutation through that path.
- No internal code currently mutates `TensorStorage` after construction.

**Possible fix:**
Add a `_flattenedVersion` counter incremented on mutation and checked on
cache read.

---

## 3. RowNorms uses column-major iteration per row

**Severity:** Medium (performance only, correctness is fine)

**Where:** `src/Nivara/NivaraFrame.cs` — `RowNorms<T>()`

**What:**
`RowNorms` iterates every column for each row, extracting one element at a
time via the column interface. For frames with many columns, this is
significantly slower than extracting columns to a flat buffer once and
iterating the flat buffer row-by-row.

**Details:**
- `N` rows × `M` columns = `O(N * M)` column interface calls.
- Each call goes through storage indirection.
- No `TensorPrimitives.Norm` batch optimization (requires row-major layout).

**Workaround:**
Extract all numeric columns to a 2D tensor via `frame.ToTensor<T>()` and
compute norms manually using `TensorPrimitives.Norm` on each row slice.

**Possible fix:**
Use a row-major intermediate buffer (`T[N * M]`), apply
`TensorPrimitives.Norm` over each row slice, and pool the buffer for
large frames.

---

## 4. NullMask length checks are inconsistent

**Severity:** Low (no crash in practice, but fragile)

**Where:** Various `src/Nivara/NivaraColumn.cs` paths.

**What:**
`ReadOnlyMemory<bool>?` has `HasValue == true` even when the memory is
empty (`.Length == 0`). Some slicing paths guard with `.Length > 0`, some
don't. The pattern `if (!nullMask.IsEmpty)` works correctly when the mask
was constructed from a non-empty array, but is fragile if a zero-length
mask is passed explicitly.

**Example pattern:**
```csharp
// Safe: checks Length > 0
if (nullMask.HasValue && nullMask.Value.Length > 0) { ... }

// Fragile: IsEmpty returns true for zero-length memory,
// but only if it was created from a non-empty array originally
if (!nullMask.IsEmpty) { ... }
```

---

## 5. Legacy series-level tensor helpers may still throw on nulls

**Severity:** Low (documented behavior, not a regression)

**Where:** `src/Nivara/Tensors/TensorExtensions.cs`
- `DotProduct<T>()`
- `Norm<T>()`
- `SumTensor<T>()`
- `AddTensor<T>()`
- `MultiplyTensor<T>()`

**What:**
Frame-level batch operations (`Dot`, `CosineSimilarity`, `ColumnNorms`,
`RowNorms`) propagate nulls via mask-OR. The individual series-level
helpers above still throw `InvalidOperationException` on nulls.
This is inconsistent from a user perspective.

**Mitigation:**
Documented in XML comments. Users should prefer frame-level batch ops
when null handling is required.

---

## 6. ColumnStorageFactory unconditionally copies spans

**Severity:** Low (performance only)

**Where:** `src/Nivara/Storage/ColumnStorageFactory.cs` — `Create<T>(ReadOnlySpan<T>, ...)`

**What:**
The factory always copies `values.ToArray()` even when the caller already
owns an array (e.g., a rented buffer). This means the pre-copy work done
in `NivaraColumn`'s vectorized paths (`ArrayPool.Rent` → copy elements →
`ColumnStorageFactory.Create`) creates a second copy.

**Details:**
- Pooled data buffer → `result.AsSpan()` → `ColumnStorageFactory.Create` → `values.ToArray()` = double copy.
- The pooled buffer is returned, the copied array is kept as storage.
- This is correct but wastes the pre-copy effort.

---

## 7. `DiagnosticsTracker` thread safety

**Severity:** Low (current usage is single-threaded)

**Where:** `src/Nivara/Diagnostics/DiagnosticsTracker.cs`

**What:**
`DiagnosticsTracker` is not thread-safe. Concurrent `RecordOperation`
calls could corrupt the internal list. Not a problem today because
Nivara operations are synchronous and single-threaded, but worth
noting if parallel query execution is added.
