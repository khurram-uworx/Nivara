# Tensor Back-off — Execution Plan

## Decisions (per user, 2026-06-08)

| Question | Decision |
|----------|----------|
| T3 target namespace | Leave in current namespace, mark `[Obsolete]` with BCL redirect |
| T3 test strategy | Delete tests for deprecated APIs |
| T2 scope | Full: column + frame row-span APIs |
| T1 audit scope | Public + internal APIs |
| T2/T4/T5 ordering | Merge into single pass |
| MLNet batch tensor methods | Leave in Extensions, no change |
| AutoDiff tensor path | Leave as-is, no change |

---

## Phase 0: Audit (T1)

**Goal:** Catalog every public+internal tensor interop method with null policy, copy semantics, and recommendation.

**Files to analyze:**
- `src/Nivara/Tensors/TensorInteropExtensions.cs` — 11 extension methods
- `src/Nivara/Tensors/TensorExtensions.cs` — 7 extension methods
- `src/Nivara/NivaraColumn.cs` — `ToTensor()`, `ToTensor(T)`, `AsSpan()`, `AsWritableSpan()`
- `src/Nivara/NivaraSeries.cs` — `ToTensor()`, `ToTensor(T)`
- `src/Nivara/NivaraFrame.cs` — `ToTensors<T>()`, `Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms`
- `src/Nivara/IColumnStorage.cs` — `AsSpan()`, `AsWritableSpan()`
- `src/Nivara/Storage/TensorStorage.cs` — `AsTensorSpanIfNoNulls`, `GetFlattenedSpan`, `GetFlattenedNullMask`
- `src/Nivara/Storage/MemoryStorage.cs` — `AsSpan()`, `AsWritableSpan()`

**Output:** Categorize each method as:
- **Keep** — interop that aligns with new direction
- **Redesign** — candidates for Phase 2
- **Deprecate** — mark `[Obsolete]` in Phase 1
- **Delete** — internal dead code
- **Expose** — internal null-mask methods to make public

**Depends on:** Nothing

---

## Phase 1: Mark Obsolete + Delete Tests (T3)

### 1a. Add `[Obsolete]` to math wrappers

| Method | File | Line | Obsolete Message |
|--------|------|------|-----------------|
| `NivaraFrame.Dot<T>()` | `NivaraFrame.cs` | 245 | `Use TensorPrimitives.Dot<T>(ReadOnlySpan<T>, ReadOnlySpan<T>) instead` |
| `NivaraFrame.CosineSimilarity<T>()` | `NivaraFrame.cs` | 292 | `Use TensorPrimitives.CosineSimilarity<T>(ReadOnlySpan<T>, ReadOnlySpan<T>) instead` |
| `NivaraFrame.ColumnNorms<T>()` | `NivaraFrame.cs` | 352 | `Use TensorPrimitives.Norm<T>(ReadOnlySpan<T>) per column instead` |
| `NivaraFrame.RowNorms<T>()` | `NivaraFrame.cs` | 392 | `Use TensorPrimitives.Norm<T>(ReadOnlySpan<T>) per row instead` |
| `TensorExtensions.DotProduct<T>()` | `TensorExtensions.cs` | 198 | `Use TensorPrimitives.Dot<T>(ReadOnlySpan<T>, ReadOnlySpan<T>) instead` |
| `TensorExtensions.Norm<T>()` | `TensorExtensions.cs` | 259 | `Use TensorPrimitives.Norm<T>(ReadOnlySpan<T>) instead` |
| `TensorExtensions.SumTensor<T>()` | `TensorExtensions.cs` | 144 | `Use TensorPrimitives.Sum<T>(ReadOnlySpan<T>) instead` |
| `TensorExtensions.AddTensor<T>()` | `TensorExtensions.cs` | 21 | `Use TensorPrimitives.Add<T>(ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>) instead` |
| `TensorExtensions.MultiplyTensor<T>()` | `TensorExtensions.cs` | 83 | `Use TensorPrimitives.Multiply<T>(ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>) instead` |
| `TensorExtensions.TransformTensor<T>()` | `TensorExtensions.cs` | 321 | `Use TensorPrimitives with manual mapping instead` |
| `TensorExtensions.MatrixMultiply<T>()` | `TensorExtensions.cs` | 362 | `Use BCL matrix multiply APIs instead` |

### 1b. Delete obsolete tests

- `tests/Nivara.Tests/Tensors/TensorInteropTests.cs` — remove all tests calling `Dot`, `CosineSimilarity`, `ColumnNorms`, `RowNorms`
- `tests/Nivara.Tests/NivaraSeriesIsValidTests.cs` — remove all tests calling `DotProduct`, `Norm`, `SumTensor`, `AddTensor`, `MultiplyTensor`, `TransformTensor`
- `tests/Nivara.Tests/Diagnostics/DiagnosticsTests.cs` — remove 4 test entries (lines 171, 195, 217, 239)

**Depends on:** Phase 0

---

## Phase 2: Redesign Interop + Span Helpers + Null Mask (T2+T4+T5 merged)

### 2a. `NivaraColumn<T>` additions (`NivaraColumn.cs`)

```csharp
// Zero-copy when null-free
public bool TryGetSpan(out ReadOnlySpan<T> span)

// Explicit fill with user-provided destination + optional mask
public void CopyTo(Span<T> destination, T fillValue, Span<bool>? maskDestination = null)

// Expose null mask as span
public bool TryGetNullMask(out ReadOnlySpan<bool> mask)
```

`TryGetSpan` — returns `true` + zero-copy `ReadOnlySpan<T>` when `!HasNulls`; `false` + empty span when nulls exist.

`CopyTo` — copies all data to destination, fills null positions with `fillValue`. If `maskDestination` is provided, writes `true` at null positions.

`TryGetNullMask` — returns `true` + mask span when `HasNulls`; `false` + empty span when no nulls.

### 2b. `NivaraFrame<T>` additions (`NivaraFrame.cs`)

```csharp
// Row-major span over pooled buffer (temporary, use promptly)
public bool TryGetRowMajorSpan<T>(out ReadOnlySpan<T> span)

// Explicit copy to user-provided span
public void CopyToRowMajor<T>(Span<T> destination, T fillValue)
```

`TryGetRowMajorSpan` — allocates a temporary buffer, fills row-major. `true` if no nulls. Document as pooled/temporary.

`CopyToRowMajor` — fills a user-provided span with row-major data. Nulls filled with `fillValue`.

### 2c. Update `IColumnStorage<T>` interface (`IColumnStorage.cs`)

```csharp
internal bool TryGetSpan(out ReadOnlySpan<T> span);
```

### 2d. Storage implementations

- `TensorStorage.cs` — implement `TryGetSpan` using existing `GetFlattenedSpan()` (zero-copy) when no nulls
- `MemoryStorage.cs` — implement `TryGetSpan` using existing `data.Span` (zero-copy) when no nulls

### 2e. `TensorInteropExtensions` additions

```csharp
// New: frame from ReadOnlyTensorSpan with column names
public static NivaraFrame FromTensorSpan<T>(ReadOnlyTensorSpan<T> span, string[]? columnNames = null)

// New: column convenience wrapper
public static ReadOnlySpan<T> AsSpanOrDefault<T>(this NivaraColumn<T> column, ReadOnlySpan<T> fallback)
```

### 2f. Tests for new APIs

Add tests covering:
- `TryGetSpan` — null-free column returns true + span; column with nulls returns false
- `CopyTo` — destination filled correctly, null mask written correctly
- `TryGetNullMask` — mask span matches column nulls
- `TryGetRowMajorSpan` — frame data in row-major order
- `CopyToRowMajor` — correct fill with null replacement
- `FromTensorSpan` — round-trip consistency
- Edge cases: empty columns, all-null columns, disposed column throws

**Depends on:** Phase 1 (avoids merge conflicts with deleted test code)

---

## Phase 3: Clean up TENSORS-GAPS.md (T6)

| Item | Action |
|------|--------|
| P1 RowNorms optimization | Close — API now obsolete |
| P2 TopKDescending threshold | Keep — Nivara-specific |
| E1 Row-wise cosine similarity | Close — won't add |
| E2 Normalize helper | Keep — NivaraSeries op delegating to TensorPrimitives |
| E3 Dot vs DotProduct naming | Close — both obsolete |
| E4 Nullable tensor conversion | Re-evaluate after T2 (likely covered) |
| E5 Frame tensor conversion metadata | Re-evaluate after T2 |
| E6 Axis enum | Close — no tensor math APIs |
| E7 Collection expressions | Keep — Nivara ergonomics (separate work) |
| E8 Shape constructors | Re-evaluate — keep as interop |
| E9 TopKDescending result type | Keep — Nivara-specific |
| E10 Scalar tensor reducers | Close — being deprecated |
| T1 Benchmarks | Reframe — focus on interop paths |
| D1-D4 Design items | Close — not Nivara's concern |

**Depends on:** Phases 0-2

---

## Phase 4: Documentation (T7)

- XML doc comments on `[Obsolete]` methods — include `<see cref="TensorPrimitives.X"/>` redirects
- No migration guide needed (APIs stay, just marked obsolete)
- No changes to EXAMPLES.md needed (already revised)

**Depends on:** Phases 1-3

---

## File Change Summary

| File | Phase | Change |
|------|-------|--------|
| `src/Nivara/NivaraColumn.cs` | 2 | Add `TryGetSpan`, `CopyTo`, `TryGetNullMask` |
| `src/Nivara/NivaraFrame.cs` | 1, 2 | `[Obsolete]` on 4 methods; add `TryGetRowMajorSpan`, `CopyToRowMajor` |
| `src/Nivara/NivaraSeries.cs` | 0, 2 | Audit; delegate to column — no structural change |
| `src/Nivara/Tensors/TensorInteropExtensions.cs` | 0, 2 | Add `FromTensorSpan`, `AsSpanOrDefault`; keep existing |
| `src/Nivara/Tensors/TensorExtensions.cs` | 1 | `[Obsolete]` on all 7 methods |
| `src/Nivara/IColumnStorage.cs` | 2 | Add `TryGetSpan` to interface |
| `src/Nivara/Storage/TensorStorage.cs` | 2 | Add `TryGetSpan` impl |
| `src/Nivara/Storage/MemoryStorage.cs` | 2 | Add `TryGetSpan` impl |
| `tests/Nivara.Tests/Tensors/TensorInteropTests.cs` | 1, 2 | Delete obsolete tests; add new interop tests |
| `tests/Nivara.Tests/NivaraSeriesIsValidTests.cs` | 1 | Delete obsolete test file |
| `tests/Nivara.Tests/Diagnostics/DiagnosticsTests.cs` | 1 | Delete 4 diagnostic test entries |
| `docs/TENSORS-GAPS.md` | 3 | Mark closed items |

## Execution Order

```
Phase 0 (T1: Audit) ──→ Phase 1 (T3: [Obsolete] + delete tests)
                            │
                            └──→ Phase 2 (T2+T4+T5: span redesign)
                                      │
                                      └──→ Phase 3 (T6: clean gaps doc)
                                                │
                                                └──→ Phase 4 (T7: docs)
```
