# T1 Audit — Tensor Interop Surface Catalog

Categorized by disposition.

---

## KEEP — Public interop that aligns with new direction

| API | File | Line | Null Policy | Copy? | Notes |
|-----|------|------|-------------|-------|-------|
| `NivaraColumn<T>.ToTensor(T)` | `NivaraColumn.cs:336` | Fills nulls | Copy | Keep — explicit fill with defaultValue is the public fallback for CopyTo |
| `NivaraSeries<T>.ToTensor(T)` | `NivaraSeries.cs:382` | Fills nulls | Copy | Keep — delegates to column |
| `TensorInteropExtensions.FromTensor<T>(Tensor<T>)` | `TensorInteropExtensions.cs:54` | No nulls in result | Copy | Keep — primary ingress from BCL 1D tensors to series |
| `TensorInteropExtensions.FromTensor<T>(Tensor<T>, string[]?)` | `TensorInteropExtensions.cs:222` | No nulls in result | Copy | Keep — primary ingress from BCL 2D tensors to frame |
| `TensorInteropExtensions.FlattenFromTensor<T>(Tensor<T>)` | `TensorInteropExtensions.cs:278` | No nulls | Copy | Keep — shape interop, not math |
| `TensorInteropExtensions.FlattenFromTensorSpan<T>(ReadOnlyTensorSpan<T>)` | `TensorInteropExtensions.cs:307` | No nulls | Copy | Keep — shape interop, not math |
| `TensorInteropExtensions.ReshapeToTensor<T>(NivaraSeries, int[])` | `TensorInteropExtensions.cs:335` | Throws on nulls | Copy | Keep — shape interop, not math |
| `TensorInteropExtensions.ToBatchTensors<T>(IEnumerable<NivaraSeries>)` | `TensorInteropExtensions.cs:384` | Throws on nulls | Copy | Keep — batch interop |
| `TensorInteropExtensions.FromBatchTensors<T>(Tensor<T>[])` | `TensorInteropExtensions.cs:399` | No nulls in result | Copy | Keep — batch interop |
| `NivaraColumn<T>.NullCount` | `NivaraColumn.cs:2358` | N/A | N/A | Keep — null info |
| `NivaraColumn<T>.GetNullIndices()` | `NivaraColumn.cs:2382` | N/A | N/A | Keep — null info |
| `NivaraColumn<T>.FillNull(T)` | `NivaraColumn.cs:2407` | N/A | Copy | Keep — Nivara value-add |
| `NivaraColumn<T>.CreateFromNullable(Array)` | `NivaraColumn.cs:196` | N/A | Copy | Keep — Nivara value-add |

---

## REDESIGN — Add span-based accessors in Phase 2

| API | File | Line | Null Policy | Current | Target |
|-----|------|------|-------------|---------|--------|
| `NivaraColumn<T>.ToTensor()` | `NivaraColumn.cs:319` | Throws on nulls | Copy → Tensor | Replace with `TryGetSpan` for zero-copy path |
| `NivaraSeries<T>.ToTensor()` | `NivaraSeries.cs:372` | Throws on nulls | Copy → Tensor | Keep `ToTensor()`, add `TryGetSpan` delegating to column |
| `TensorInteropExtensions.ToTensorSpan<T>()` | `TensorInteropExtensions.cs:92` | Throws on nulls | Copy → ReadOnlyTensorSpan | Keep as copy-based span bridge. Add `TryGetSpan` on column for zero-copy |
| `TensorInteropExtensions.FromTensorSpan<T>()` | `TensorInteropExtensions.cs:128` | No nulls | Copy | Keep — ingress from ReadOnlyTensorSpan |
| `IColumnStorage<T>.AsSpan()` | `IColumnStorage.cs:62` | No guard | View (internal) | Add `TryGetSpan(out ReadOnlySpan<T>)` to interface |
| `TensorStorage<T>.GetFlattenedSpan()` | `TensorStorage.cs:200` | No guard | View (cached) | Wrap behind `TryGetSpan` |
| `MemoryStorage<T>.Data.Span` | `MemoryStorage.cs:167` | No guard | View | Wrap behind `TryGetSpan` |
| `NivaraColumn<T>.AsSpan()` | `NivaraColumn.cs:2840` | No guard | View (internal) | Add public `TryGetSpan` |

**Phase 2 additions:**
- `NivaraColumn<T>.TryGetSpan(out ReadOnlySpan<T>)` — public zero-copy when no nulls
- `NivaraColumn<T>.CopyTo(Span<T>, T, Span<bool>?)` — explicit fill
- `NivaraColumn<T>.TryGetNullMask(out ReadOnlySpan<bool>)` — expose mask
- `NivaraFrame.TryGetRowMajorSpan<T>(out ReadOnlySpan<T>)` — pooled temp
- `NivaraFrame.CopyToRowMajor<T>(Span<T>, T)` — explicit fill
- `IColumnStorage<T>.TryGetSpan(out ReadOnlySpan<T>)` — interface contract
- `TensorInteropExtensions.FromTensorSpan<T>(ReadOnlyTensorSpan<T>, string[]?)` — 2D span to frame
- `TensorInteropExtensions.AsSpanOrDefault<T>(NivaraColumn<T>, ReadOnlySpan<T>)` — convenience

---

## DEPRECATE — Mark `[Obsolete]` in Phase 1

| API | File | Line | Null Policy | Redirect To |
|-----|------|------|-------------|-------------|
| `NivaraFrame.Dot<T>(NivaraSeries<T>)` | `NivaraFrame.cs:245` | Null → null result | `TensorPrimitives.Dot<T>` |
| `NivaraFrame.CosineSimilarity<T>(NivaraSeries<T>)` | `NivaraFrame.cs:292` | Null → null result | `TensorPrimitives.CosineSimilarity<T>` |
| `NivaraFrame.ColumnNorms<T>()` | `NivaraFrame.cs:352` | Null → null result | `TensorPrimitives.Norm<T>` per column |
| `NivaraFrame.RowNorms<T>()` | `NivaraFrame.cs:392` | Null → null result | `TensorPrimitives.Norm<T>` per row |
| `TensorExtensions.AddTensor<T>(NivaraSeries, NivaraSeries)` | `TensorExtensions.cs:21` | Null → nullable path | `TensorPrimitives.Add<T>` |
| `TensorExtensions.MultiplyTensor<T>(NivaraSeries, NivaraSeries)` | `TensorExtensions.cs:83` | Null → nullable path | `TensorPrimitives.Multiply<T>` |
| `TensorExtensions.SumTensor<T>(NivaraSeries)` | `TensorExtensions.cs:144` | Throws on nulls | `TensorPrimitives.Sum<T>` |
| `TensorExtensions.DotProduct<T>(NivaraSeries, NivaraSeries)` | `TensorExtensions.cs:198` | Throws on nulls | `TensorPrimitives.Dot<T>` |
| `TensorExtensions.Norm<T>(NivaraSeries)` | `TensorExtensions.cs:259` | Throws on nulls | `TensorPrimitives.Norm<T>` |
| `TensorExtensions.TransformTensor<T>(NivaraSeries, Func<T,T>)` | `TensorExtensions.cs:321` | Throws on nulls | Manual LINQ/loop |
| `TensorExtensions.MatrixMultiply<T>(NivaraFrame, NivaraFrame)` | `TensorExtensions.cs:362` | Throws via ToTensor | BCL matrix multiply |

---

## INTERNAL — No change needed

| API | File | Line | Notes |
|-----|------|------|-------|
| `TensorStorage<T>.AsTensorSpanIfNoNulls()` | `TensorStorage.cs:187` | Internal — used by internal kernels only. Keep. |
| `TensorStorage<T>.GetFlattenedNullMask()` | `TensorStorage.cs:216` | Internal — wrapped by `NullMask` property. Keep. |
| `IColumnStorage<T>.AsWritableSpan()` | `IColumnStorage.cs:70` | Internal — creates copy. Keep. |
| `GradTensor<T>.AsTensor()` | `GradTensor.cs:124` | Extensions/AutoDiff — leave as-is per user decision |
| `TensorConversions.ToBatchTensors/FromBatchTensors` | `TensorConversions.cs` | Extensions/MLNet — leave as-is per user decision |
| `NivaraFrame.ToTensors<T>()` | `NivaraFrame.cs:226` | Internal (used by `MatrixMultiply` which is also being deprecated). Keep for now. |
| `NivaraSeries<T>.ToTensor()` | `NivaraSeries.cs:372` | Keep — delegates to column. Not deprecated. |

---

## Summary

| Disposition | Count |
|-------------|-------|
| KEEP | 14 |
| REDESIGN (Phase 2) | 11 (8 existing + 3 new) |
| DEPRECATE (Phase 1) | 11 methods + 1 unused |
| NO CHANGE | 7 |
