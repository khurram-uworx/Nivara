# Memories

- Phase A completed (June 2026):
- A1: MatMul/Transpose null mask handling fixed. MatMul result null if any contributing element is null. Transpose preserves null positions. Tests added.
- A2: TypeValidator now throws TypeValidationException (with ExpectedType/ActualType). Tests added for long/decimal rejection.
- 156 AutoDiff tests passing. <!-- id=a595994a9677428c8abb03616ffa07a8 entity=default type=milestone ts=2026-06-07T10:00:55.3376436+00:00 v=1 tags=autodiff,phase-a,completed -->
- Completed all 7 tasks from TENSORS-TASKS.md (T1-T7) across 4 phases:
- Phase 0 (T1): Audited 32 interop methods, produced TENSORS-AUDIT.md
- Phase 1 (T3): Added [Obsolete] to 4 NivaraFrame methods (Dot, CosineSimilarity, ColumnNorms, RowNorms) and 7 TensorExtensions methods (AddTensor, MultiplyTensor, SumTensor, DotProduct, Norm, TransformTensor, MatrixMultiply); deleted deprecated tests from 3 test files
- Phase 2 (T2+T4+T5): Added TryGetSpan, CopyTo, GetNullMask on NivaraColumn; TryGetRowMajorSpan, CopyToRowMajor on NivaraFrame; TryGetSpan on IColumnStorage/TensorStorage/MemoryStorage; AsSpanOrDefault extension in TensorInteropExtensions
- Phase 3 (T6): Marked closed items in TENSORS-GAPS.md (7 items closed: E1-E4, E6, E10, P1)
- Phase 4 (T7): Added XML doc <see cref="..."/> redirects to all 11 deprecated methods
- Fixed RowNorms missing [Obsolete] attribute <!-- id=6afa7c7413f44b938917241083297dfb entity=default type=milestone ts=2026-06-08T06:15:42.2662000+00:00 v=1 tags=tensors,,plan-complete,,phase-summary -->
