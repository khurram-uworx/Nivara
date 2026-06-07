# Memories

- Phase A completed (June 2026):
- A1: MatMul/Transpose null mask handling fixed. MatMul result null if any contributing element is null. Transpose preserves null positions. Tests added.
- A2: TypeValidator now throws TypeValidationException (with ExpectedType/ActualType). Tests added for long/decimal rejection.
- 156 AutoDiff tests passing. <!-- id=a595994a9677428c8abb03616ffa07a8 entity=default type=milestone ts=2026-06-07T10:00:55.3376436+00:00 v=1 tags=autodiff,phase-a,completed -->
