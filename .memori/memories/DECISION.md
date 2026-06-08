# Memories

- AutoDiff execution plan decisions:
- Task 2 (Shape): Add Shape/Rank directly to GradTensor<T>
- Task 3 (Optimizer): Return-new-tensor approach (immutable-style)
- Sequencing: Phase A (Foundation) → B (Architecture) → C (Surface)
- Task 7 priority: Fix null mask handling in MatMul/Transpose first (urgent bug) <!-- id=a7bc90f4c2614b74a7672f3a1db64b98 entity=default type=decision ts=2026-06-07T09:54:09.1624496+00:00 v=1 tags=autodiff,planning -->
- User decisions for tensor back-off plan: T3 - mark [Obsolete] in-place, don't move namespaces. Tests for deprecated APIs - delete. T2 scope - full redesign (column span + frame row-span). T1 - public + internal audit. Merge T2+T4+T5. MLNet batch tensor methods - leave in Extensions. AutoDiff - leave as-is. <!-- id=cab2c3797b174eaa871170ff483692f4 entity=default type=decision ts=2026-06-08T06:00:41.0002528+00:00 v=1 tags=nivara,tensor,decisions,plan -->
