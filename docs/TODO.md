# TODO — Close issue #155: span-native compiled fused target

## Problem
Phase 2 of `docs/plan/POLARS-ROADMAP.md` is delivered except for the residual #155:
the compiled (`System.Linq.Expressions`) fused path is not span-native. It has two
avoidable inefficiencies:

1. A per-call `object[]` allocation in the dispatch wrapper
   (`CompiledFusedInvoke` over `object[] leafArrays`).
2. A **whole-slice copy** of sliced leaf columns in `SnapshotLeaf`
   (`FusedExpressionEvaluator.cs:669-677`) when `segment.Offset != 0` — a full-column
   copy per evaluation for any sliced frame.

The IR/span path (`FusedKernel` + `TensorPrimitivesKernel`) is already zero-copy and
chunked. The compiled path should match it in memory behavior.

## Hard constraint
`Expression.Lambda(...).Compile()` cannot bind a `Span<T>` parameter — `Span<T>` is a
ref struct and expression trees reject ref-struct parameters. So a *literally* `Span<T>`
compiled delegate is impossible through `System.Linq.Expressions`. The achievable,
equivalent outcome is span-equivalence: zero-copy contiguous-array reads + per-leaf
base-offset slicing, plus no per-call dispatch allocation. We do NOT rewrite
`BuildCompiledDelegate` into raw `Reflection.Emit` IL.

## Proposed changes (all in `src/Nivara/Expressions/FusedExpressionEvaluator.cs`)

### 1. Per-leaf base offset → zero-copy reads for sliced columns
- Add `int[] baseOffsets` parameter to `CompiledFusedInvoke` and thread it through
  `BuildCompiledDelegate` (the outer lambda + the `object[]` wrapper at lines 801-819).
- In `BuildCompiledNode`, for `KernelOp.Column` (line 834-835), change the index from
  `Expression.Add(indexVar, startParam)` to
  `Expression.Add(Expression.Add(indexVar, startParam), Expression.ArrayAccess(baseOffsetParam, Expression.Constant(node.Left)))`.
  For non-sliced leaves the offset is 0 → behavior unchanged.
- Change `SnapshotLeaf`/`SnapshotLeaves` (lines 649-677) to return the *underlying*
  array (`segment.Array`, zero-copy whether or not sliced) plus its `segment.Offset` as
  the base offset. Build a cached `int[] baseOffsets` per plan (`plan.Columns.Count`
  entries, constant across evaluations — cache on the plan or per signature).
- `EvaluateCompiled` passes the underlying arrays + `baseOffsets` instead of copied
  slices. Result array allocated once at offset 0; `destStart = start` unchanged.

### 2. Eliminate the per-call `object[]` allocation
- Rent the leaf-reference array from `ArrayPool<object>.Shared` around the
  `invoke(...)` call in `EvaluateCompiled` and return it after the call. The strongly
  typed captured `T[]` params inside the compiled delegate are already cached.

### 3. Update the class-header note (lines 19-26)
- Replace "the delegate consumes `T[]` arrays rather than spans" with the precise
  statement: the compiled target is span-equivalent via zero-copy contiguous-array +
  per-leaf base-offset slicing; a literal `Span<T>` compiled delegate is impossible
  under `System.Linq.Expressions` (ref-struct ban) and not worth a `Reflection.Emit`
  rewrite.

## Blast radius
- **Files changed:** `src/Nivara/Expressions/FusedExpressionEvaluator.cs` only (plus
  doc/test files).
- **Symbols touched:** `CompiledFusedInvoke` delegate signature, `compiledKernelCache`
  build, `BuildCompiledDelegate`, `BuildCompiledNode` (Column case), `SnapshotLeaf`,
  `SnapshotLeaves`, `EvaluateCompiled`. The public API surface is unchanged (`Evaluate`,
  `EvaluateChunked`, `EvaluateBoolean` signatures stay the same).
- **Downstream callers:** `EvaluateCore` (routing), `EvaluateChunked`, all query
  operators that fuse expressions. No caller changes — only internal dispatch.
- **Tests covering this code:** `tests/Nivara.Tests/Query/FusedExpressionEvaluatorTests.cs`
  (guardrail `CompiledPathEvaluationCount` assertions at lines 85, 502, 917, etc.),
  `QueryOptimizationPropertyTests`, `QueryExecutionPropertyTests`, streaming/property
  suites, and `tests/Nivara.PerformanceTests/Program.cs` (`EvaluateChunked` benchmark).
- **Risk:** low. The base-offset change is a no-op for non-sliced leaves (offset 0).
  The `object[]` rental is bounded (returned synchronously after invoke). Cache key
  stays `plan.Signature`, which already encodes concrete element types
  (`ExpressionTypeInferer.cs:186`) so it remains type-safe.

## Verification
1. `dotnet build Nivara.slnx`
2. `dotnet test` — focus `FusedExpressionEvaluatorTests`, `QueryOptimizationPropertyTests`,
   `QueryExecutionPropertyTests`, streaming/property suites.
3. Optional perf check in `Nivara.PerformanceTests/Program.cs` (`EvaluateChunked`
   benchmark) on a sliced-frame workload to confirm no regression.

## Planned commit list
1. `docs: plan #155 span-native compiled target in TODO.md`
2. `refactor: zero-copy leaf reads in compiled fused path via per-leaf base offset`
3. `perf: rent dispatch wrapper array from ArrayPool in compiled fused path`
4. `docs: clarify span-equivalence + ref-struct constraint in FusedExpressionEvaluator`
5. `test: add sliced-leaf zero-copy guardrails for compiled fused path`
6. `docs: mark #155 resolved in POLARS-ROADMAP.md`
7. `docs: remove TODO.md — plan executed` (only after branch review)

## GitHub issues log

- [ ] #NNN — (created during execution if any deferred work/concern is found)
