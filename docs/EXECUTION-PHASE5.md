# Phase 5: Clean Up Dead Code & Naming

## Purpose

Remove unused API surface, fix naming inconsistencies, replace magic strings with constants, and fix a latent bug where `"Concatenation"` string checks never match the actual `ConcatenationOperation.OperationType` value.

This phase targets Issues #5, #6, #10, #11, #12, #13 from the execution plan.

## Suggested Execution Order

1. **Task 1**: Verify dead API surface is already removed (check Phase 1 completeness)
2. **Task 2**: Extract operation type constants — eliminates magic strings across 10+ files
3. **Task 3**: Fix `"Concatenation"` bug — actual `OperationType` is `$"Concatenate{direction}"`, never `"Concatenation"`
4. **Task 4**: Normalize private method naming across strategies to PascalCase
5. **Task 5**: Fix `QueryExecutor.EstimateExecutionCost` return type (`int` → `long`) and remove redundant schema validation
6. **Task 6**: Make `ExecutionEngine.SetOptimizer` public or remove it; improve `QueryPlan` serialization

## Coordination Notes

- **Task 1 is a verification gate** — no code changes, just confirm Phase 1 actually removed the declared API surface.
- **Tasks 2 & 3 are tightly coupled** — extracting operation type constants is a prerequisite for fixing the `"Concatenation"` bug cleanly. Do them together.
- **Tasks 4, 5, 6 are independent** — can run in parallel after Task 1.
- **Task 2 affects the most files** and will create merge conflicts with any branch touching strategy cost estimation or `ColumnEliminationRule.AnalyzeColumnUsage`. Coordinate with other phases.

## Task 1: Verify Dead API Surface Removal

### Priority

High (gate)

### Goal

Confirm that `IQueryOperation<T>.Transform`, `IQueryOperation<T>.ExecuteAsync`, and `IQueryOperation.Strategy` (declared for removal in Phase 1) are already absent.

### Why this exists

Phase 1 was supposed to remove these members. Phase 5 must not double-work or work on already-removed surface. Verify before proceeding.

### Scope

- Check current `IQueryOperation<T>` signature in `src/Nivara/Query/IQueryInterfaces.cs`
- Check current `IQueryOperation` signature in `src/Nivara/Query/IQueryInterfaces.cs`
- Check no callers or overrides of `Transform`, `ExecuteAsync`, or `Strategy` remain anywhere in `src/Nivara/`

### Acceptance criteria

- Confirmed that `IQueryOperation<T>` only has `QueryPlan Plan { get; }`
- Confirmed that `IQueryOperation` only has `OperationType`, `TransformSchema`, `Execute`
- No dead member references found in source or tests

### Files likely involved

- `src/Nivara/Query/IQueryInterfaces.cs`

---

## Task 2: Extract Operation Type Constants

### Priority

High

### Goal

Replace all magic string operation type literals with a `static class OperationType` constants class, eliminating duplication across 10+ files.

### Why this exists

Issue #10: Magic strings `"Filter"`, `"Select"`, `"Sort"`, `"GroupBy"`, `"Join"`, `"Projection"`, `"Slice"`, `"Concatenation"`/`"Concatenate"` are scattered across strategy files, cost estimation, visitor dispatch, and optimization rules. No single source of truth, prone to typos and drift.

### Scope

- Create `src/Nivara/Query/OperationType.cs` with `public static class OperationType` containing `const string` members:
  - `Filter`, `Select`, `Sort`, `GroupBy`, `Join`, `Projection`, `Slice`, `ConcatenationPrefix` (= `"Concatenate"`)
- Update every operation class to return the constant:
  - `SortOperation.OperationType` → `OperationType.Sort`
  - `GroupByOperation.OperationType` → `OperationType.GroupBy`
  - `JoinOperation.OperationType` → `OperationType.Join`
  - `ProjectionOperation.OperationType` → `OperationType.Projection`
  - `SelectOperation.OperationType` → `OperationType.Select`
  - `SliceOperation.OperationType` → `OperationType.Slice`
  - `ConcatenationOperation.OperationType` → `$"{OperationType.ConcatenationPrefix}{direction}"`
- Update all switch/pattern matches in:
  - `ParallelExecutionStrategy.isParallelizable` and `EstimateExecutionCost`
  - `StreamingExecutionStrategy` (non-streamable set, cost estimation)
  - `EagerExecutionStrategy` (cost estimation)
  - `LazyExecutionStrategy` (cost estimation)
  - `QueryExecutor.EstimateExecutionCost`
  - `QueryOptimizer.AnalyzeOptimizationOpportunities`
  - `ColumnEliminationRule.AnalyzeColumnUsage`
  - `QueryPlanVisitorBase.Visit` and `QueryPlanTransformerBase<T>.Visit`
  - `ExecutionDiagnostics.GenerateReport` (line 285-286 string checks)
- Update all test files that reference these strings as operation types

### Constraints

- `ConcatenationOperation.OperationType` is computed (`$"Concatenate{direction}"`); maintain this pattern but use constant prefix
- Do NOT introduce a runtime type switch — keep `OperationType` as a string-based approach for loose coupling
- Use `StartsWith(OperationType.ConcatenationPrefix, StringComparison.Ordinal)` where prefix matching is needed (currently in `ParallelExecutionStrategy`)

### Acceptance criteria

- Zero magic string operation type literals remain in `src/Nivara/` source code
- All strategy cost-estimation switch statements use `OperationType.X` constants
- All visitor dispatch in `QueryPlanVisitorBase` and `QueryPlanTransformerBase<T>` uses constants
- Build passes with no warnings
- Existing tests still pass without modification (they use strings; test helpers return strings — moved to constants too)

### Files likely involved

- `src/Nivara/Query/IQueryInterfaces.cs`
- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Execution/LazyExecutionStrategy.cs`
- `src/Nivara/Query/QueryExecutor.cs`
- `src/Nivara/Query/QueryOptimizer.cs`
- `src/Nivara/Query/QueryPlanVisitor.cs`
- `src/Nivara/Optimization/ColumnEliminationRule.cs`
- `src/Nivara/Operations/SortOperation.cs`
- `src/Nivara/Operations/GroupByOperation.cs`
- `src/Nivara/Operations/JoinOperation.cs`
- `src/Nivara/Operations/ProjectionOperation.cs`
- `src/Nivara/Operations/SelectOperation.cs`
- `src/Nivara/Operations/SliceOperation.cs`
- `src/Nivara/Operations/ConcatenationOperation.cs`
- `tests/Nivara.Tests/Execution/ParallelExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/EagerExecutionStrategyTests.cs`
- `tests/Nivara.Tests/Execution/ExecutionTestHelpers.cs`

---

## Task 3: Fix `"Concatenation"` — Dead Code Bug

### Priority

High (bug fix)

### Goal

Fix the latent bug where all 8 checks for `"Concatenation"` in source code never match because the actual `ConcatenationOperation.OperationType` returns `$"Concatenate{direction}"` (e.g. `"ConcatenateVertical"`, `"ConcatenateHorizontal"`).

### Why this exists

Issue #11: `ConcatenationOperation.OperationType` was defined as `$"Concatenate{direction}"`, but all callers check for the string `"Concatenation"` — a value that is never produced. This means:
- Streaming strategy's `StreamableOperationTypes` set never includes concatenation operations
- Cost estimation switches miss the concatenation case (falls to `_` default branch, silently wrong)
- `ColumnEliminationRule.AnalyzeColumnUsage` never matches concatenation
- `QueryPlanVisitor` visitor dispatch never matches concatenation operations

### Scope

- Replace all `"Concatenation"` string literals with the correct check after Task 2's constants are in place:
  - Use `OperationType.ConcatenationPrefix` for `StartsWith` checks
  - Use `OperationType.ConcatenationPrefix` in switch arms / case labels (requires switch with `when` clause or pattern match)
  - For sets like `StreamableOperationTypes`, store the prefix and check with `StartsWith` instead of `Contains`
- Add a test that creates a `ConcatenationOperation` and verifies its `OperationType` produces a string that the fixed checks can match

### Constraints

- Must be done after or concurrent with Task 2 (constants extraction) to avoid rework
- The fix should be in the caller checks, not by changing `ConcatenationOperation.OperationType` (which correctly reflects the direction)

### Acceptance criteria

- No `"Concatenation"` string literal remains for operation type matching
- `StreamingExecutionStrategy` correctly identifies `ConcatenateVertical`/`ConcatenateHorizontal` as streamable
- Cost estimation in all strategies correctly matches concatenation operations
- `QueryPlanVisitorBase.Visit` and `QueryPlanTransformerBase<T>.Visit` dispatch to `VisitConcatenation` for actual concatenation operations
- `ColumnEliminationRule.AnalyzeColumnUsage` matches concatenation operations
- New test verifies end-to-end matching

### Files likely involved

- `src/Nivara/Execution/StreamingExecutionStrategy.cs`
- `src/Nivara/Execution/LazyExecutionStrategy.cs`
- `src/Nivara/Execution/EagerExecutionStrategy.cs`
- `src/Nivara/Optimization/ColumnEliminationRule.cs`
- `src/Nivara/Query/QueryPlanVisitor.cs`
- `tests/Nivara.Tests/Execution/StreamingExecutionStrategyTests.cs`

---

## Task 4: Normalize Private Method Naming

### Priority

Medium

### Goal

Rename all private methods in execution strategy classes to PascalCase to match C# conventions.

### Why this exists

Issue #5: `ParallelExecutionStrategy` and `StreamingExecutionStrategy` use `camelCase` for private methods (e.g., `isParallelizable`, `shouldUseParallelism`, `validateParallelismConfiguration`, `createSliceExecuteKernel`, `executeOperationParallelSync`, `isSuitableForStreaming`, `calculateChunkSize`, `executeOperationsOnData`, `readSourceAsync`, `executeCoreInternalAsync`). C# convention is PascalCase for all method names.

### Scope

- Rename all `camelCase` private methods in `ParallelExecutionStrategy.cs` to PascalCase:
  - `isParallelizable` → `IsParallelizable`
  - `shouldUseParallelism` → `ShouldUseParallelism`
  - `validateParallelismConfiguration` → `ValidateParallelismConfiguration`
  - `createSliceExecuteKernel` → `CreateSliceExecuteKernel`
  - `executeOperationParallelSync` → `ExecuteOperationParallelSync`
  - `executeSortParallelSync` → `ExecuteSortParallelSync`
  - `executeFilterParallelSync` → `ExecuteFilterParallelSync`
  - `executeGroupByParallelSync` → `ExecuteGroupByParallelSync`
  - `executeJoinParallelSync` → `ExecuteJoinParallelSync`
  - `executeConcatenationParallelSync` → `ExecuteConcatenationParallelSync`
  - `GetReferenceColumn` (already PascalCase, keep)
  - `GetColumnName` (already PascalCase, keep)
  - `executeOperationParallelAsync` → `ExecuteOperationParallelAsync`
  - `executeFilterParallelAsync` → `ExecuteFilterParallelAsync`
  - `executeGroupByParallelAsync` → `ExecuteGroupByParallelAsync`
  - `executeJoinParallelAsync` → `ExecuteJoinParallelAsync`
  - `executeConcatenationParallelAsync` → `ExecuteConcatenationParallelAsync`
  - `readSourceAsync` → `ReadSourceAsync`
  - `executeCoreInternalAsync` → `ExecuteCoreInternalAsync`
- Rename all `camelCase` private methods in `StreamingExecutionStrategy.cs` to PascalCase:
  - `isSuitableForStreaming` → `IsSuitableForStreaming`
  - `isOperationStreamable` → `IsOperationStreamable`
  - `calculateChunkSize` → `CalculateChunkSize`
  - `executeOperationsOnData` → `ExecuteOperationsOnData`
  - `executeCoreInternalAsync` → `ExecuteCoreInternalAsync`
- No changes needed for `EagerExecutionStrategy` or `LazyExecutionStrategy` (no private methods)

### Acceptance criteria

- All private methods in strategy classes use PascalCase
- Build passes
- All tests pass

### Files likely involved

- `src/Nivara/Execution/ParallelExecutionStrategy.cs`
- `src/Nivara/Execution/StreamingExecutionStrategy.cs`

---

## Task 5: Fix `QueryExecutor.EstimateExecutionCost` Return Type & Remove Redundant Validation

### Priority

Medium

### Goal

Fix the type mismatch between `QueryExecutor.EstimateExecutionCost` (returns `int`) and `IExecutionStrategy.EstimateExecutionCost` (returns `long`). Remove redundant per-operation schema validation in `QueryExecutor.Execute()`.

### Why this exists

Issue #9: `QueryExecutor.EstimateExecutionCost` returns `int.MaxValue` and `int` cost values, but the interface requires `long`. Issue #8: `QueryExecutor.Execute()` calls `ValidatePlan()` then re-validates each operation's schema after execution (lines 67-86), which is redundant because `Execute()` contracts guarantee correct output shape.

### Scope

- Change `QueryExecutor.EstimateExecutionCost` return type from `int` to `long`
- Update `int.MaxValue` → `long.MaxValue`, all `int` cost variables → `long`
- Review and remove the redundant schema re-validation block in `QueryExecutor.Execute()` (lines 67-86: the `ValidatePlan` was already called, then per-operation `TransformSchema` + compatibility check)
  - Move only the null-check logic (`currentColumns == null`) outside the schema block
  - Remove `createSchemaFromColumns` and `transformedSchema` / `actualSchema` compatibility check
- Ensure no downstream code depends on the re-validation

### Constraints

- Keep the per-operation null-return check (lines 73-77) — that is not redundant
- Remove only the schema compatibility check (lines 67-69, 79-86)

### Acceptance criteria

- `QueryExecutor.EstimateExecutionCost` returns `long`
- No schema re-validation occurs inside the operation loop in `Execute()`
- All tests pass
- No behavior change for valid query plans

### Files likely involved

- `src/Nivara/Query/QueryExecutor.cs`

---

## Task 6: Clean Up `SetOptimizer` & Add Structured `QueryPlan` Serialization

### Priority

Low

### Goal

Either make `ExecutionEngine.SetOptimizer` public (with proper constructor injection) or remove it. Add JSON/structured serialization to `QueryPlan`.

### Why this exists

Issue #6: `ExecutionEngine.SetOptimizer()` is `internal` and only called from tests. Issue #12: `QueryPlan` has no serialization beyond `ToString()`, making debugging and diagnostics harder.

### Scope

- **Option A (recommended)**: Add an `ExecutionEngine(QueryOptimizer?)` constructor parameter and mark the parameterless constructor as obsolete or keep both. Remove `SetOptimizer()` entirely. Update test callers.
- **Option B**: Keep `SetOptimizer()` but make it public and add XML doc explaining usage.
- Add `QueryPlan.Serialize()` returning a structured format (JSON via `System.Text.Json`) — public for diagnostics use.
- Add `QueryPlan.ToDebugString()` that includes source schema, operation types, and result schema for quick debugging.

### Acceptance criteria

- `SetOptimizer` is either removed with test callers migrated to constructor injection, or made public with proper documentation
- `QueryPlan` has a `Serialize()` method producing JSON
- `QueryPlan` has a `ToDebugString()` for diagnostics
- All tests pass

### Files likely involved

- `src/Nivara/Execution/ExecutionEngine.cs`
- `src/Nivara/Query/QueryPlan.cs`
- `tests/Nivara.Tests/Execution/ExecutionEngineTests.cs`

---

## Task 7: Fix `ExecutionDiagnostics.MemoryAllocated` Negative Values

### Priority

Low

### Goal

Prevent `ExecutionDiagnostics.MemoryAllocated` from returning negative values when GC collects between initial and peak measurements.

### Why this exists

Issue #13: `MemoryAllocated` computes as `PeakMemoryUsage - initialMemory`, where `initialMemory` is captured in the constructor. If GC collects memory between construction and measurement, `PeakMemoryUsage` may be lower than `initialMemory`, producing a negative value.

### Scope

- Change `MemoryAllocated` property to `Math.Max(0, PeakMemoryUsage - initialMemory)`
- Add a test that verifies `MemoryAllocated` is never negative

### Acceptance criteria

- `MemoryAllocated` never returns a negative value
- Existing behavior preserved for non-negative cases
- Test validates the fix

### Files likely involved

- `src/Nivara/Diagnostics/ExecutionDiagnostics.cs`
- `tests/Nivara.Tests/Diagnostics/ExecutionDiagnosticsTests.cs`

---

## Suggested Agent Handout Batches

### Batch A: verification + infrastructure

- Task 1 (verification, 10 min)
- Task 2 (constants extraction, large blast radius — 2+ hrs)

### Batch B: bug fix + naming

- Task 3 (depends on Task 2 — do after or with)
- Task 4 (pure rename, low risk)

### Batch C: small fixes

- Task 5 (type fix + schema removal, 30 min)
- Task 6 (optional cleanup, 30 min)
- Task 7 (safety fix, 15 min)

---

## Final Checklist

- [ ] Task 1: confirms dead API is already gone
- [ ] Task 2: zero magic operation type strings remain in source
- [ ] Task 3: `"Concatenation"` dead code bug fixed; checks match actual `Concatenate{direction}` values
- [ ] Task 4: all private methods in strategies use PascalCase
- [ ] Task 5: `EstimateExecutionCost` returns `long`; redundant schema validation removed
- [ ] Task 6: `SetOptimizer` cleaned up; `QueryPlan` has JSON serialization
- [ ] Task 7: `MemoryAllocated` never negative
- [ ] All existing tests pass after each batch
- [ ] Build succeeds with no warnings
