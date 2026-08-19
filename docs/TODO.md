# Plan: Promote QueryPlan / Query / Execution types to public (Issue #275)

## Problem

`QueryFrame` is public but `ToQueryPlan()` returns the internal `QueryPlan` type, so external callers cannot inspect or custom-execute query plans. The diagnostic methods (`ExplainPlan`, `GetDiagnosticInfo`, `AnalyzeQueryPlan`, `AnalyzeOptimizations`) return strings — useful but limiting. Promoting the underlying types enables full plan inspection and custom execution.

## Scope

### Tier 1 — Core types (required)
1. `QueryPlan` (`src/Nivara/Query/QueryPlan.cs`) — `internal sealed class` → `public sealed class`
2. `QueryPlanAnalyzer` (`src/Nivara/Query/QueryPlan.cs`) — `internal static class` → `public static class`
3. `IQuerySource` (`src/Nivara/Query/IQueryInterfaces.cs`) — `internal interface` → `public interface`
4. `IQueryOperation` (`src/Nivara/Query/IQueryInterfaces.cs`) — `internal interface` → `public interface`
5. `IQueryOperation<T>` (`src/Nivara/Query/IQueryInterfaces.cs`) — `internal interface` → `public interface`
6. `OperationType` (`src/Nivara/Query/OperationType.cs`) — `internal static class` → `public static class`
7. `QueryFrame.ToQueryPlan()` (`src/Nivara/Query/QueryFrame.cs`) — `internal` → `public`

### Tier 2 — Execution types (required for custom execution)
8. `ExecutionEngine` (`src/Nivara/Execution/ExecutionEngine.cs`) — `internal sealed class` → `public sealed class`
9. `IExecutionStrategy` (`src/Nivara/Execution/ExecutionEngine.cs`) — `internal interface` → `public interface`
10. `NivaraExecutionContext` (`src/Nivara/Execution/NivaraExecutionContext.cs`) — `internal sealed class` → `public sealed class`
11. `ExecutionProgress` (`src/Nivara/Execution/NivaraExecutionContext.cs`) — `internal sealed class` → `public sealed class`

### Tier 3 — Diagnostics (nice-to-have)
12. `QueryDiagnostics` (`src/Nivara/Query/QueryDiagnosticMode.cs`) — `internal static class` → `public static class`

### Keep internal (not in scope)
- Concrete operations (FilterOperation, SelectOperation, etc.) — users inspect via `IQueryOperation.OperationType`
- QueryOptimizer, OptimizationEngine, OptimizationRule, OptimizationResult — optimization internals
- IPredicatePushdownSource — implementation detail
- QueryNode / visitor hierarchy — separate tree representation

## Blast radius

- All test files reference these types via `InternalsVisibleTo` — no test changes needed for the visibility change itself.
- No behavioral changes — only access modifiers change.
- ~15+ production files already use these types internally; they remain compatible.

## Verification
- Build: `dotnet build Nivara.slnx`
- Tests: ask human before `dotnet test`

## Commit plan
1. `docs: plan #275 in TODO.md`
2. `promote: make QueryPlan, IQuerySource, IQueryOperation, OperationType public`
3. `promote: make ExecutionEngine, IExecutionStrategy, NivaraExecutionContext, ExecutionProgress public`
4. `promote: make QueryPlanAnalyzer and QueryDiagnostics public`
5. `promote: make QueryFrame.ToQueryPlan() public`
6. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)
