# Plan: Issue #1167 — DiagnosticsTracker global/unbounded/locked sink

## Problem

`DiagnosticsTracker` is a process-global static: one `bool` and one unbounded `List<OperationDiagnostics>`,
guarded by a single global lock. It cannot be scoped to a query, grows without limit while enabled,
and every parallel kernel write serialises on the same lock. `ExecutionDiagnostics` already exists
with the right per-execution scope, so the global sink is a duplicate with worse properties.

## Proposed changes

1. Introduce `OperationDiagnosticsCollector` as the real bounded storage (capacity default 10,000,
   overflow counter, per-instance lock).
2. `IOperationDiagnosticsSink` implemented by both `OperationDiagnosticsCollector` and
   `ExecutionDiagnostics`, so a query's `ExecutionDiagnostics` can be the recording target directly.
3. `DiagnosticsTracker` becomes a router: bounded global collector + `volatile bool globalEnabled` +
   `AsyncLocal<IOperationDiagnosticsSink?>` current sink; `BeginScope(...)`; lock-free fast path when
   disabled (`Volatile.Read`).
4. `OperationMeasurement` captures the target sink instead of a bool.
5. `ExecutionDiagnostics` implements the sink interface (per-instance lock).
6. `ExecutionEngine.Execute` / `ExecuteAsync` begin a diagnostics scope around strategy execution so
   kernel ops land in that query's diagnostics (isolated from concurrent queries).
7. Keep existing public API; add aliases for capacity/overflow names.

## Verification steps

- Build `Nivara.slnx`.
- Targeted tests: Diagnostics, Execution, Expressions, Metrics namespaces.
- Docs snippet gate if docs touched.
- Concurrency smoke test (two queries, disjoint KernelOperations).

## Planned commits

1. docs: plan #1167 in TODO.md
2. feat(diagnostics): add OperationDiagnosticsCollector + sink interface
3. feat(diagnostics): route DiagnosticsTracker through scoped sinks, bound global collector
4. feat(execution): scope diagnostics per query in ExecutionEngine
5. feat(diagnostics): ExecutionDiagnostics implements sink
6. test: capacity + concurrency coverage (if needed)
7. docs: remove TODO.md — plan executed

## Blast radius

- `src/Nivara/Diagnostics/OperationDiagnostics.cs` (DiagnosticsTracker, OperationMeasurement)
- `src/Nivara/Diagnostics/ExecutionDiagnostics.cs`
- `src/Nivara/Execution/ExecutionEngine.cs`
- Call sites unchanged (AutoDiff, Metrics, FusedExpressionEvaluator, NivaraColumn, Onnx) route via
  DiagnosticsTracker as before.
- Tests rely on global tracker; router preserves behaviour.

## GitHub issues log

- (none yet)
