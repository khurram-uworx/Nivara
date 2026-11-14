# Plan: Issue #1172 — async query API sync-over-async and cancellation

## Problem
`Collect()` is sync-over-async (`CollectAsync(default).GetAwaiter().GetResult()`), and `CollectAsync` awaits
without `ConfigureAwait(false)`, so a UI host with a `SynchronizationContext` can deadlock. Operation
`ExecuteAsync` defaults ignore the token; `IQuerySource.ExecuteAsync` runs synchronously on the caller
thread. Several awaits in `QueryFrame`, `ExecutionEngine`, `DiagnosticHelper`, and `ParallelExecutionStrategy`
also lack `ConfigureAwait(false)`.

## Proposed changes
1. Add `ConfigureAwait(false)` to missing awaits (`DiagnosticHelper`, `ExecutionEngine`, `QueryFrame`
   `CollectAsync`/`AsStream`, `ParallelExecutionStrategy` `Parallel.ForEachAsync`).
2. Make `Collect()` deadlock-safe via `Task.Run(() => CollectAsync(default))`.
3. Add cancellable `Execute(input, ct)` overloads to `FilterOperation`, `JoinOperation`,
   `SortOperation`, `GroupByOperation`; check `ct` in their main loops; update `ExecuteAsync`.
4. Update catch filters to exclude `OperationCanceledException`.
5. Make `IQueryOperation.ExecuteAsync` default observe the token.
6. Document `IQuerySource.ExecuteAsync` synchronous fallback.
7. Add tests (deadlock, cancellation, source default, ConfigureAwait gate).
8. Update `docs/LINQ.md`.

## Verification steps
- `dotnet build Nivara.slnx`
- targeted tests for async/cancellation/docs
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan issue #1172 in TODO.md
2. fix async: ConfigureAwait(false) gaps
3. fix async: deadlock-safe Collect()
4. feat cancellation: Filter/Join/Sort/GroupBy
5. fix interfaces: token observation + docs
6. test: deadlock/cancellation/source default/ConfigureAwait gate
7. docs: LINQ sync/async split
8. docs: remove TODO.md

## Blast radius
- `src/Nivara/Query/QueryFrame.cs`, `src/Nivara/Linq/NivaraQuery.cs`
- `src/Nivara/Execution/ExecutionEngine.cs`, `ParallelExecutionStrategy.cs`
- `src/Nivara/Diagnostics/DiagnosticHelper.cs`
- `src/Nivara/Operations/{Filter,Join,Sort,GroupBy}Operation.cs`
- `src/Nivara/Query/IQueryInterfaces.cs`
- tests under `tests/Nivara.Tests`

## GitHub issues log
- (none yet)
