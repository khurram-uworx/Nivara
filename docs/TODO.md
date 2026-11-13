# Plan: #1170 — explicit query-source disposal stops swallowing failures

**Branch:** `khurram/1170` (off `main` at `f650c94e`)

## Problem

`QuerySourceHandle.Release`/`ReleaseAsync` and the abandoned-resource cleanup path
wrap `source.Dispose()` / `DisposeAsync()` in a bare `catch { }`. A failed file
flush, network close error, or broken async dispose is reported as success. The
XML docs state this explicitly ("disposal errors are swallowed"), conflating
**abandoned** cleanup (where swallowing is defensible) with **explicit
caller-initiated** dispose (where it is not). A second bare `catch { }` sits on
the streaming producer path (`StreamingExecutionStrategy`).

## Superseded premise (recorded, not resurrected)

#1168 (`68d9c9ec`) deleted `NivaraResourceManager` and the whole abandoned-resource
tracking subsystem (timer, dictionary, `Enable`/`Disable`/`ForceCleanup`/...). So:
- the "abandoned cleanup stays best-effort but counts the failure" AC is obsolete —
  there is no abandoned path left. Marked done, no code change.
- `NivaraResourceManager.CleanupAbandonedResources` and `NivaraResourceManager.cs` are
  gone; the issue's "Affected area" entry for it is stale. Comment on #1170.
- `src/Nivara/Linq/NivaraQuery.cs` needs no change (it only delegates to the
  frame).

A separate, related swallow lives one layer below the handle:
`ParquetLazySource.Dispose()` swallows `ParquetReader.DisposeAsync()` failures and
then skips `gate.Dispose()` because the throw escapes. Fixed in commit 2.

## Blast radius

- `src/Nivara/Query/QuerySourceHandle.cs` — `Release()`, `ReleaseAsync()` (internal).
- `src/Nivara/Query/QueryFrame.cs` — `Dispose()`, `DisposeAsync()` (public).
- `src/Nivara/Linq/NivaraQuery.cs`, `NivaraGroupedQuery` — delegate; no edit, doc
  only.
- `src/Nivara.Extensions/IO/ParquetDataSource.cs` — `ParquetLazySource.Dispose()`.
- `src/Nivara/Execution/StreamingExecutionStrategy.cs` — teardown catch block.
- `src/Nivara/Execution/ExecutionStrategyBase.cs` — exception-wrap path (no edit
  needed; the strategy throws `QueryExecutionException` itself).
- Tests: `QuerySourceHandleTests`, `AsyncStreamingTests`, `StreamingExecutionStrategyTests`.
- Docs: `docs/LINQ.md`, `docs/STREAMING.md`, `CHANGELOG.md` (prose only).

## Decisions (resolved with the human before implementation)

1. **Dispose failure shape:** raw original exception, no wrap — preserves the
   exact type (`IOException` etc.) and stack, matching `FileStream.Dispose`.
2. **Idempotency on failure:** `released` is set *before* `source.Dispose()`, so
   a failed release still counts as released; a retry is not attempted. Second
   dispose is a silent no-op.
3. **Parquet swallow:** fixed in this branch (separate commit).
4. **Streaming secondary failure:** throw `QueryExecutionException` whose inner is
   an `AggregateException(primary, producer)`, except expected cancellation and the
   "same faulted task twice" case.
5. **AC 3 (abandoned cleanup counts failures):** obsolete; recorded on #1170.
6. **No `TryDispose(out Exception)` overload** — nothing in-tree needs a
   non-throwing explicit path.

## Red flag / grounding note (Resistance G1)

Microsoft's dispose-pattern guidance says "AVOID throwing an exception from within
`Dispose(bool)`". This issue explicitly overrides that for *explicit*
caller-initiated dispose of an I/O-backed source, because a lost flush/close
error means lost data and there is no other signal. The human chose "raw original
exception" knowing this tension. The rationale is recorded in the PR body. No
finalizer/SafeHandle path is involved (the sources are not finalizable), so the
guidance's primary concern (finalizers) does not apply.

## Planned commits

1. `QuerySourceHandle`: explicit release propagates
2. `ParquetLazySource`: reader-close failure propagates; semaphore released in finally
3. `StreamingExecutionStrategy`: producer teardown no longer swallows unqualified failure
4. Docs + changelog

## Verification steps

- `dotnet build Nivara.slnx`
- Targeted tests (Release):
  `QuerySourceHandleTests`, `AsyncStreamingTests`, `StreamingExecutionStrategyTests`,
  `StreamingProducerCleanupTests`, `LazySourceDisposalTests`,
  `ResourceManagementPropertyTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs gate if docs text changes: `--filter "FullyQualifiedName~Nivara.Tests.Docs"`
- Full suite only after asking.

## GitHub issues log

- [x] #1170 — explicit query-source disposal swallows every exception (this branch).
- [ ] #1189 — lazy query sources leak handle until GC (related, out of scope for this branch; no new issue).
- [ ] (to create while executing) if any deferred work surfaces — record number here.
