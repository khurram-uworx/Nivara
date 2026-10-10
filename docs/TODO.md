# Plan: #508 — `QueryFrame.DisposeAsync` vs `Dispose` disposal-error contract

Branch: `khurram/508` (off `khurram/556`).

## Problem

`QueryFrame.Dispose()` swallows source-disposal errors while `QueryFrame.DisposeAsync()`
propagates them, so a caller using `await using` can get an exception from cleanup that the
synchronous `using` form silently swallows.

The asymmetry is not in `QueryFrame` itself — both methods forward to the shared
`QuerySourceHandle`:

- `QuerySourceHandle.Release()` (`src/Nivara/Query/QuerySourceHandle.cs`) — `Interlocked`
  idempotency guard, then `try { source.Dispose(); } catch { }` → **swallows**.
- `QuerySourceHandle.ReleaseAsync()` — same guard, then
  `if (source is IAsyncDisposable) await …; else source.Dispose();` → **no catch, propagates**.

Two corrections to the issue's framing:

1. **Only half-unobservable.** The `await IAsyncDisposable.DisposeAsync()` branch is dead
   (no `IQuerySource` in `src/` or `tests/` also implements `IAsyncDisposable`), but the
   `else source.Dispose()` branch is live: a synchronous source whose `Dispose()` throws
   makes `DisposeAsync()` throw today while `Dispose()` swallows.
2. **Partly documented already, just not where the issue wants.** `QuerySourceHandle.ReleaseAsync`
   says "disposal errors propagate" in XML; `QueryFrame.DisposeAsync` is bare `/// <inheritdoc />`.
   CHANGELOG #268 records the split as deliberate.

## Decision (confirmed by the human)

**Both paths swallow.** Matches the .NET dispose-pattern guidance the issue quotes ("DO NOT
throw from Dispose") and the `IAsyncDisposable.DisposeAsync` requirement that repeated calls
must not throw; matches the abandoned-resource cleanup path, which also swallows; smallest
change. Root-caused in the `QuerySourceHandle` layer, which is where the fork actually is.

## Blast radius

- `src/Nivara/Query/QuerySourceHandle.cs` — `ReleaseAsync` body + XML doc. Internal type,
  one caller (`QueryFrame.DisposeAsync`).
- `src/Nivara/Query/QueryFrame.cs` — XML docs on `Dispose`/`DisposeAsync` only; no body change.
- `src/Nivara/Linq/NivaraQuery.cs` — forwards to `QueryFrame`; no change (optionally re-check
  docs still accurate).
- `tests/Nivara.Tests/Query/QuerySourceHandleTests.cs` — new test doubles + tests. Introduces
  the repo's first `IAsyncDisposable` `IQuerySource`, making the previously-dead await branch live.
- `CHANGELOG.md` — one entry under `## [Unreleased]` → `### Fixed`.
- No public API change. No interface change (`IQuerySource` stays `IDisposable`).

## Grounding (G1)

- Microsoft Learn: `IAsyncDisposable.DisposeAsync` — "the object must not throw an exception if
  its DisposeAsync method is called multiple times"; FxDG disposal pattern — avoid throwing from
  `Dispose`. The async path is the deviant one.
- code-memory: `QuerySourceHandle` is `internal sealed`, `Release`/`ReleaseAsync` only; `QueryFrame`
  is the sole forwarder; `NivaraQuery`/`NivaraGroupedQuery` forward too. `IAsyncDisposable` appears
  only in `QueryFrame.cs` and `NivaraQuery.cs` outside this file.
- No red flags requiring a further decision: the human already chose "both swallow".

## Proposed changes

### 1. `QuerySourceHandle.ReleaseAsync` — swallow both branches

```csharp
internal async ValueTask ReleaseAsync()
{
    if (Interlocked.Exchange(ref released, 1) != 0) return;

    NivaraResourceManager.UntrackResource(this);

    try
    {
        if (source is IAsyncDisposable asyncDisposable)
            await asyncDisposable.DisposeAsync().ConfigureAwait(false);
        else
            source.Dispose();
    }
    catch
    {
        // Ignore disposal errors, mirroring Release() and the abandoned-resource cleanup path.
    }
}
```

Doc: change "Idempotent; disposal errors propagate." → "Idempotent; disposal errors are
swallowed, mirroring `Release` and the abandoned-resource cleanup path so neither disposal
path throws."

### 2. `QueryFrame.Dispose` / `DisposeAsync` — explicit XML docs

Replace `/// <inheritdoc />` on both with docs stating: one source one release shared across
the chain (like `FileStream`); idempotent — repeated calls are no-ops; disposal failures are
swallowed on both paths so `using` and `await using` behave identically; `DisposeAsync` awaits
`IAsyncDisposable` sources and falls back to `Dispose()` otherwise.

### 3. Tests — `QuerySourceHandleTests.cs`

Add two doubles and focused NUnit 4 tests (AAA, `Method_Scenario_ExpectedBehavior`):

- `ThrowingSource : IQuerySource` — `Dispose()` throws.
- `RecordingAsyncSource : IQuerySource, IAsyncDisposable` — records `DisposeAsync` calls; can throw.

Tests:

- `Release_SourceDisposeThrows_DoesNotPropagate`
- `ReleaseAsync_SourceDisposeThrows_DoesNotPropagate` (exercises the live `else` branch)
- `ReleaseAsync_AsyncDisposableThrows_DoesNotPropagate` (exercises the previously-dead branch)
- `ReleaseAsync_AsyncDisposable_DisposesViaAsyncPath` (asserts one async disposal, sync `Dispose` not called)
- `ReleaseAsync_CalledTwice_DisposesSourceOnceAndDoesNotThrow` (idempotency parity with the sync test)

### 4. CHANGELOG

Under `## [Unreleased]` → `### Fixed`: `DisposeAsync` no longer propagates source-disposal errors;
both `Dispose` and `DisposeAsync` swallow them, matching the dispose-pattern guidance and the
abandoned-resource cleanup path, and their XML docs state the shared contract explicitly (#508).

## Verification steps

1. `dotnet build Nivara.slnx`
2. `dotnet test -c Release --filter "FullyQualifiedName~QuerySourceHandleTests"`
3. Full filtered suite — **only with human confirmation**: `dotnet test -c Release --filter "Category!=Performance"`

## Planned commits

1. `docs: plan #508 disposal-error contract in TODO.md`
2. `fix(query): swallow source-disposal errors in QuerySourceHandle.ReleaseAsync` (+ doc)
3. `docs(query): state the shared Dispose/DisposeAsync swallow contract`
4. `test(query): pin sync/async disposal-error parity incl. the async branch`
5. `docs: record #508 disposal-contract fix in CHANGELOG`
6. `docs: remove TODO.md — plan executed` (after G2)

## GitHub issues log

- none yet — record any deferred work here at discovery time.
