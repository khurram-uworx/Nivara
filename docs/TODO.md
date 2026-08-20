# Deep Streamix Integration — Issue #171

## Problem

Issue #171's initial Streamix bridge landed (PR #316) with 4 methods. The second comment
on #171 lists 6 deeper integration ideas that would complete the bidirectional pipeline
and enable event-time windowing, mini-batch framing, and ASP.NET Core streaming.

## Scope

### API additions (new code in `src/Nivara.Extensions/Streamix/NivaraFlux.cs`)

1. **`IFlux<NivaraRow>.ToNivaraFrame(ct)`** — row-level reverse terminal.
   Buffers rows via `ToListAsync`, infers schema from first row's column names/types,
   allocates typed arrays, fills from buffered rows, builds `NivaraFrame`.

2. **`QueryFrame.ToFluxWithTimestamp(Func<NivaraRow, DateTimeOffset>, chunkSize, name)`
   → `IFlux<Timestamped<NivaraRow>>`** — event-time bridge.
   Chains `ToFluxRows(chunkSize)` → `.Map(row => Timestamped.Create(row, selector(row)))`.
   Convenience overload on `NivaraFrame`.

3. **`IFlux<NivaraRow>.BufferByCount(int, ct)` → `IFlux<IList<NivaraRow>>`** — wraps
   Streamix `.Buffer(count)` with cancellation.

4. **`IFlux<NivaraRow>.BufferFrames(int, ct)` → `IFlux<NivaraFrame>`** — mini-batch
   framing. Chains `.BufferByCount(batchSize)` → `.Map(rows.ToNivaraFrame())`.

### Documentation additions (`docs/STREAMING.md`)

5. **Publish/Replay fan-out pattern** — usage guidance section.
6. **ASP.NET Core SSE** — controller example with `Streamix.AspNetCore`.
7. **Diagnostic operators** — `.Checkpoint()`, `.Trace()`, `.Named()` usage.

## Blast radius

- **Modified:** `src/Nivara.Extensions/Streamix/NivaraFlux.cs` (1 file, ~120 lines added)
- **Tests:** `tests/Nivara.Tests/Streamix/StreamixBridgeTests.cs`, `StreamixBridgeIntegrationTests.cs`
- **Docs:** `docs/STREAMING.md`
- **Downstream:** none — all new public API on the existing `NivaraFlux` static class
- **No new project references** — uses existing `Streamix` package only
- **No core changes** — `src/Nivara/` untouched

## Planned commits

1. `docs: plan deep Streamix integration in TODO.md`
2. `feat: add IFlux<NivaraRow>.ToNivaraFrame() row-level reverse terminal`
3. `test: add unit + integration tests for ToNivaraFrame row-level round-trip`
4. `feat: add ToFluxWithTimestamp event-time bridge`
5. `test: add tests for ToFluxWithTimestamp`
6. `feat: add BufferByCount + BufferFrames mini-batch framing`
7. `test: add tests for BufferByCount and BufferFrames`
8. `docs: add Publish/Replay, ASP.NET Core SSE, and diagnostic operator sections`

## GitHub issues log

- [ ] none yet
