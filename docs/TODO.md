# Plan: #502 — hold the lazy `QueryFrame` instead of discarding it

Branch: `khurram/502`, created off `khurram/501` (per human instruction).

## Problem

`tests/Nivara.Tests/Streamix/StreamixBridgeIntegrationTests.cs:273`:

```csharp
using var expected = Csv.ScanAsQueryFrame(csvPath).Collect();
```

The `using` binds to the returned `NivaraFrame`. The `QueryFrame` is a discarded
temporary and is the **sole owner** of the `CsvLazySource` — `CsvExtensions.ScanFrame`
news up the source and hands it to exactly one `QueryFrame` (`CsvExtensions.cs:48-49`).

No deterministic release path exists for it:

| Mechanism | Status |
|---|---|
| `QueryFrame.Dispose()` → `source.Dispose()` | `QueryFrame.cs:1111-1128` — never called here |
| `NivaraResourceManager` tracking | `QueryFrame.cs:33` — gated on `IsEnabled`, **off by default** (`NivaraResourceManager.cs:20`) |
| Finalizer on `QueryFrame` | none |

Even when the manager is enabled it is doubly non-deterministic:
`CleanupAbandonedResources` only fires the cleanup action once `weakRef.Target == null`
(`NivaraResourceManager.cs:235`) — i.e. after the frame has been *finalized* — and only
on a 30-second timer (`:31`).

### Why it is invisible today

`Collect()` → `Execute()` → `ReadAllChunks()` (`CsvDataSource.cs:217-309`) loops
`ReadChunk` until an empty chunk; the final `ReadChunk` hits `!csv.Read()` and calls
`DisposeChunkReader()` (`CsvDataSource.cs:402-407`), releasing the handle with no
`Dispose` involved. The source's own EOF self-release masks the abandonment.

Consequence: a partial read (filter, cancellation, early break) would leave
`chunkCsvReader` non-null and the handle genuinely open. CI is `ubuntu-latest` only
(`.github/workflows/ci.yml:13`), where Unix unlinks open files unconditionally, so a
delete-based assertion cannot fail there either.

## Sweep results

**Code — one hit, this line.** Two regex sweeps across all `*.cs`:

- `ScanAsQueryFrame\([^)]*\)\s*\.` → only `StreamixBridgeIntegrationTests.cs:273` and
  `AsyncStreamingTests.cs:796`
- `using var X = (Csv|Json|NivaraParquetReader|Parquet).M(...).Op(...)` → only
  `StreamixBridgeIntegrationTests.cs:273`

`AsyncStreamingTests.cs:796` (`using var queryFrame = Csv.ScanAsQueryFrame(csvPath).Sort("Age")`)
is **benign**: `Sort` returns `new QueryFrame(source, ...)` (`QueryFrame.cs:259-271`), so the
surviving frame holds the *same* source instance and its `Dispose()` releases the handle. The
dropped intermediate owns nothing unique. Left alone deliberately — changing it would add
noise for no behavioural gain.

**Docs — nine hits.** The discarded-terminal-frame shape is the shape agents copy:

| File | Line | Shape |
|---|---|---|
| `docs/STREAMING.md` | 153 | `await foreach (... in Csv.ScanAsQueryFrame(p).Sort(...).AsStream())` |
| `docs/STREAMING.md` | 221 | `await Csv.ScanAsQueryFrame(p).Filter(...).ToFlux(...)...ForEachAsync(...)` |
| `docs/STREAMING.md` | 234 | `await Csv.ScanAsQueryFrame(p).ToFluxWithTimestamp(...)...` |
| `docs/STREAMING.md` | 284 | `var fluxRows = Csv.ScanAsQueryFrame(p).ToFluxRows(...)` — held, never disposed |
| `docs/STREAMING.md` | 315 | `var shared = Csv.ScanAsQueryFrame(p)...Publish()` — held, never disposed |
| `docs/STREAMING.md` | 354 | controller: `var flux = Csv.ScanAsQueryFrame(p)...ToFlux(...)` returned as `FluxResult` |
| `docs/STREAMING.md` | 370 | minimal API: `await Csv.ScanAsQueryFrame(p)...ToSseAsync(response)` |
| `docs/STREAMING.md` | 386 | `await Csv.ScanAsQueryFrame(p).ToFlux(...)...ForEachAsync(...)` |
| `docs/ACCELERATION.md` | 214 | `await foreach (... in Csv.ScanAsQueryFrame(p).Filter(...).AsStream(...))` |

## Proposed changes

### Commit A — `fix(test): hold the lazy QueryFrame in ToFluxRows_ToNivaraFrameAsync_CsvRoundTrips`

```csharp
using var expectedFrame = Csv.ScanAsQueryFrame(csvPath);
using var expected = expectedFrame.Collect();
```

Safe, with two points verified rather than assumed:

- **Disposal order** is reverse-declaration, so `expected` (the `NivaraFrame`) disposes
  before `expectedFrame`. Correct either way.
- **The collected frame cannot be invalidated** by disposing the source: `ReadAllChunks`
  returns columns freshly built by `ColumnFactory.Create` from copied `object?[]` values
  (`CsvDataSource.cs:295, 305`), never the source's chunk columns. All assertions also
  precede both disposals.

### Commit B — `docs: bind the lazy frame in streaming examples instead of discarding it`

Bind to a named local at all nine sites. Six are mechanical — each is `await`ed to
completion or `foreach`ed to exhaustion, so `using var` / `await using var` scoped to the
block is correct.

**Three are deferred-lifetime sites and are the real risk in this commit.** `ToFlux()`
returns an `IFlux` whose enumeration is deferred, so a `using` in the setup method would
dispose the frame before the body is read — converting a leak into a use-after-dispose.

- **354 (ASP.NET controller)** — tie the frame to the response, not to the method:
  `HttpContext.Response.RegisterForDispose(frame)`.
- **315 (`.Publish()` fan-out)** — the shared subscription outlives the setup. Show the
  frame being disposed alongside the subscriptions at teardown, with a note on ownership.
- **370 (minimal API `ToSseAsync`)** — mechanically fine (awaited to completion), but sits
  in the same block family as 354; confirm the two read consistently.

## Verification

**The fix's behavioural effect is unobservable on this code path.** A green handle probe
after this change proves nothing about the fix, because it is green now with the bug
present. Do not claim the probe validates it. Two temporary local edits give actual
evidence; neither is committed:

1. **Prove the gate has teeth.** Remove `frame.Dispose();` from
   `ScanAsQueryFrameHandleTests.cs:156` → run that test → **expect FAIL** at
   `AssertUnlocked` (`:158`). Restore. This is the only check that can fail for the
   class #502 belongs to.
2. **Prove the #502 site itself leaks.** Change line 273 to a *partial* read
   (`AsStream(5)` + `break`) and insert `FileHandleProbe.AssertLocked(csvPath, ...)` after
   it → **expect PASS**, i.e. the handle is still open with the old shape. Revert.

Non-regression check only, after the fix:

3. `dotnet test -c Release --filter "FullyQualifiedName~StreamixBridgeIntegrationTests"`
   and `...~ScanAsQueryFrameHandleTests`. Release, matching CI. Capture the exit status of
   `dotnet test` itself, not of a pipeline stage. **Green means "not regressed", not "the
   fix was load-bearing".**
4. **The docs commit has no mechanical verification** — the snippets are not compiled.
   They are checked by review against real signatures (`AsStream(int chunkSize,
   CancellationToken ct)` — `QueryFrame.cs:492`; `ToFlux(QueryFrame, int,
   ChannelBackpressureMode, int, string?)` — `NivaraFlux.cs:10`).

## Blast radius

- **Commit A** — one line in one test fixture. No public API, no production code, no other
  caller. Teardown interaction: `StreamixBridgeIntegrationTests.TearDown` does
  `Directory.Delete(tempDir, true)`, which is where a leaked handle would surface on
  Windows (`IOException`). Not observable on `ubuntu-latest`.
- **Commit B** — documentation only, zero runtime blast radius. Two of the nine sites
  (354, 315) encode lifetime semantics that a naive `using` would actively break, so the
  diff needs review rather than a mechanical sweep.
- **Explicitly out of scope:** #501 (`NivaraQuery<T>` is not `IDisposable`), flipping
  `NivaraResourceManager` on by default, adding a `QueryFrame` finalizer, and any static
  gate for the class of bug that is invisible at runtime.

## GitHub issues log

- [ ] #NNN — stale API in doc snippets (`Filter("status","OK")`, `AsStream(enforced: true)`,
  `Csv.ScanAsQueryFrame(path, chunkSize)`, `Parquet.ScanAsQueryFrame`) — created while
  working on #502