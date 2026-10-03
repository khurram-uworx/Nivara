# Plan: #502 — hold the lazy `QueryFrame` instead of discarding it

Branch: `khurram/502`, created off `khurram/501` (per human instruction).

## Problem

`tests/Nivara.Tests/Streamix/StreamixBridgeIntegrationTests.cs:273`:

```csharp
using var expected = Csv.ScanAsQueryFrame(csvPath).Collect();
```

The `using` binds to the returned `NivaraFrame`. The `QueryFrame` is a discarded
temporary and is the **sole owner** of the `CsvLazySource` — `CsvExtensions.ScanFrame`
news up the source and hands it to exactly one `QueryFrame` (`CsvExtensions.cs:48-49`),
which wraps it in the single `QuerySourceHandle` for that source
(`QueryFrame.cs:26`, `QuerySourceHandle.cs:27-49`).

No deterministic release path exists for it:

| Mechanism | Status |
|---|---|
| `QueryFrame.Dispose()` → `handle.Release()` → `source.Dispose()` | `QueryFrame.cs:1080-1083`, `QuerySourceHandle.cs:65-79` — never called here |
| `NivaraResourceManager` tracking | `QuerySourceHandle.cs:35` — gated on `IsEnabled`, **off by default** (`NivaraResourceManager.cs:20`) |
| Finalizer on `QuerySourceHandle` or `QueryFrame` | **none** |

So with `NivaraResourceManager` disabled — the shipping default — **nothing** releases an
abandoned lazy source. It waits on the inner `FileStream`'s own finalizer.

Even when the manager is enabled it is doubly non-deterministic:
`CleanupAbandonedResources` only fires the cleanup action once `weakRef.Target == null`
(`NivaraResourceManager.cs:235`) — i.e. after the handle has been collected — and only on a
30-second timer (`:31`).

> Note on base-branch drift: this plan was first drafted against `khurram/499`. `khurram/502`
> branches off `khurram/501`, which refactored `QueryFrame` onto a shared `QuerySourceHandle`
> and made `NivaraQuery<T>` disposable. The conclusion is unchanged, but every mechanism
> reference and line number below was re-verified against `khurram/502`.

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
is **correct**, not merely benign. `Sort` returns a frame sharing the same
`QuerySourceHandle` (`QueryFrame.cs:228`), so disposing the surviving frame releases the
source for the whole chain. That is the documented "one source, one release" contract
(`docs/LINQ.md:88-93`) and it is gated by
`QuerySourceHandleTests.DerivedFrame_Abandoned_DoesNotReleaseSourceWhileSiblingLives`.
Left alone deliberately — changing it would add noise for no behavioural gain.

`docs/LINQ.md:99` (`using var grouped = NivaraParquetReader.ScanQuery<Row>(path).GroupBy(...)`)
is correct for the same reason: the `using` binds to the terminal query, which owns the
shared handle.

**Docs — the gap is that `STREAMING.md` teaches a shape `LINQ.md` now forbids.** #501
documented the disposal contract for the typed `ScanQuery` surface (`docs/LINQ.md:73-101`,
including "Dispose the query you created" at `:93`) but left the `ScanAsQueryFrame`
examples untouched. Eleven sites contradict it:

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
| `docs/LINQ.md` | 505 | `var adults = Csv.ScanQuery<Person>(p).Where(...).ToObjects();` — terminal is a `List<Person>`, so **nothing** in the chain is ever disposed. Same shape as the test defect. |
| `docs/LINQ.md` | 64, 66 | `var query = Json.ScanQuery<Person>(...)` / `var csvQuery = Csv.ScanQuery<Person>(...)` — held in locals, never disposed, 10 lines above the guidance that says to wrap them in `using` |

## Proposed changes

### Commit A — `fix(test): hold the lazy QueryFrame in ToFluxRows_ToNivaraFrameAsync_CsvRoundTrips`

```csharp
using var expectedFrame = Csv.ScanAsQueryFrame(csvPath);
using var expected = expectedFrame.Collect();
```

Safe, with three points verified rather than assumed:

- **Disposal order** is reverse-declaration, so `expected` (the `NivaraFrame`) disposes
  before `expectedFrame`. Correct either way.
- **The collected frame cannot be invalidated** by disposing the source: `ReadAllChunks`
  returns columns freshly built by `ColumnFactory.Create` from copied `object?[]` values
  (`CsvDataSource.cs:295, 305`), never the source's chunk columns. All assertions also
  precede both disposals.
- **Double-release is safe if ordering ever changes**: `QuerySourceHandle.Release()`
  guards on `Interlocked.Exchange(ref released, 1)` (`QuerySourceHandle.cs:67`) and
  `QueryFrame.Dispose()` no longer keeps a per-frame `disposed` flag — the whole chain
  shares one release point.

### Commit B — `docs: bind the lazy frame in examples instead of discarding it`

Bind to a named local at the eleven sites. Most are mechanical — each is `await`ed to
completion or `foreach`ed to exhaustion, so `using var` / `await using var` scoped to the
block is correct.

**Three are deferred-lifetime sites and are the real risk in this commit.** `ToFlux()`
returns an `IFlux` built by `Flux.From(queryFrame.AsStream(...))` (`NivaraFlux.cs:19-20`),
so enumeration is deferred and a `using` in the setup method would dispose the frame before
the body is read — converting a leak into a use-after-dispose.

- **354 (ASP.NET controller)** — tie the frame to the response, not to the method:
  `HttpContext.Response.RegisterForDispose(frame)`. Grounded: the API takes an
  `IDisposable` and the host disposes it once the request has finished
  (`Microsoft.AspNetCore.Http`). `QueryFrame` implements `IDisposable`, so it applies
  directly.
- **315 (`.Publish()` fan-out)** — the shared subscription outlives the setup. Show the
  frame being disposed alongside the subscriptions at teardown, with a note on ownership.
- **370 (minimal API `ToSseAsync`)** — mechanically fine (awaited to completion), but sits
  in the same block family as 354; confirm the two read consistently.

`docs/LINQ.md:505` needs its own shape: the terminal is a `List<Person>`, so binding a
local is not enough — the query has to be named, held, and disposed.

## Verification

**The fix's behavioural effect is unobservable on this code path.** A green handle probe
after this change proves nothing about the fix, because it is green now with the bug
present. Do not claim the probe validates it. Two temporary local edits give actual
evidence; neither is committed:

1. **Prove the gate has teeth.** Remove `frame.Dispose();` from
   `ScanAsQueryFrameHandleTests.cs:152` → run that test → **expect FAIL** at
   `AssertUnlocked` (`:154`). Restore. This is the only check that can fail for the
   class #502 belongs to.
2. **Prove the #502 site itself leaks.** Change line 273 to a *partial* read
   (`AsStream(5)` + `break`) and insert `FileHandleProbe.AssertLocked(csvPath, ...)` after
   it → **expect PASS**, i.e. the handle is still open with the old shape. Revert.

Non-regression check only, after the fix:

3. `dotnet test -c Release --filter "FullyQualifiedName~StreamixBridgeIntegrationTests"`
   and `...~ScanAsQueryFrameHandleTests`, plus `...~QuerySourceHandleTests` since the fix
   depends on that contract. Release, matching CI. Capture the exit status of
   `dotnet test` itself, not of a pipeline stage. **Green means "not regressed", not "the
   fix was load-bearing".**
4. **The docs commit has no mechanical verification** — the snippets are not compiled.
   They are checked by review against real signatures (`AsStream(int chunkSize = 10000,
   CancellationToken ct)` — `QueryFrame.cs:461`; `ToFlux(QueryFrame, int chunkSize,
   ChannelBackpressureMode, int, string?)` — `NivaraFlux.cs:10-15`).

## Blast radius

- **Commit A** — one line in one test fixture. No public API, no production code, no other
  caller. Teardown interaction: `StreamixBridgeIntegrationTests.TearDown` does
  `Directory.Delete(tempDir, true)`, which is where a leaked handle would surface on
  Windows (`IOException`). Not observable on `ubuntu-latest`.
- **Commit B** — documentation only, zero runtime blast radius. Two of the eleven sites
  (354, 315) encode lifetime semantics that a naive `using` would actively break, so the
  diff needs review rather than a mechanical sweep.
- **Explicitly out of scope:** flipping `NivaraResourceManager` on by default, adding a
  finalizer to `QuerySourceHandle`, and any static gate for the class of bug that is
  invisible at runtime. #501 is already landed on this branch's base.

## GitHub issues log

- [ ] #NNN — stale API in doc snippets (`Filter("status","OK")`, `AsStream(enforced: true)`,
  `Csv.ScanAsQueryFrame(path, chunkSize)`, `Parquet.ScanAsQueryFrame`) — created while
  working on #502