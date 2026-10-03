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

**Docs — the gap is that the examples teach a shape `LINQ.md` now forbids.** #501
documented the disposal contract for the typed `ScanQuery` surface (`docs/LINQ.md:73-101`,
including "Dispose the query you created" at `:94`) but left the example snippets
untouched, in `docs/` and at the repo root. Fifteen sites contradict it:

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
| `GETTING-STARTED.md` | 355 | `var customCsv = Csv.ScanQuery<Employee>(p, opts).Collect();` — **identical to the test defect**, in the onboarding doc |
| `GETTING-STARTED.md` | 343 | `var csvQuery = Csv.ScanQuery<Employee>(...)...; var result = csvQuery.Collect();` — held, never disposed |
| `GETTING-STARTED.md` | 373 | `var jsonQuery = Json.ScanQuery<User>(...)...; var jsonResult = jsonQuery.Collect();` — held, never disposed |
| `GETTING-STARTED.md` | 940 | `var query = Csv.ScanQuery<Employee>(...)...; var result = query.Collect(); query.ExplainPlan();` — held, never disposed, and used after `Collect()` |

> **Sweep correction.** The first pass reported "eleven doc sites" and was wrong: the grep
> was scoped to `docs/`, so it never saw the repo-root markdown. A repo-wide sweep during
> execution added the four `GETTING-STARTED.md` rows, one of which (`:355`) is the same
> defect as the test. Recorded rather than quietly absorbed, because the original plan text
> is the thing G2 checks the branch against.

Also swept and confirmed **clean**: no `.cs` hit outside the one test line, no
`Csv`/`Json` reader pointed at a `.parquet` path outside the three `STREAMING.md`
instances corrected in this commit, and no other root-level markdown (`README.md`,
`ARCHITECTURE.md`, `EXAMPLES.md`, `CHANGELOG.md`) containing a discarded-scan shape.

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

### Commit B — `docs: make the examples obey the disposal contract #501 documented`

Bind to a named local at the fifteen sites across four files. Most are mechanical — each is
`await`ed to completion or `foreach`ed to exhaustion, so `using var` / `await using var`
scoped to the block is correct.

**Also rewrites the `STREAMING.md` "Resource management" section** (`:164-190`). The examples
are only self-consistent if the section states the rules they now follow, so this commit adds
four bullets: dispose the scan you created; one source, one release (cross-linked to the
authoritative `LINQ.md#resource-management`); a full read masks a missing `Dispose`; and
deferred operators need the frame to outlive setup. Without them the new `await using` lines
read as unexplained verbosity.

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

`docs/LINQ.md:505` needs its own shape: `ToObjects()` returns `IReadOnlyList<T>`
(`NivaraQuery.cs:358`), which is not disposable, so binding a local to the result is not
enough — the query has to be named, held, and disposed. `GETTING-STARTED.md:355` is the same:
`Collect()` returns the disposable `NivaraFrame` (`NivaraQuery.cs:303`), so a `using` on the
chained expression binds to the result and the query still needs its own.

**Three snippets are edited for both disposal and API correctness** (human decision at G1,
extended during execution). Where a snippet is already being rewritten for disposal, leaving a
call that cannot work would be worse than leaving either problem alone:

- `docs/ACCELERATION.md:214-216` — `.AsStream(enforced: true)` and `.Filter("status", "OK")`
  have no counterpart in `QueryFrame.AsStream(int chunkSize = 10000, CancellationToken ct)`
  (`QueryFrame.cs:461`) or `QueryFrame.Filter(ColumnExpression)` (`QueryFrame.cs:110`). The
  inline comment "chunk memory is guaranteed to stay within the budget" was propped up by the
  fake `enforced:` parameter, so it goes too, replaced by a pointer to the execution-context
  route shown directly above it in the same file. The yielded chunk was also never disposed,
  which violates the rule the commit is enforcing, so it is wrapped.
- `docs/STREAMING.md:135` — the same `.Filter("status", "OK")`, in the same code block as
  site 153, so the two are corrected together or not at all.
- `docs/STREAMING.md:222, 315, 386` — `Csv.ScanAsQueryFrame("*.parquet")`. A CSV reader on a
  Parquet file fails at read time; the correct factory is `NivaraParquetReader.ScanAsQueryFrame`.
  All three instances were being edited for disposal anyway, and making two of three correct
  would be worse than making none.

`QueryFrame.Select(params string[] columnNames)` **does** exist (`QueryFrame.cs:121`), so
`.Select("timestamp", "value")` at `STREAMING.md:136` was already valid. An earlier draft of
this commit "fixed" it to `ColumnExpressions.Col(...)`; that was reverted rather than leave an
unnecessary change in the diff. Verified against the signature, not assumed.

Stale mentions in files this branch does not otherwise touch (`ARCHITECTURE.md:356-358`,
`ARCHITECTURE.md:319`, `ARCHITECTURE.md:1024`, `README.md:84`,
`samples/NivaraIncident/README.md:83, 167-168`) are filed as #510 rather than folded in.
`CHANGELOG.md:443-444` is left alone deliberately: a changelog entry is a historical record of
what shipped, not a usage example.

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

### Verification results (all run on Windows, Release)

| # | Check | Expected | Actual |
|---|---|---|---|
| 1 | Drop `frame.Dispose()` from the CSV partial-read gate | red | **red, exit 1** — `AssertUnlocked` fired with correct attribution |
| 2 | Abandon the frame mid-file in the #502 fixture, then probe | locked | **locked, and TearDown failed** with `IOException: ... is being used by another process` |
| 3 | Non-regression across 4 fixtures | green | **30 passed, exit 0** |

Two findings that changed what could be claimed:

- **The Windows teardown symptom is provable after all.** The issue reasons that
  `Directory.Delete` cannot fail on `ubuntu-latest` and stops there. CI is Linux-only, but the
  verification host is Windows, where neither masking condition applies. So check 2 produced
  the actual #496/#502 failure rather than a proxy reading.
- **`$LASTEXITCODE` after a pipe reports the pipe's status.** The first pass printed
  `EXITCODE=0` next to `Failed: 1`. Every result above was re-captured without a pipeline.
  This is the `AGENTS.md` "capture the exit status of the process you care about" rule biting.

### G2 review finding — one commit superseded

The docs commit wrote `shared.Subscribe(...)` and disposed its return value. **`Subscribe` does
not exist in Streamix.** Confirmed three ways: reflection over Streamix 1.2.3, a grep of the
Streamix source for `public static .*Subscribe`, and its public interface list. The real API is
`Publish`/`Replay` → `IConnectableStream<T> : IFlux<T>` with `Connect() -> IDisposable`,
`RefCount()`, `WhenRefCountDisconnectedAsync()`. There is no subscribe call because none is
needed — enumerating the connectable stream is what subscribes, and `ForEachAsync` returns a
`Task`, not a disposable.

Fixed in an additional commit rather than by rewriting history. The ownership handle the
section needed already existed and was being discarded: `Connect()` returns the `IDisposable`
that disconnects the shared subscription, which belongs beside the frame.

Every other Streamix operator in these docs was verified real in the same pass — `Named`,
`Retry`, `Checkpoint`, `WindowByTime`, `FlatMap`, `Trace`, `Log`, `Filter`, `ToSseAsync`,
`FluxResult`. `Subscribe` was the only one missing. Checked upstream
(`khurram-uworx/streamix`) for a matching gap and found none: Streamix's own
`GETTING-STARTED.md:175-176` already demonstrates `Connect()` + `ForEachAsync`, so this was
our documentation error, not a Streamix defect. No upstream issue filed — filing one would be
noise.

Subscribe-then-connect ordering was checked in `ConnectableStream.cs`, not assumed:
`GetAsyncEnumerator` registers an unbounded channel per subscriber (`:305-324`) before
`Connect()` starts the source enumeration (`:278-296`), so no items are lost.

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

- [x] #510 — stale API in doc snippets (`Filter("status","OK")`, `AsStream(enforced: true)`,
  `Csv.ScanAsQueryFrame(path, chunkSize)`, `Parquet.ScanAsQueryFrame`) — created while
  working on #502. The two snippets #502 edits are fixed there; the rest is #510.