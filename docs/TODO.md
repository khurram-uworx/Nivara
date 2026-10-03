# Plan — #501: the typed LINQ query surface is undisposable, and query-source lifetime is tracked per-frame instead of per-source

Branch: `khurram/501` (off `khurram/499`). PR target: `khurram/499`.

## Problem

Two defects, one reported and one found while planning the fix. The second is the real one.

### 1. Reported (#501): no `using` on the typed LINQ surface

`NivaraQuery<T>` (`src/Nivara/Linq/NivaraQuery.cs:19`) is `public sealed class` with no interface.
It holds `readonly QueryFrame frame`; the only release path is `query.AsQueryFrame().Dispose()`.

`NivaraGroupedQuery<TKey, T>` (`NivaraQuery.cs:410`) is worse — it holds `baseFrame` and has **no
`AsQueryFrame()` at all**, so `NivaraParquetReader.ScanQuery<T>(p).GroupBy(k).Collect()` has no
caller-side release path whatsoever.

The lazy sources open their file with `FileShare.Read` and no `FileShare.Delete`
(`CsvDataSource.cs:583`, `JsonStreamReader.cs:86-87`, `ParquetDataSource.cs:396`), so a leaked handle
blocks deletion on Windows — the misleading "being used by another process" teardown failure behind
#496 and #498.

### 2. Found while planning: the resource's lifetime is tracked per-frame, but the resource is shared per-source

`QueryFrame`'s **derived** constructor (`QueryFrame.cs:56-77`) — the one every fluent method calls —
does this:

```csharp
if (source.IsLazy && NivaraResourceManager.IsEnabled)
    NivaraResourceManager.TrackResource(this, "LazyQueryFrame", 0, () => { source?.Dispose(); });
```

It tracks **every derived frame**, not just the root, with a cleanup action that disposes **the
shared source**. `CleanupAbandonedResources` (`NivaraResourceManager.cs:220-266`) fires that action
when `weakRef.Target == null` — when *that frame* is unreachable.

```csharp
NivaraResourceManager.Enable();
var root = Csv.ScanAsQueryFrame(path);   // tracked -> action disposes source
var mid  = root.Filter(c1);              // tracked -> action disposes source
var leaf = mid.Filter(c2);               // tracked -> action disposes source
// `mid` is now unreachable
GC.Collect(); NivaraResourceManager.ForceCleanup();   // W(mid).Target == null -> source.Dispose()
leaf.Collect();                          // dead - killed by a garbage-collection timer
```

**This is a live correctness bug**, not a hypothetical. `NivaraGroupedQuery.Collect()` manufactures
that exact shape on every call: `baseFrame.WithOperation(...)` (`:431`) creates a tracked frame that
is immediately garbage.

The existing gate `ResourceManagementPropertyTests.cs:153` only exercises a single un-derived root,
which is why it never caught this.

`#501`'s naive fix (implement `IDisposable` and forward to the frame) would have **left this bug in
place and added a second, easier route into it**: `using` on a derived query releases the source out
from under its siblings, exactly as the timer does.

## Why not reference counting

Considered and rejected. Derived frames are temporaries inside expressions
(`q.Where(...).Collect()`) and are essentially never disposed, so a per-frame count never reaches
zero:

```csharp
using var q = Csv.ScanQuery<Person>(path);   // count 1
var d = q.Where(...);                        // count 2, never disposed
d.Collect();                                 // ok
// scope exit: q.Dispose() -> count 1. File never released. Leak.
```

Refcounting turns the common case from "works" into "leaks". Closing that gap needs a finalizer per
frame, which reintroduces exactly the nondeterminism #501 exists to remove. Wrong tool.

**Non-goal, stated deliberately:** independent sibling lifetimes require GC timing. One source, one
release is the correct contract for a shared resource (the same contract `FileStream` has), and it is
made *coherent and documented* rather than accidental.

## Proposed changes

### 1. `src/Nivara/Query/QuerySourceHandle.cs` (new)

Internal, non-generic. One instance per source, created only by the root constructor.

- wraps one `IQuerySource`
- `Source`, `Schema`, `IsLazy` forward to it
- `bool Released`
- `Release()` is idempotent — untracks, then disposes (async-aware) on first call only
- registers the `NivaraResourceManager` entry **here, once**, keyed on the handle

Every frame references the handle, so the handle's reachability *is* "can anyone still reach this
source". Tracking the handle instead of the frame makes the cleanup condition exactly correct: the
timer can no longer fire while a live frame exists.

### 2. `src/Nivara/Query/QueryFrame.cs`

- `readonly IQuerySource source` -> `readonly QuerySourceHandle handle`
- derived ctor takes the handle: `QueryFrame(QuerySourceHandle, IEnumerable<IQueryOperation>)`
- **root ctor signature unchanged** (`QueryFrame(IQuerySource)`), so all 8 root call sites are
  untouched (`NivaraFrame.cs:322`, `JsonExtensions.cs:49`, `CsvExtensions.cs:49`,
  `NivaraParquetReader.cs:113`, + 4 test sites)
- delete `bool disposed`; every guard reads `handle.Released`
- `Dispose` / `DisposeAsync` -> `handle.Release()`
- 18 derivation sites `new QueryFrame(source, ...)` -> `new QueryFrame(handle, ...)`
- 6 `new QueryPlan(source, ...)` -> `new QueryPlan(handle.Source, ...)`
- `ToString()` (`:587`,`:589`) uses `handle.Source.GetType().Name`; output must stay byte-identical
  (`QueryFrameTests.QueryFrame_ToString_ReturnsDescription` asserts on it)

**Behavior change (confirmed with human):** delegating guards to `handle.Released` means a *sibling*
derived frame now also throws `ObjectDisposedException` after another link is disposed, where it
previously passed its own guard and failed deeper inside the source wrapped in
`QueryExecutionException`. This is the intended, consistent outcome. Verified by the full suite in
step 4 — any test that depended on the old inconsistency is a finding to report, not to patch.

### 3. `src/Nivara/Linq/NivaraQuery.cs`

- `NivaraQuery<T>` and `NivaraGroupedQuery<TKey, T>` implement `IDisposable`, `IAsyncDisposable`,
  forwarding to the underlying frame
- class-level `<remarks>` on both stating the shared-source rule

`DisposeAsync` is a forward only: **no `IQuerySource` in `src/` implements `IAsyncDisposable`**, so it
is future-proofing and is described as such rather than implying async release exists.

### 4. Tests

**Repoint — `tests/Nivara.Tests/IO/ScanAsQueryFrameHandleTests.cs`.** Mandatory, not cleanup: PR #503
commit `769535e` put the pre-#501 constraint into the *failure text* of all three typed cases
(`:137`, `:178-179`, `:220-221` — "do not 'fix' this by adding a 'using'"). The day #501 lands that
text is actively misleading, suppressing the correct fix.

| case | test | comment | release call | failure text |
|------|------|---------|--------------|--------------|
| Parquet | 116 | 130 | 133 | 137 |
| CSV | 162 | 174 | 175 | 178-179 |
| JSON | 204 | 216 | 217 | 220-221 |

Swap `query.AsQueryFrame().Dispose()` -> `query.Dispose()`, rename
`*_ReleasesHandleOnAsQueryFrameDispose*` -> `*_ReleasesHandleOnDispose*`, rewrite the #501 guidance.

**New — `tests/Nivara.Tests/Query/QuerySourceHandleTests.cs`** (per AGENTS.md: one test class per
source class):

1. `Release_IsIdempotent` — two releases, no throw, handle released once.
2. `DerivedFrame_Abandoned_DoesNotReleaseSourceWhileSiblingLives` — **the regression test for the
   timer bug.** Root -> mid -> leaf, drop `mid`, GC, `ForceCleanup()`, assert `leaf` still reads.
   Goes red on today's code.
3. `Frame_Disposed_SiblingFrame_ThrowsObjectDisposedException` — pins the behavior change in 2.
4. `Frame_Disposed_SiblingFrame_ReportsSameDisposedState` — `IsLazy`/`Schema` agree across a chain.

**New — `tests/Nivara.Tests/Query/NivaraQueryDisposalTests.cs`**:

1. `NivaraQuery_ImplementsDisposableAndAsyncDisposable` — type-shape gate for both types. This is
   what makes "forgot to dispose" compile-time-visible, as #501 asks.
2. `NivaraQuery_UsingDeclaration_ReleasesFileHandle` — real `using var`, mid-file `break`, scope exit,
   `FileHandleProbe.AssertUnlocked`.
3. `NivaraGroupedQuery_Dispose_ReleasesFileHandle` — **Parquet**, full `Collect()` through
   `.GroupBy(...).Select(...)`, `AssertLocked` -> dispose -> `AssertUnlocked`. Parquet because
   disposal is load-bearing there; CSV/JSON self-release at EOF and would pass vacuously.
4. `NivaraQuery_Dispose_DoesNotDisposeSourceFrameColumns` + `DisposeAsync` twin — mirrors
   `QueryFrameTests.cs:332-355`.
5. `NivaraQuery_Collect_AfterDispose_ThrowsObjectDisposedException` — mirrors `QueryFrameTests.cs:342`.
6. `NivaraQuery_DisposingDerivedQuery_InvalidatesSiblingChain` — pins the ownership decision.
   Asserts a throw whose chain contains `ObjectDisposedException`, **not**
   `Throws.TypeOf<ObjectDisposedException>()`: the disposed object is the *source*, not the frame, so
   `CollectAsync`'s catch-all (`:434`) wraps it in `QueryExecutionException`. Pinning the outer type
   would pin an accident of the wrapper.

**Gate constraint carried from #503:** the leak window for CSV/JSON is **partial-read only** — a full
`Collect()` reaches EOF and self-releases (`CsvDataSource.cs:405`). #503 verified by probing that
rewriting a case as a full `Collect()` with no disposal at all still **passes**. So every new
handle-release case must either break out of `AsStream` mid-file and assert `AssertLocked` first, or
use Parquet. A full-`Collect` case would be vacuous.

`FileHandleProbe` (`tests/Nivara.Tests/IO/FileHandleProbe.cs`) is reused unchanged — one
authoritative probe, per AGENTS.md rule 8.

### 5. Docs

- XML: add Parquet-style `<remarks>` to `Csv.ScanQuery<T>` (`CsvExtensions.cs:76`) and
  `Json.ScanQuery<T>` (`JsonExtensions.cs:76`) — neither has one today. Rewrite
  `NivaraParquetReader.cs:139-142` and `:125-126`, which instruct callers to reach through
  `AsQueryFrame()`.
- `docs/LINQ.md`: "Resource management" subsection under "From file sources (lazy)" (near `:69-71`).
- `CHANGELOG.md`: `## [Unreleased]` -> `### Added`, plus the `QueryFrame` behavior change.

## Blast radius

- **`src/Nivara/Query/QueryFrame.cs`** — 2 ctors, field, 2 Dispose methods, 18 derivation sites,
  6 `QueryPlan` sites, every `disposed` guard. Public behavior change (see 2).
- **`src/Nivara/Query/QuerySourceHandle.cs`** — new, internal.
- **`src/Nivara/Linq/NivaraQuery.cs`** — 2 public types gain 2 interfaces each. Additive; no existing
  caller breaks. `NivaraQuery<T>.ToList()`/`ToObjects()` are declared on the type, and no extension in
  `src/` takes `IDisposable`, so there is no overload-resolution change.
- **Root `QueryFrame` ctor sites — unchanged** (8 of them).
- **Tests covering this:** `ScanAsQueryFrameHandleTests` (6 cases + negative control),
  `QueryFrameTests`, `AsyncStreamingTests`, `ResourceManagementPropertyTests`, `NivaraQueryFeatureTests`,
  `NivaraQueryToObjectsAsyncTests`, `LinqQueryTests`, `ParquetStreamingTests`, `JsonStreamingTests`,
  and #499's new Incident-surface gate (PR #505) — which now exercises `NivaraGroupedQuery` via
  `Analysis.AnalyzeGroupedAggregationWithTypedLinq`.
- **First-party `NivaraGroupedQuery` consumer is not a handle leak.** `Analysis.cs:250` already does
  `using var frame = Ingestion.LoadParquet(...)` and the typed path runs off the in-memory `collected`
  -> `MemoryQuerySource`, which holds no handle. `NivaraGroupedQuery`'s missing release path stays a
  **library-surface gap**, reachable only via `ScanQuery<T>(parquet).GroupBy(...)`.

## Verification steps

1. `dotnet build Nivara.slnx` — 0 warnings, 0 errors.
2. `QuerySourceHandleTests` targeted, `-c Release`. **Step 2's test 2 must be verified RED against
   pre-change `main`** (stash the src change, run it) — a regression test that was never seen red
   proves nothing.
3. `ScanAsQueryFrameHandleTests` + `NivaraQueryDisposalTests` + `QueryFrameTests` +
   `ResourceManagementPropertyTests` + `AsyncStreamingTests`, `-c Release`. Negative control
   (`Probe_DetectsDeliberatelyLeakedHandle`) must stay green — it is what gives the gate teeth on
   `ubuntu-latest`.
4. Full suite `-c Release --filter "Category!=Performance"`, capturing the **process** exit status,
   not a pipeline filter's. This is what confirms the `handle.Released` behavior change is safe.
5. Re-run 3 to confirm the negative control is stable, not flaky.

## Planned commits

1. `docs: plan #501 in TODO.md`
2. `refactor: track query-source lifetime on a shared handle instead of per-frame`
3. `test: gate the shared-source release contract and the abandoned-frame regression`
4. `feat: implement IDisposable and IAsyncDisposable on the typed LINQ query types`
5. `test: repoint the ScanQuery handle gate at query.Dispose()`
6. `docs: update ScanQuery disposal guidance in XML docs, LINQ.md and CHANGELOG`

> As each task executes, if you find deferred work or a concern outside this plan, create a tracked
> issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number below — do
> not rely on memory, as compaction during execution can lose items.

## GitHub issues log

- [ ] #501 — `NivaraQuery<T>` is not `IDisposable`, so `ScanQuery<T>` consumers have no `using` and
  must call `query.AsQueryFrame().Dispose()`. **This work**, widened to include the shared-source
  lifetime defect above.
- [ ] #502 — `StreamixBridgeIntegrationTests.cs:273` abandons a `QueryFrame`
  (`Csv.ScanAsQueryFrame(...).Collect()` binds to the result, not the frame). Different type, out of
  scope here. Note from #503: the handle gate does **not** catch it and cannot validate a fix by
  "probe is green afterwards" — it is green now with the bug present. A partial read is the check
  with teeth.
- [ ] #498 / PR #503 — gate for the `ScanAsQueryFrame` consumers. Lands in this branch's history;
  commit 5 repoints its typed cases.
- [ ] #499 / PR #505 — Incident-surface gate. In this branch's history; adds the live
  `NivaraGroupedQuery` consumer that must stay green.