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
- `Release()` is idempotent — untracks, then disposes on first call only
- `ReleaseAsync()` — same, async-aware
- registers the `NivaraResourceManager` entry **here, once**, keyed on the handle

Every frame references the handle, so the handle's reachability *is* "can anyone still reach this
source". Tracking the handle instead of the frame makes the cleanup condition exactly correct: the
timer can no longer fire while a live frame exists.

**The cleanup closure must capture the source, never the handle.** `_trackedResources` is a static
dictionary holding `ResourceInfo.CleanupAction`, so anything the closure captures stays reachable
forever. Writing `() => Release()` would capture `this`, keep the handle permanently alive, and make
`weakRef.Target == null` unsatisfiable — silently disabling the abandoned-resource cleanup this
whole mechanism exists for, with no error. The closure must close over the raw `IQuerySource`
local, which is what today's code does (it captures `source`, not `this`) and why the frame's weak
reference can currently go null at all.

### 2. `src/Nivara/Query/QueryFrame.cs`

- `readonly IQuerySource source` -> `readonly QuerySourceHandle handle`
- derived ctor takes the handle: `QueryFrame(QuerySourceHandle, IEnumerable<IQueryOperation>)`
- **root ctor signature unchanged** (`QueryFrame(IQuerySource)`), so all 8 root call sites are
  untouched (`NivaraFrame.cs:322`, `JsonExtensions.cs:49`, `CsvExtensions.cs:49`,
  `NivaraParquetReader.cs:113`, + 4 test sites)
- delete `bool disposed`; every guard reads `handle.Released` (23 `ObjectDisposedException.ThrowIf`
  sites, plus the `if (disposed)` at `:580` in `ToString`)
- `Dispose` -> `handle.Release()` (swallows source disposal errors, as today);
  `DisposeAsync` -> `handle.ReleaseAsync()` (propagates them, as today — see Grounding)
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
2. `Handle_Abandoned_ReleasesSourceWhenNothingReferencesIt` — guards the closure-capture trap:
   enable the resource manager, build a frame, drop every reference, GC, `ForceCleanup()`, assert the
   source was released. Silently passes forever if the closure ever captures the handle.
3. `DerivedFrame_Abandoned_DoesNotReleaseSourceWhileSiblingLives` — **the regression test for the
   timer bug.** Root -> mid -> leaf, drop `mid`, GC, `ForceCleanup()`, assert `leaf` still reads.
   Goes red on today's code.
4. `Frame_Disposed_SiblingFrame_ThrowsObjectDisposedException` — pins the behavior change in 2.
5. `Frame_Disposed_SiblingFrame_ReportsSameDisposedState` — `IsLazy`/`Schema` agree across a chain.

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
   **Correction after execution:** the plan hedged and asserted on the exception *chain* rather than
   `Throws.TypeOf<ObjectDisposedException>()`, on the reasoning that the disposed object is the source
   and `CollectAsync`'s catch-all would wrap it in `QueryExecutionException`. That reasoning was
   wrong: the frame's own guard fires *before* the try block (`:379`), so the survivor throws a bare
   `ObjectDisposedException` naming `Nivara.Query.QueryFrame`. The test now asserts
   `Throws.TypeOf<ObjectDisposedException>()` directly — a stronger, less accidental pin than the
   plan could have specified.

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
- **`src/Nivara.Extensions/Streamix/NivaraFlux.cs`** — `ToFlux`, `ToFluxWithTimestamp` (x2),
  `EnumerateRows`, `ToFluxRows` all accept a `QueryFrame` and do **not** own it. Signatures untouched;
  listed because they are the #502 surface and should be re-checked at G2.
- **Every `IQuerySource` implementor** (verified via code-memory, 6 in `src/`: `CsvLazySource`,
  `CsvEagerSource`, `JsonLazySource`, `JsonEagerSource`, `ParquetLazySource`, `MemoryQuerySource`;
  13 in `tests/`) keeps its existing contract. `QuerySourceHandle` holds the interface and never
  re-implements it.

## Grounding (G1)

Checked against Microsoft Learn and code-memory.

**`IAsyncDisposable` is essentially unused in this repository.** ripgrep finds it in exactly two
places, both in `QueryFrame.cs`: the class declaration (`:15`) and `if (source is IAsyncDisposable
asyncDisposable)` (`:1137`). No `IQuerySource` implementor — in `src/` or `tests/` — implements it, so
**that branch at `:1137` is currently unreachable**, and `ReleaseAsync`'s async-awareness is
forward-looking only. Stated so the plan does not overclaim an async release path that nothing
exercises today.

*(Method note: code-memory's `Implements` index reported zero `IAsyncDisposable` implementors, which
is wrong — `QueryFrame` is one. The index does not resolve interface lists, so this fact was taken
from ripgrep instead.)*

**Implementing both interfaces is correct and required, not stylistic.** Microsoft Learn is explicit
that a type using `IAsyncDisposable` should normally also implement `IDisposable`, because a consumer
who calls `Dispose` would never reach `DisposeAsync` and would leak. It is also required for
compile-time use: CS8417 (`await using` on `IDisposable`-only) and CS8418 (`using` on
`IAsyncDisposable`-only) are both errors. Implementing only `IDisposable` would make `await using` on
`NivaraQuery<T>` a compile error, so `ScanQuery<T>` consumers could not use the async form at all.

**Idempotent release is the documented requirement.** The dispose-pattern guidelines say "DO allow
`Dispose` to be called more than once — the method might choose to do nothing after the first call,"
`IDisposable.Dispose` says a repeated call "must ignore all calls after the first one" and "must not
throw," and they also say "X AVOID throwing an exception from within `Dispose(bool)`". `Release()`
being idempotent and non-throwing matches all three.

**Throwing after dispose is the documented requirement, which is what validates the behavior change
above.** The guidelines say "✓ DO throw an `ObjectDisposedException` from any member that cannot be
used after the object has been disposed of." The old code violated this for siblings — a disposed
frame reported `disposed == false` to everyone else in the chain. Delegating guards to
`handle.Released` brings `QueryFrame` into line.

**Sealed types need no `Dispose(bool)` / `DisposeAsyncCore()`.** CA1063 applies to unsealed types, and
the async docs state that when an `IAsyncDisposable` implementation is sealed, `DisposeAsyncCore` is
unnecessary. Both target types are `sealed` (confirmed via code-memory `IsSealed`), so implementing
`Dispose()` and `DisposeAsync()` directly is the correct shape and adds no analyzer burden.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` — 0 warnings, 0 errors.
2. **RED proof** of the regression test against the pre-fix `QueryFrame.cs` (restored from
   `c2a0363`, with `QuerySourceHandle.cs` removed and the fixture reduced to its four frame-level
   tests, since the handle cannot exist there). A regression test never seen red proves nothing.
3. `ScanAsQueryFrameHandleTests` + `NivaraQueryDisposalTests` + `QuerySourceHandleTests` +
   `QueryFrameTests` + `ResourceManagementPropertyTests` + `AsyncStreamingTests`, `-c Release`.
   Negative control (`Probe_DetectsDeliberatelyLeakedHandle`) must stay green — it is what gives the
   gate teeth on `ubuntu-latest`.
4. Full suite `-c Release --filter "Category!=Performance"`, capturing the **process** exit status,
   not a pipeline filter's. This is what confirms the `handle.Released` behavior change is safe.
5. Re-run 3 to confirm the negative control is stable, not flaky.

## Corrections after execution

Four things the plan got wrong or understated. Each was found by running the code, not by reading it.

**The label rename touched two assertions, not one.** The plan named
`ResourceManagementPropertyTests.cs:178` as the only site. There is a second,
`AbandonedQueryCleanup_ShouldAutomaticallyCleanupResources` at `:318`, in a different test. It
surfaced only when the targeted run failed on it. Both are updated, and each commit that caused a
break carries the fix for it, so no commit is left red.

**The `QueryExecutionException` hedge was wrong, and the reality is stronger.** Test 6 was specified
to assert on the exception *chain* rather than `Throws.TypeOf<ObjectDisposedException>()`, reasoning
that the released source would fail deeper and get wrapped. It does not: the frame's own guard fires
*before* `CollectAsync`'s try block, so a survivor throws a bare `ObjectDisposedException` naming
`Nivara.Query.QueryFrame`. The test asserts the precise type instead. Worth recording because the
plan's version would have been a weaker gate written from a plausible but unverified story.

**The handle needs no `Schema` forwarder.** The plan specified `Source` / `Schema` / `IsLazy`
forwarders. `QueryFrame` only ever reads `source.IsLazy`; `IQuerySource.Schema` has no reader in the
file, so a forwarder would be dead code.

**`IAsyncDisposable` is dead code today, not just unused.** ripgrep finds it in exactly two places
repo-wide, both in `QueryFrame.cs`, and **no** `IQuerySource` implementor implements it. So the
`source is IAsyncDisposable` branch — which this change moves into `ReleaseAsync` — is unreachable and
has never executed. It is in the right place for the future, but nothing exercises it yet.

## Verification results

| Step | Result |
| --- | --- |
| 1. Build `-c Release` | 0 warnings, 0 errors. Adding `IDisposable` broke no overload resolution anywhere in the solution (samples, Extensions, tests). |
| 2. RED proof | **4/4 fail** against pre-fix `QueryFrame.cs`, each for the predicted reason. `DerivedFrame_Abandoned_...`: `DisposeCount` 1 vs expected 0 *and* the surviving leaf's `Collect()` threw `ObjectDisposedException` — the live bug. `DerivedFrame_Added_...`: `TotalTrackedResources` 3 vs 1. `..._ReportsSameDisposedState`: sibling's `ToString()` reported a live pipeline. `..._ThrowsObjectDisposedException`: got `QueryExecutionException` wrapping `ObjectDisposedException` — the exact inconsistency the handle removes. |
| 3. Targeted fixtures | 102 pass, exit 0, after the second label fix. `AbandonedQueryCleanup_ShouldAutomaticallyCleanupResources` passing is independent confirmation that the cleanup closure does not capture the handle. |
| 3b. `NivaraQueryDisposalTests` | 8/8 pass. Two authoring bugs found and fixed first: the in-memory frame's column was `A` against a `Person` mapping `Age` (`SchemaValidationException`), and `GroupBy(row => "all")` is a constant key the operation does not accept. |
| 3c. `ScanAsQueryFrameHandleTests` | 7/7 pass, negative control included. No `AsQueryFrame().Dispose()` remains. |
| 4. Full suite | **NOT RUN** — declined by the human at G2; they will run it. Unverified. |
| 5. Re-run | **NOT RUN** — same. |

> **Coverage gap, stated plainly.** Steps 1–3c cover the four fixtures that touch `QueryFrame`
> lifetime and the source manager. The full suite has not been run on this branch, so the
> `handle.Released` guard change is **not** verified against the rest of the test surface — any test
> elsewhere in the repository that disposed one frame and kept reading from a sibling would fail,
> and none was found by reading, which is exactly what the step-2 RED proof showed reading cannot
> guarantee. Treat steps 4–5 as a required gate before this branch merges, not as a formality.
>
> To run them:
>
> ```
> dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "Category!=Performance"
> dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ScanAsQueryFrameHandleTests"
> ```
>
> Capture the **process** exit status, not the status of a filter in a pipeline. Both must be 0.

## Commits (as landed)

1. `docs: plan #501 in TODO.md`
2. `docs: record G1 grounding findings for #501 in TODO.md`
3. `refactor: track query-source lifetime on a shared handle instead of per-frame`
4. `test: gate the shared-source release contract and the abandoned-frame regression`
5. `feat: implement IDisposable and IAsyncDisposable on the typed LINQ query types`
6. `test: repoint the ScanQuery handle gate at query.Dispose()`
7. `docs: update ScanQuery disposal guidance in XML docs, LINQ.md and CHANGELOG`

> The plan proposed 6, and it became 7. The G1 findings needed their own commit so the refactor
> could cite them, and the feature and its tests landed together rather than apart: a commit that
> adds a public `IDisposable` contract with nothing pinning it is not verified on its own.

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

### Filed during this work

- [x] **#507** — `NivaraQuery<T>.Frame` is dead internal code, and now an ambiguous disposal path.
  Confirmed by ripgrep for `\.Frame\b`: zero readers anywhere. Left in place here to keep the branch
  to one reason, but it is now a hazard — a property named `Frame` on a type that owns the frame's
  lifetime invites `query.Frame.Dispose()`, which releases the source for the whole chain while
  carrying none of the documentation.
- [x] **#508** — `QueryFrame.DisposeAsync` propagates source disposal errors while `Dispose` swallows
  them. Preserved deliberately here so the refactor stayed a pure move; it is currently unobservable
  because no `IQuerySource` implements `IAsyncDisposable`.