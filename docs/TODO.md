# Plan: Issue #1168 — NivaraResourceManager dead, slow and inverted

## Problem

`NivaraResourceManager` (`src/Nivara/Helpers/NivaraResourceManager.cs`) is `static class` (internal)
but every member is `public`, so the abandoned-resource cleanup it implements is unreachable
from a consumer and dead in the shipped package. Inside: `UntrackResource` is an O(n) scan of a
`ConcurrentDictionary<WeakReference, ResourceInfo>` under a global lock on every frame/column
dispose (O(n²) across n disposals); `CleanupAbandonedResources` runs cleanup actions inside
`lock (_lock)` while the comment above claims they run outside; `Disable()` drops tracked
resources without running their cleanup; `Enable()` starts a 30 s timer; `ResourceStatistics`
is public but only reachable via an internal type.

## Proposed changes

### Commit 1 — split memory advice out of NivaraResourceManager (pure move)

- `src/Nivara/Helpers/NivaraResourceManager.cs` → `src/Nivara/Helpers/MemoryAdvisor.cs`,
  `internal static class MemoryAdvisor`, holding only `GetMemoryRecommendations` (`public` →
  `internal`).
- Move `MemoryRecommendations` and `MemoryWarningLevel` into `src/Nivara/Helpers/MemoryRecommendations.cs`
  (unchanged, still `public`).
- Fix `MemoryRecommendations.AvailableMemory`'s `<see cref="NivaraResourceManager.GetMemoryRecommendations(long, string, long?)"/>`
  to point at `NivaraFrame.GetMemoryRecommendations(string, long?)` (a public type must not
  reference an internal one).
- `NivaraFrame.cs:1259` call site → `MemoryAdvisor.GetMemoryRecommendations(...)`.

No behaviour change; no doc-gate impact (no doc carries a `File.cs::Symbol` citation to this file).

### Commit 2 — delete the abandoned-resource tracking subsystem

Production:

- Delete `_trackedResources`, `_lock`, `_isEnabled`, `_cleanupTimer`, `Enable`, `Disable`,
  `ForceCleanup`, `GetResourceStatistics`, `TrackResource`, `UntrackResource`,
  `CleanupAbandonedResources`, `ResourceInfo`, and the public `ResourceStatistics` type from
  the helper.
- `src/Nivara/NivaraColumn.cs`: drop the ctor tracking block (`:465-468`) and the
  `UntrackResource` call in `Dispose` (`:3259`); then `estimateMemoryUsage()` (`:1568`) and
  `getTypeSize(Type)` (`:269`) have no callers left — delete both (verified no other use in
  the file; both are private/static with no external caller).
- `src/Nivara/NivaraFrame.cs`: drop the ctor tracking block (`:281-285`), the
  `long estimatedMemoryUsage = 0;` local and the unconditional
  `estimatedMemoryUsage += estimateColumnMemoryUsage(column);` (`:235`, `:269`), and the
  `UntrackResource` call in `Dispose` (`:1275`). Keep `estimateColumnMemoryUsage` /
  `estimateFrameMemoryUsage` for `GetMemoryRecommendations`. No behaviour change for
  disposed columns (length validation already touches `column.Length` first).
- `src/Nivara/Query/QuerySourceHandle.cs`: drop the whole `if (source.IsLazy && NivaraResourceManager.IsEnabled)`
  block incl. closure (`:31-48`), the two `UntrackResource` calls in `Release`/`ReleaseAsync`,
  and reword the class remarks + the three "mirroring the abandoned-resource cleanup path"
  comments to describe only the explicit path. `Interlocked.Exchange` idempotence unchanged.
- `src/Nivara/Query/QueryFrame.cs:1113`: drop the "and the abandoned-resource cleanup path"
  clause from the `Dispose` remark.

Tests (same commit, per the change→test coupling rule):

- `tests/Nivara.Tests/ResourceManagementPropertyTests.cs`: delete fixture setup
  (`OneTimeSetUp`/`OneTimeTearDown`/`Setup`/`TearDown`, all tracking-only) and the three
  mechanism properties `ResourceTrackingDisabledByDefault_ShouldNotTrackResources`,
  `AbandonedQueryCleanup_ShouldAutomaticallyCleanupResources` (plus `CreateAndAbandonQueries`),
  `ResourceTracking_ShouldProvideAccurateStatistics`. Keep `IDisposablePattern_ShouldBeProperlyImplemented`,
  `ColumnDisposablePattern_ShouldBeProperlyImplemented`,
  `MemoryManagement_ShouldProvideGuidanceForLargeDatasets`. Strip the `statsBefore`/tracked/abandoned
  assertions from `ResourceDisposal_ShouldCleanupAllUnderlyingResources` and
  `QueryFrameDisposal_ShouldCleanupLazyQueryResources`, keeping the disposed-accessor assertions;
  in the latter, assert the temp file can be `File.Delete`d after dispose (the user-visible
  guarantee, stronger than the tracking assertion it replaces).
- `tests/Nivara.Tests/Query/QuerySourceHandleTests.cs`: remove `SetUp`/`TearDown` and the
  `Nivara.Helpers` using; delete `Handle_Abandoned_ReleasesSourceWhenNothingReferencesIt` and
  `DerivedFrame_Added_RegistersNoAdditionalTrackedResource`; rewrite
  `DerivedFrame_Abandoned_DoesNotReleaseSourceWhileSiblingLives` without GC/`ForceCleanup`
  (drop the intermediate, assert the leaf still collects and the source was never disposed).
  Delete helpers `CreateAbandonedChain` and `CollectAbandoned`; update the fixture summary comment.

Coverage statement: the deleted tests exercised a mechanism no consumer could activate. Every
guarantee they encoded that a user can observe is retained — disposed frames throw, a shared
source survives sibling disposal, an abandoned intermediate never releases a live sibling's
source, an explicit dispose releases the file handle. The one guarantee with no remaining
mechanism (abandoned handle release) moves to follow-up issue #1189 rather than being silently
dropped.

### Commit 3 — record the decision in docs

- `CHANGELOG.md`: new `## [Unreleased]` section at the top:
  - `### Breaking changes (migration)`: `ResourceStatistics` (removed) → no replacement; no
    shipped code path produced one. Delete any reference.
  - `### Removed`: internal `NivaraResourceManager` tracking (Enable/Disable/ForceCleanup/
    GetResourceStatistics, the 30 s timer, the tracking dictionary) — internal and unreachable;
    explicit `Dispose` is the only release path for frames, columns and query sources.
  - Note: `MemoryRecommendations` / `MemoryWarningLevel` are unchanged and still returned by
    `NivaraFrame.GetMemoryRecommendations`.
- `docs/REVIEW-2026-11-12.md`: per §8, record the closure reason against the M5 row in §2;
  update the §5 theme-3 sentence ("The abandoned-resource manager is unreachable and cannot
  arbitrate (M5)") and the §7 item-6 phrasing so the document stays truthful. No new citations
  → no pin arithmetic changes.
- `docs/STREAMING.md` §Resource management: one sentence making the contract explicit — there
  is no process-global cleanup facility; an undisposed scan is released only when its underlying
  stream is finalized, so "dispose the scan you created" is the whole contract.
- `ARCHITECTURE.md`: no removal needed (verified: no tracking/abandoned references); the
  ownership bullets #1159 added stay as-is.
- `AGENTS.md`: no entry for this type, so no edit.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ResourceManagement|FullyQualifiedName~QuerySourceHandle|FullyQualifiedName~NivaraQueryDisposal"`
- Docs namespace (because docs + CHANGELOG change):
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
  (expects zero pin changes; if a pin moves, justify it in the pin's arithmetic rather than bumping)
- `grep -rn "NivaraResourceManager\|ResourceStatistics"` must return nothing in src/tests outside comments.

## Planned commits

1. docs: plan #1168 in TODO.md
2. refactor(helpers): split memory advice out of NivaraResourceManager
3. refactor(resources): delete the abandoned-resource tracking subsystem
4. docs: record the resource-manager removal (#1168)

## Blast radius

- `src/Nivara/Helpers/NivaraResourceManager.cs` (deleted → split into `MemoryAdvisor.cs` +
  `MemoryRecommendations.cs`)
- `src/Nivara/NivaraColumn.cs` (ctor, Dispose, dead `estimateMemoryUsage`/`getTypeSize`)
- `src/Nivara/NivaraFrame.cs` (ctor, Dispose, call site)
- `src/Nivara/Query/QuerySourceHandle.cs` (ctor, Release, ReleaseAsync, remarks)
- `src/Nivara/Query/QueryFrame.cs` (Dispose remark)
- `tests/Nivara.Tests/ResourceManagementPropertyTests.cs`, `tests/Nivara.Tests/Query/QuerySourceHandleTests.cs`
- `CHANGELOG.md`, `docs/REVIEW-2026-11-12.md`, `docs/STREAMING.md`

Downstream callers: `NivaraFrame.GetMemoryRecommendations` (public) delegates to the helper —
behaviour unchanged. No other production caller of `NivaraResourceManager` exists (verified by
grep: only NivaraColumn, NivaraFrame, QuerySourceHandle, and the two test fixtures).

## GitHub issues log

- [x] #1189 — lazy query sources hold their reader with no finalizer/SafeHandle; a dropped,
  partially-read QueryFrame leaks the handle until GC (created while planning #1168)
- #1168 — this issue (in progress)