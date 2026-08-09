# Plan: Issue #174 — make `NivaraResourceManager` opt-in, performance-first by default

## Problem

Every `NivaraColumn<T>` construction (and frames/series transitively) registers itself in a
global `NivaraResourceManager` — a `ConcurrentDictionary<WeakReference, ResourceInfo>` — and the
process runs a permanent 30-second background `Timer`. This is hidden per-allocation overhead
(`WeakReference` + boxed `ResourceInfo` + concurrent-dict insert) and global state with no opt-out,
on the hot path of a performance-oriented library. See `docs/REVIEW.md` finding #3.

## Direction (confirmed)

- **Static opt-in switch** on `NivaraResourceManager` (`IsEnabled` / `Enable()` / `Disable()`),
  default **OFF**.
- **Timer created lazily** only on `Enable()`; a default host never starts a timer thread.
- **Column tracking preserved** when enabled — opt-in telemetry identical to today.
- Public surface unchanged: `MemoryRecommendations`, `MemoryWarningLevel`, `ResourceStatistics`,
  `NivaraFrame.GetMemoryRecommendations` all stay.
- Tests enable tracking in the resource-management fixture; a new test asserts default-off
  (no tracking, no dictionary growth).

## Blast radius

| Change | Files | Downstream / tests |
|---|---|---|
| Manager gate + lazy timer | `src/Nivara/Helpers/NivaraResourceManager.cs` | Only internal consumers: column/frame/QueryFrame ctors + Disposes; `ResourceManagementPropertyTests` |
| Gate column ctor tracking | `src/Nivara/NivaraColumn.cs` (`:606-613`, `:2690`) | All column creation funnels through the single internal ctor (factory methods + arithmetic ops). No behavior change when disabled; identical when enabled |
| Gate frame ctor tracking | `src/Nivara/NivaraFrame.cs` (`:280`, `:1236`) | `NivaraFrame` single public ctor; `GetMemoryRecommendations` unaffected |
| Gate QueryFrame tracking | `src/Nivara/Query/QueryFrame.cs` (`:32`, `:61`, `:804`) | Lazy-source abandoned cleanup (`CleanupAction`) becomes opt-in — the one behavioral case |
| Test fixture enable + default-off test | `tests/Nivara.Tests/ResourceManagementPropertyTests.cs` | 6 existing tracking tests must stay green |
| Docs | `docs/REVIEW.md`, `CHANGELOG.md` | — |

No samples, performance tests, or other consumers reference these APIs (verified via grep).

## Behavioral note

The only real behavior carried by tracking — `QueryFrame` disposing abandoned lazy sources via its
`CleanupAction` — becomes opt-in. Everything else is observational telemetry.

## Changes

1. **`NivaraResourceManager.cs`**
   - Remove static-ctor timer start.
   - `private static volatile bool _isEnabled;` + `private static Timer? _cleanupTimer;`
   - `bool IsEnabled`, `void Enable()`, `void Disable()`.
     - `Enable()`: under `_lock`, set flag; lazily create + start 30s timer (idempotent).
     - `Disable()`: under `_lock`, clear flag; dispose/null timer; clear `_trackedResources`.
   - `TrackResource` / `UntrackResource` / `CleanupAbandonedResources`: `if (!_isEnabled) return;`
   - `ForceCleanup` / `GetResourceStatistics`: stay functional (empty dict when disabled).

2. **`NivaraColumn.cs`** — internal ctor: `if (NivaraResourceManager.IsEnabled) { TrackResource(..., estimateMemoryUsage()); }`.
   `Dispose` keeps `UntrackResource(this)` (internally guarded no-op).

3. **`NivaraFrame.cs`** — public ctor: gate `TrackResource(this, "NivaraFrame", estimatedMemoryUsage)`
   behind `IsEnabled`. `Dispose` keeps `UntrackResource`. `GetMemoryRecommendations` unchanged.

4. **`QueryFrame.cs`** — both ctors: gate `TrackResource` behind `IsEnabled`; XML-doc note that
   abandoned-lazy-source cleanup is now opt-in.

5. **`ResourceManagementPropertyTests.cs`** — `[OneTimeSetUp]` → `Enable()`,
   `[OneTimeTearDown]` → `Disable()`. New test: default-off — with tracking disabled,
   creating columns/frames leaves `GetResourceStatistics().TotalTrackedResources == 0`.

6. **Docs** — `REVIEW.md`: mark finding #3 resolved + strike item 17.
   `CHANGELOG.md`: `[Unreleased]` entry.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` on `Nivara.Tests` (resource tests) — **requires human confirmation**
  (AGENTS.md rule).
- Default-off test proves no dictionary growth.

## Planned commits

1. `docs: plan #174 opt-in resource tracking in TODO.md`
2. `refactor: make NivaraResourceManager opt-in with lazy timer (core)`
3. `test: enable tracking in resource fixture; add default-off test`
4. `docs: resolve REVIEW finding #3 and add CHANGELOG entry`
5. `docs: remove TODO.md — #174 plan executed`

## GitHub issues log

- [ ] (none yet — create issues as deferred work is discovered during execution)
