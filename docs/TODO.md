# Plan: Fix #1162 — `GetMemoryRecommendations` reports managed heap as available memory (KB thresholds)

## Problem

`NivaraFrame.GetMemoryRecommendations()` (public) delegates to `NivaraResourceManager.GetMemoryRecommendations()`,
which sets `AvailableMemory = GC.GetTotalMemory(false)` (the managed heap size, not available memory) and uses
kilobyte thresholds (`5 KB`, `20 KB`, `100 KB`), so every realistic frame is reported "large"/"very large".
`estimateFrameMemoryUsage`/`estimateColumnMemoryUsage` already include the null mask but are undocumented as
approximations.

## Proposed changes

1. `src/Nivara/Helpers/NivaraResourceManager.cs`
   - Signature: `GetMemoryRecommendations(long estimatedDataSize, string operationType, long? memoryBudgetBytes = null)`.
   - `availableMemory` = supplied budget (if > 0) else `GC.GetGCMemoryInfo().TotalAvailableMemoryBytes` (fallback 1 GB).
   - Thresholds scaled to megabytes: 5 MB / 20 MB / 100 MB.
   - Update `MemoryRecommendations.AvailableMemory` doc + add `<remarks>` cross-referencing `MemoryBudget`.
2. `src/Nivara/NivaraFrame.cs`
   - `GetMemoryRecommendations(string operationType = "general", long? memoryBudgetBytes = null)` forwards the budget.
   - Document `estimateFrameMemoryUsage`/`estimateColumnMemoryUsage` as lower-bound approximations.
3. `docs/STREAMING.md` — prose subsection: advisory heuristics vs. `MemoryBudget` governor.
4. `tests/Nivara.Tests/ResourceManagementPropertyTests.cs` — adjust assertions for MB thresholds.

## Verification

- `dotnet build Nivara.slnx`
- Targeted: `ResourceManagementPropertyTests`
- Docs snippet gate (prose-only change; run for safety)
- Manual: 40 KB frame → `WarningLevel == None`, `AvailableMemory != GC.GetTotalMemory(false)`.

## Planned commits

1. `docs: plan #1162 fix in TODO.md`
2. `fix: report real available memory and MB-scale thresholds in GetMemoryRecommendations`
3. `docs: document estimateFrameMemoryUsage as lower-bound approximation`
4. `docs: relate GetMemoryRecommendations to MemoryBudget in STREAMING.md`
5. `test: adjust ResourceManagementPropertyTests for MB thresholds`
6. `docs: remove TODO.md — plan executed`

## Blast radius

- `NivaraResourceManager.GetMemoryRecommendations` (static, public) — signature gains optional param (source-compatible).
- `NivaraFrame.GetMemoryRecommendations` (public) — signature gains optional param (source-compatible).
- `MemoryRecommendations.AvailableMemory` (public property) — value semantics change (managed heap → available memory). Binary/source compatible.
- `estimateColumnMemoryUsage`/`estimateFrameMemoryUsage` (internal) — unchanged behavior, docs only.
- Downstream: `StreamingExecutionStrategy` uses `estimateFrameMemoryUsage` (unchanged).
- Tests: `ResourceManagementPropertyTests` only consumer of `GetMemoryRecommendations`.

## GitHub issues log

- (none yet)