# Plan: Fix MemoryBudget deadlock on cancellation + over-release clamping (issue #1161)

## Problem
`MemoryBudget.Release` disposes a waiter's `CancellationTokenRegistration` while holding `gate`. The registration callback takes `lock (budget.gate)`. If cancellation races a release, `Registration.Dispose()` blocks waiting for a callback blocked on the lock the releasing thread holds → hard deadlock. `Dispose()` has the same defect. Separately, `current = Math.Max(0, current - releasedBytes)` silently absorbs double-release/mismatched-byte bugs, corrupting the cap.

## Proposed changes
- `src/Nivara/Execution/MemoryBudget.cs`:
  - Collect registrations to dispose into a local list inside `lock (gate)`, dispose them after releasing the lock (in both `Release` and `Dispose`).
  - Replace `Math.Max` clamp with explicit over-release check: throw `InvalidOperationException` when `releasedBytes > current`.
- No call-site changes expected; `StreamingExecutionStrategy` acquires/releases matching bytes. Verify via tests.

## Verification steps
- Build: `dotnet build Nivara.slnx`
- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~MemoryBudget"` (ask before running)
- Streaming tests if time permits.
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan #1161 in TODO.md
2. fix: MemoryBudget deferred registration disposal + over-release throw
3. test: MemoryBudget regression/stress tests (if needed)
4. docs: remove TODO.md — plan executed

## Blast radius
- `MemoryBudget` used in enforced streaming mode (`StreamingExecutionStrategy.executeCoreInternalAsync`, `StreamChunksAsync`).
- Public-ish internal type; tests in `MemoryBudgetTests`.

## GitHub issues log
- (none yet)
