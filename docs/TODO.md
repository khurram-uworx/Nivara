# Plan: #325 Phase 1 — byte-aware backpressure primitive + enforced mode

Branch: `khurram/325` (off `main` at 3749d474).

## Problem

`NivaraExecutionContext.MemoryBudget` exists and reaches `StreamingExecutionStrategy`, but it only sizes chunk/channel capacity and feeds an advisory `StreamingBudgetTracker` that warns at 2× budget. There is no hard in-flight ceiling and no byte-aware resource governor. Issue #325 asks for spill-to-disk; the issue comment argues the correct first step is backpressure at the source (byte-aware, before the read), with spill reserved for boundary operators that genuinely need the whole dataset. Phase 2 (spill/external memory) is tracked separately as #743.

## Proposed changes

### 1. `BudgetEnforcement` enum
New file `src/Nivara/Execution/BudgetEnforcement.cs`:
```csharp
namespace Nivara.Execution;
public enum BudgetEnforcement { Advisory, Enforced }
```

### 2. `MemoryBudget` primitive
New file `src/Nivara/Execution/MemoryBudget.cs`: public sealed `IDisposable` governor with
`AcquireAsync(long requestedBytes, CancellationToken ct = default)`, `Release(long)`,
`CurrentUsage`, `Capacity`. Weighted (byte) capacity; FIFO wait queue; `ValueTask.CompletedTask`
on the happy path; throws when a single request exceeds capacity; cancellation-safe.

### 3. Context integration
`src/Nivara/Execution/NivaraExecutionContext.cs`: add
`BudgetEnforcement BudgetEnforcement { get; set; }` (default `Advisory`); include in `Clone()`
and `ToString()`; add `WithBudgetEnforcement(...)` factory.

### 4. Strategy integration
`src/Nivara/Execution/StreamingExecutionStrategy.cs`:
- Materializing async path: producer acquires budget before `WriteAsync` (release on failure);
  consumer releases accumulated frame bytes after concatenation/disposal.
- `StreamChunksAsync`: pure-streamable path wraps each yielded frame in acquire/release via
  `try/finally` around `yield return`; boundary-accumulation paths acquire per frame and
  release after concat/disposal.
- Keep `StreamingBudgetTracker` in both modes for diagnostics.

### 5. Tests
- New `tests/Nivara.Tests/Execution/MemoryBudgetTests.cs`.
- Extend `ExecutionContextTests` and `StreamingExecutionStrategyTests`.

### 6. Docs
Update `docs/STREAMING.md` §AC3 and `docs/STREAMING-PROPOSAL.md` status to reflect Phase 1 shipped.

## Verification steps
- `dotnet build Nivara.slnx`
- Targeted: `MemoryBudgetTests`, `ExecutionContextTests`, `StreamingExecutionStrategyTests`, `StreamingBackpressureTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. `docs: plan #325 phase 1 in TODO.md`
2. `feat(execution): add BudgetEnforcement enum and MemoryBudget primitive`
3. `feat(execution): wire BudgetEnforcement into NivaraExecutionContext`
4. `feat(execution): enforce MemoryBudget in StreamingExecutionStrategy (async paths)`
5. `test(execution): MemoryBudget and enforced-mode coverage`
6. `docs: mark #325 phase 1 shipped`
7. `docs: remove TODO.md — plan executed`

## Blast radius
- New public types in `Nivara.Execution`: `BudgetEnforcement`, `MemoryBudget`.
- `NivaraExecutionContext`: new property (additive, default preserves behavior).
- `StreamingExecutionStrategy`: async paths only; sync `ExecuteCore` remains advisory in Phase 1.
- No changes to source readers, channel capacity formula, or `AsStream` public signatures.

## GitHub issues log
- [x] #743 — Phase 2: external-memory / spill-to-disk for boundary operators (#325) (created while planning Phase 1)
EOF