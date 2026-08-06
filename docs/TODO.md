# Plan: Restore optimizer momentum buffers on LoadStateDict (fresh optimizer)

Status: proposed (branch `khurram/optimizer-loadstate-fix` off main @ f9214b8)

## Problem

`Adam<T>`, `AdamW<T>`, and `SGD<T>` build their per-parameter momentum buffers
(`expAvg` / `expAvgSq` / `velocity`) lazily on the first `Step()`. Their
`LoadStateDict` implementations only copy state into **already-created**
buffers:

- `src/Nivara/AutoDiff/Optimizer/Adam.cs` — `LoadStateDict` loops
  `for (i < expAvgBuffers.Count)`.
- `src/Nivara/AutoDiff/Optimizer/AdamW.cs` — same pattern.
- `src/Nivara/AutoDiff/Optimizer/SGD.cs` — same pattern over `velocityBuffers`.

On a **fresh** optimizer (no prior `Step()`), the buffer lists are empty, so a
`LoadStateDict` silently drops `expAvg`/`expAvgSq`/`velocity` while still
restoring `step`. Adam then applies near-1 bias correction over zeroed moments —
a latent loss-explosion bug when gradients are large.

This is exactly the path our merged retraining flow (PR #126) documents:
`TrainingLoop.LoadCheckpoint` → `_optimizer.LoadStateDict` on a fresh optimizer
(NivaraChat `TrainIncremental`, `docs/RETRAINING.md` example). The bug is real
on main; our end-to-end verification looked fine only because the fine-tune was
near convergence (small gradients → Adam-from-zero is well-behaved).

Sourced from open PR #125 (`khurram/retraining`), which contains this core fix
plus overlapping NivaraChat/docs work already merged via #126, plus
SHAKESPEARE/ADR-003 noise to ignore.

## Changes

### 1. `src/Nivara/AutoDiff/Optimizer/Adam.cs` — `LoadStateDict`

Replace the existing-buffer loop with a state-driven loop that allocates buffers
from the state keys:

```csharp
int i = 0;
while (state.TryGetValue($"expAvg_{i}", out var buf))
{
    ensureBuffer(i, buf.Length);
    buf.AsSpan(0, Math.Min(buf.Length, expAvgBuffers[i].Length)).CopyTo(expAvgBuffers[i]);
    if (state.TryGetValue($"expAvgSq_{i}", out var sqBuf))
        sqBuf.AsSpan(0, Math.Min(sqBuf.Length, expAvgSqBuffers[i].Length)).CopyTo(expAvgSqBuffers[i]);
    i++;
}
```

(`ensureBuffer` already exists — lazy create via `ArrayPool<T>.Shared`.)

### 2. `src/Nivara/AutoDiff/Optimizer/AdamW.cs` — `LoadStateDict`

Same shape as Adam, using its `EnsureBuffer(idx, size)` helper.

### 3. `src/Nivara/AutoDiff/Optimizer/SGD.cs` — `LoadStateDict`

```csharp
int i = 0;
while (state.TryGetValue($"velocity_{i}", out var buf))
{
    ensureVelocityBuffer(i, buf.Length);
    buf.AsSpan(0, Math.Min(buf.Length, velocityBuffers[i].Length)).CopyTo(velocityBuffers[i]);
    i++;
}
```

### 4. `tests/Nivara.Tests/AutoDiff/OptimizerTests.cs` — regression tests

Port the three fresh-optimizer round-trip tests from PR #125:

- `Adam_StateDict_LoadStateDict_FreshOptimizer_RestoresState`
- `AdamW_StateDict_LoadStateDict_FreshOptimizer_RestoresState`
- `Sgd_StateDict_LoadStateDict_FreshOptimizer_RestoresMomentum`

Each: train 3 steps → `StateDict()` → fresh optimizer + fresh param →
`LoadStateDict` → `StateDict()` → assert same keys, `step`, and per-buffer
momentum values within 1e-6.

## Verification

- `dotnet build src/Nivara/Nivara.csproj` — clean
- `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj` — clean
- Targeted NUnit run: the 3 new tests + existing optimizer StateDict/LoadStateDict
  tests (ask before running `dotnet test`)

## Commits (iterative)

1. `docs: plan optimizer LoadStateDict fresh-buffer fix in TODO.md`
2. `fix: restore momentum buffers on LoadStateDict for fresh Adam/AdamW/SGD`
   (code + tests in one logical change)

## Follow-up

- After merge into main: close PR #125 (its only non-overlapping value is the
  optimizer fix absorbed here).
- Optional hardening (not in scope): add a loss-decreases assertion on the resume
  path so buffer restore is guarded end-to-end.
