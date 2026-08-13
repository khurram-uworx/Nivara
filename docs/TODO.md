# Plan: Loss<T>.Reduce Forward overload cleanup (issue #215)

## Problem

Issue #215 (`code-quality`, low): `Loss<T>.Reduce` is reached from two
`Forward` overloads with divergent `Reduce` call paths.

Investigation found the issue's core suggestion is **already implemented** in
current HEAD:

- `src/Nivara/AutoDiff/Nn/Functional/Loss.cs:18-19` — the base 2-arg
  `Forward(predictions, targets)` already delegates to the 3-arg overload:
  `=> Forward(predictions, targets, Reduction)`.
- All 5 subclasses (`MSELoss`, `L1Loss`, `BCELoss`, `BCEWithLogitsLoss`,
  `CrossEntropyLoss` — the only `Loss<T>` subclasses in the repo) override only
  the 3-arg abstract `Forward`. None override the 2-arg. The
  `CrossEntropyLoss.Forward(logits, int[] targets)` overload
  (`CrossEntropyLoss.cs:31`) is a distinct convenience overload (not an
  override) and routes through the base 2-arg → 3-arg chain.

Remaining "divergence" is intentional and correct:

- `CrossEntropyLoss.Forward` passes `batchSize` as the `Reduce` divisor
  (`CrossEntropyLoss.cs:27-28`) — PyTorch batch-averaged CE.
- Other losses rely on `divisor ?? elementwiseLoss.Length` (`Loss.cs:42`) —
  element-average, matches PyTorch MSE/L1/BCE.

## Proposed changes (no behavior change)

1. **Regression tests** in `tests/Nivara.Tests/AutoDiff/LossTests.cs`:
   - `LossBase_AllLosses_TwoArgForward_DelegatesToThreeArg` — all 5 losses ×
     Sum/Mean/None: `Forward(p,t)` equals `Forward(p,t, storedReduction)`.
   - `CrossEntropyLoss_Mean_UsesBatchSizeDivisor` — `[2,3]` logits, Mean →
     `sum/2`, not `sum/6`; locks in batch-average semantics.
   - `CrossEntropyLoss_IntArrayTargets_MatchesOneHotTensorPath` — `int[]` label
     overload equals one-hot tensor path.
2. **Doc comments** (non-obvious design decisions only):
   - `Loss.cs:21` — subclasses override only the 3-arg `Forward`; the 2-arg is
     non-virtual and delegates.
   - `CrossEntropyLoss.cs:27` — `batchSize` divisor matches PyTorch batch-averaged
     CE, deliberately differing from the element-count fallback.

## Blast radius

- `src/Nivara/AutoDiff/Nn/Functional/Loss.cs` — comment only; API unchanged.
- `src/Nivara/AutoDiff/Nn/Functional/CrossEntropyLoss.cs` — comment only; API
  unchanged.
- `tests/Nivara.Tests/AutoDiff/LossTests.cs` — 3 new tests.
- Downstream callers of the loss API (TrainingLoop, samples, NivaraTorch tests)
  are unaffected: no signature or behavior change.
- Test coverage: `tests/Nivara.Tests/AutoDiff/LossTests.cs` and
  `tests/Nivara.Tests/NivaraTorch/LossTests.cs` exercise all 5 losses.

## Verification

- `dotnet build Nivara.slnx` (after each step).
- `dotnet test` for the AutoDiff `LossTests` and NivaraTorch `LossTests`
  fixtures (requires explicit confirmation before running).

## Planned commits

1. `docs: plan loss Forward overload cleanup (#215) in TODO.md`
2. `test(autodiff): lock in Loss<T> two-arg Forward delegation contract (#215)`
3. `docs(autodiff): clarify Loss<T> Forward overload + CE batch divisor (#215)`

## GitHub issues log

- No new issues discovered during planning.
