# Plan: Resolve issues #211 and #213

Branch: `khurram/issues` (off `main`). These are low-priority code-quality issues
filed from `docs/REVIEW-2026-08-12.md` findings #15 and #17.

## Problem

### #211 — `TensorsHelper.RowNorms`/`ColumnNorms` orphaned dead code

AGENTS.md/CHANGELOG claim the frame tensor-axis `RowNorms`/`ColumnNorms` were
deleted in the AutoDiff refactor, but the `TensorsHelper.RowNorms` kernel
survives with no production callers. Investigation findings:

- `TensorsHelper.RowNorms<T>` exists at `src/Nivara/Tensors/TensorsHelper.cs:570-597`.
- `TensorsHelper.ColumnNorms` does **not** exist — zero matches in any `.cs` file
  (issue body's line 595 reference is stale; that line is now inside `RowNorms`).
- Only callers are two unit tests:
  `RowNorms_NoNulls_MatchesPerRowScalar` and
  `RowNorms_WithRowNulls_MasksOnlyNullRows`
  at `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs:476-511`.
  The issue's "verify no callers" step is wrong — the tests must go too.
- Surviving kernels `RowDot`, `RowCosineSimilarity`, `AnyTrue`,
  `ValidateRowKernelArgs`, and the `FillRowMajor` test helper all remain in use
  by `NivaraFrame.RowDot`/`RowCosineSimilarity` and other tests. Keep them.

### #213 — `TrainingLoop<T>`/`DataParallelTrainer<T>` unsealed

The issue claims neither class has virtual/protected members and neither is
subclassed, so they should be `sealed`. Investigation findings (both claims
factually wrong):

- Both classes DO have a designed `protected virtual` hook surface:
  - `TrainingLoop<T>`: `OnEpochStart`, `OnBatchEnd`, `OnEpochEnd`, `Dispose(bool)`
    (`src/Nivara/AutoDiff/Training/TrainingLoop.cs:155-173`).
  - `DataParallelTrainer<T>`: `OnEpochStart`, `OnEpochEnd`
    (`src/Nivara/AutoDiff/Training/DataParallelTrainer.cs:267-268`).
- Both ARE subclassed and exercised by tests:
  - `HookTrackingTrainingLoop : TrainingLoop<float>`
    (`tests/Nivara.Tests/AutoDiff/TrainingTests.cs:431`),
    asserted by `TrainingLoop_VirtualMethods_AreCalled`.
  - `HookTrackingTrainer : DataParallelTrainer<float>`
    (`tests/Nivara.Tests/AutoDiff/DataParallelTests.cs:164`),
    asserted by `Run_Hooks_AreCalled`.
- Sealing would break the test build and remove the designed extension API.
  This falls under AGENTS.md's "unless inheritance is explicitly designed"
  carve-out. **Decision (human-confirmed): close #213 as not applicable.**

## Proposed changes

1. Delete `RowNorms<T>` (incl. its XML doc comment) from
   `src/Nivara/Tensors/TensorsHelper.cs`.
2. Delete the two orphaned `RowNorms` tests from
   `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`.
3. Post findings comment on #213 and close it as not applicable.

## Blast radius

- **`src/Nivara/Tensors/TensorsHelper.cs`** — removes one public static method.
  No downstream callers besides the deleted tests (verified via grep + code-memory
  dependency trace).
- **`tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`** — removes 2 test methods;
  `FillRowMajor` stays (used by `RowDot`/`RowCosineSimilarity` tests).
- **Public API** — `RowNorms` was public; removal is a source-breaking change but
  consistent with the documented "already removed" claim. No public API baseline
  (PublicAPI.txt / ApiCompat) exists in the repo.
- **`#213`** — no source change; only issue bookkeeping on GitHub.

## Verification

- `rg -n "RowNorms" --type csharp` → only doc references remain.
- `dotnet build Nivara.slnx` (confirm before long-running tests).
- Ask before running `dotnet test`; run `TensorsHelperTests` at minimum.

## Planned commits

1. `docs: plan issues #211/#213 in TODO.md`
2. `refactor: remove orphaned TensorsHelper.RowNorms dead code` (+ test removal)
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #211 — TensorsHelper.RowNorms/ColumnNorms orphaned dead code (implementing)
- [ ] #213 — TrainingLoop/DataParallelTrainer unsealed (closing as not applicable:
      protected virtual hooks are a designed inheritance surface, exercised by tests)

> As each task executes, if you find deferred work or a concern outside this plan,
> create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`)
> and record its number here — don't rely on memory.
