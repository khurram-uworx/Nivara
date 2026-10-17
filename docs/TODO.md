# Plan: Fix #489 — partially-saturated attention mask row

## Problem

When an attention mask fill is a large finite value that annihilates *some* but not all
cells in a query row, the backward pass still produces a non-zero `dS` for the
annihilated cells. Issue #454 fixed only the all-dead (fully saturated) row case
via a per-row flag; a partially saturated row is not flagged, so its dead cells are
never identified.

## Proposed changes

1. `AttentionKernels.ApplyMask` (bool overload): accept a span whose length is
   either `rows` (row mode, existing behaviour) or `rows*cols` (cell mode, write
   each cell's dead flag). Dead test: `mask != 0 && (score + mask) == mask`,
   plus the `-inf` branch.
2. `GradKernels.SoftmaxGradient`: when the flag span length is `rows*classCount`,
   compute the normal gradient for every row, then zero cells where the flag is
   set. Keep row mode for length `rows`.
3. `AttentionKernels.SoftmaxBackwardRows`: same length-based dispatch; bitmask
   overloads (`ulong`/`long`/`int`/`uint`) for `kvLen <= 64`.
4. Thread per-cell deadness through reverse/forward attention (single + batched):
   replace `bool[] savedScoreIndependentRows` (`numHeads*qLen`) with
   `bool[] savedDeadCells` (`numHeads*qLen*kvLen`); pass per-head slices of
   length `scoreLen` to `ApplyMask` and `SoftmaxBackwardRows`.
5. Add `ulong`/`long`/`int`/`uint` bitmask overloads of `ApplyMask` and
   `SoftmaxBackwardRows` (and `GradKernels.SoftmaxGradient`) that pack per-cell
   deadness into a per-row word.

## Blast radius

- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs`
- `src/Nivara/AutoDiff/Operations/GradKernels.cs`
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`
- `src/Nivara/AutoDiff/Operations/ForwardGradOperations.cs`
- Tests: `tests/Nivara.Tests/AutoDiff/AttentionKernelsTests.cs`,
  `GradKernelsTests.cs`, attention tests.

## Verification

- Build `Nivara.slnx`.
- Targeted tests: AttentionKernelsTests, GradKernelsTests, attention tests.
- Full suite only with human confirmation.

## Planned commits

1. docs: plan #489 in TODO.md
2. feat: per-cell deadness in ApplyMask + SoftmaxGradient
3. feat: bitmask overloads for ApplyMask/SoftmaxBackwardRows
4. refactor: thread per-cell deadness through attention (reverse)
5. refactor: thread per-cell deadness through attention (forward)
6. test: partial-saturation backward fixture
7. docs: remove TODO.md

## GitHub issues log

- (none yet)
