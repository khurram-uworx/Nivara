# Plan: Issue #633 — ApplyMask remarks cite retracted attention share

## Problem

`src/Nivara/AutoDiff/Operations/AttentionKernels.cs` remarks for `ApplyMask` quote a retracted
attention share: `1.3-2.0%` of a Laya-shaped attention pass, with the claim that the attention
total swings `48%` across three runs while the mask leg holds `36%`. Both are fractions of an
attention total that #492 retracted (packed multi-head buffers passed into `MatMulTransposedB`/
`MatMul` while declaring one head's dimensions, then multiplied by 16; `MultiplyCore` rents/copies/
clears by `span.Length`). The README retraction is in `tests/Nivara.PerformanceTests/README.md`.

What still stands and must be preserved: the absolute mask delta of `96-133 ms per 28-layer forward`,
timed on the correctly-sized `scores` buffer.

## Proposed changes

Edit only the "Measured, not assumed" paragraph in the `ApplyMask(Span<T>, ReadOnlySpan<T>)`
remarks (first overload). Replace the retracted percentage and swing explanation with the
absolute milliseconds and a note that any share of an attention total is retracted, pending a
same-run denominator. Do not introduce a new percentage.

Files:
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` (remarks only)

## Verification steps

- Build `Nivara.slnx` (Release) to confirm XML doc comments still compile.
- Targeted: grep for retracted phrases in src to ensure none remain.

## Planned commits

1. `docs: plan issue #633 in TODO.md`
2. `docs: remove retracted attention share from ApplyMask remarks`
3. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none new; #633 itself)

## Blast radius

Docs-only change in XML remarks. No runtime behavior, no API, no tests affected. Risk: trivial.
