# Plan: Issue #350 — de-gate flaky transpose perf probe

## Problem

`Transpose_PerformanceProbe_TiledKernelBeatsBclViewMaterialization` (tests/Nivara.Tests/Tensors/TensorsHelperTests.cs:472) is a `[Category("Performance")]` timing gate asserting a single-shape (1024×1024) ordering. It flakes on the author's machine even after #482's warmup/interleave/median fixes and `RequireOptimizedBuildForTiming()`.

## Blast radius

- tests/Nivara.Tests/Tensors/TensorsHelperTests.cs (test + const)
- docs/TENSORS.md prose claim about the gate
- No source/library changes; no citation-count pins affected.

## Proposed changes

1. Remove `const double TransposeMaxRatio = 0.9;`.
2. Remove the `Assert.That(result.MedianRatio, Is.LessThan(TransposeMaxRatio), ...)` assertion; keep the functional parity assertion and the ratio log line.
3. Update the method header comment to describe it as a non-gating measurement.
4. Update docs/TENSORS.md wording to point at the SimdProbe `transpose` mode as authoritative.

## Verification

- targeted test run passes and prints ratio
- dotnet format whitespace verify
- Docs snippet gate unaffected (no fenced blocks touched)

## Planned commits

1. docs: plan #350 in TODO.md
2. test: de-gate flaky transpose perf probe (#350)
3. docs: clarify transpose perf probe is non-gating
4. docs: remove TODO.md — plan executed

## GitHub issues log

- (none new)
