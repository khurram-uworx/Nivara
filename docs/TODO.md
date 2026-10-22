# Plan: docs/ILGPU.md — record #791 A-tile vectorization measurement (issue #806)

## Problem
Issue #791 (merged via PR #803) vectorized A-tile staging in the Row4 GEMM family
(~8–10% staging speedup, byte-identity retained), but the result is not recorded in
`docs/ILGPU.md`. Risk of re-running the experiment.

## Proposed changes
Add a measurement note subsection to `docs/ILGPU.md` §10.1 (adjacent), between the
`### #784 candidates` subsection and `### 10.2 Lesson #468`, titled
`### A-tile vectorization (#791) — promoted`. Content covers outcome, ILGPU/Silk
implementations, speedup, byte-identity claim, gate used (`--gemm`, `--silk-kernels`),
and a "do not re-run" note.

No `File.cs:NN` citations are added (counts unchanged). No fenced code blocks added.

## Verification steps
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs namespace tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- Confirm DocsCoveragePins counts unchanged.

## Planned commits
1. docs: plan #806 in TODO.md
2. docs(ILGPU): record #791 A-tile vectorization measurement
3. docs: remove TODO.md — plan executed

## Blast radius
Single documentation file `docs/ILGPU.md`. No code changes. No citation count changes.

## GitHub issues log
- (none created yet)