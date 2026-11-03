# Plan: sweep stale `File.cs:NN` citations (issue #841)

## Problem

Issue #832 fixed a class of citation defect in `docs/plan/ARROW-REVIEW.md`: a
`File.cs:NN` citation can pass the docs gate (the file exists and the line is in
range) while pointing at a different symbol than the sentence describes. The
same drift is expected in the other plan documents, which were written against
the same pre-consolidation code.

Audit scope, per the issue:

- `docs/plan/POLARS-REVIEW.md`
- `docs/plan/POLARS-ROADMAP.md`
- `docs/plan/AISTACK-REVIEW.md`
- `docs/plan/AISTACK-ROADMAP.md`
- `docs/plan/ARROW-ROADMAP.md` (untouched sections)

`docs/plan/ARROW-REVIEW.md` is out of scope: #832 already swept it. (It has
since drifted again around the #835 zero-copy export work; that is a separate
follow-up, not part of this plan.)

## Findings

1. `docs/plan/POLARS-ROADMAP.md` (Phase 1, generic-math collapse):
   `src/Nivara/NivaraColumn.cs:57-76` and shorthand `:188-208` resolve to
   tensor-primitive helper methods and `getTypeSize`, not the claimed
   "explicit `float`/`double` branches". Those branches no longer exist:
   arithmetic routes through `NumericKernelDispatcher`.
   - Retarget to the arithmetic operator overloads:
     `src/Nivara/NivaraColumn.cs:1837-1909`.
   - Correct the wording so it no longer claims explicit `float`/`double`
     branches; the remaining work is exposing generic-math methods on the
     column surface.

2. `docs/plan/ARROW-ROADMAP.md` (Where we are today):
   `ArrowInterop.cs:999-1015` now resolves to `CreateZeroCopyHalf` after the
   #835 zero-copy export work, not the chunked-array flattening loop.
   - Retarget to the shifted import loop:
     `ArrowInterop.cs:1100-1116`.

No stale citations found in:

- `docs/plan/POLARS-REVIEW.md` (full and shorthand citations verified against
  `FusedExpressionEvaluator`, `FusedKernel`, `NumericKernelDispatcher`,
  `NivaraColumn`, `IQueryInterfaces`, `SortByExpressionOperation`,
  `ColumnExpression`, `WindowSpec`).
- `docs/plan/AISTACK-REVIEW.md` (no `File.cs:NN` citations; the shorthand
  `:84-88` verifies against `TensorInteropExtensions`).
- `docs/plan/AISTACK-ROADMAP.md` (both `TensorInteropExtensions.cs:84-88`
  citations verify against the copy-to-protect-immutability path).

## Proposed changes

- Retarget the two citations above, preserving citation counts.
- Adjust the POLARS-ROADMAP sentence that made the now-false
  "explicit `float`/`double` branches" claim.
- No code changes, no new documents.

## Coverage pins

Both edits are count-preserving retargets. Expected to leave
`DocsCoveragePins.ExpectedCitationCount` (107),
`ExpectedCitedDocumentCount` (18),
`ExpectedDocumentCount` (54),
`ExpectedCodeCitationCount` (24), and
`ExpectedDocumentReferenceCount` (229) unchanged. Re-check after the branch
diff; only revise a pin if the actual count moved, and extend its arithmetic
comment rather than bumping silently.

## Blast radius

- Documentation only: two markdown files under `docs/plan/`.
- No source, test, or public-API behaviour changes.
- The docs gate (`Nivara.Tests.Docs`) is the affected test surface. The
  citation gate checks resolution, not semantics, so green before and after is
  expected; the value of this change is semantic.
- `DocsCoveragePins` expected unchanged.

## Verification

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Manual re-read of each retargeted citation against the cited source lines.

## Planned commits

1. `docs: plan citation sweep for issue #841 in TODO.md`
2. `docs(plan): retarget stale citations in POLARS and ARROW roadmaps`
3. `docs: remove TODO.md - plan executed`

## GitHub issues log

- [ ] #841 — stale `File.cs:NN` citations across remaining plan documents (this
  plan).
- [ ] #850 — `ARROW-REVIEW.md` citations drifted again after #835 zero-copy
  export (out of #841 scope).
