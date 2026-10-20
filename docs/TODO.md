# Plan: Consolidate the six docs coverage pins (issue #677)

## Problem

Six pinned coverage counts guard the documentation gates, split across two fixtures
(`DocCitationTests`, `DocReferenceTests`). Each fixture asserts its own subset in one
`Assert.Multiple`, so a partial local run comes back green while CI fails on the other
fixture. `ExpectedDocumentReferenceCount` is not currently pinned at all, even though
`DocReferenceExtractor.DocumentReferenceCount()` exists.

## Proposed changes

1. New `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` holding all six constants, each
   with its original arithmetic `<summary>` doc comment preserved verbatim.
2. New `tests/Nivara.Tests/Docs/DocsCoverageTests.cs` with one fixture and one test
   asserting all six counts in a single `Assert.Multiple`.
3. `DocCitationTests.cs`: remove the three constants and the
   `Coverage_IsAccountedForAcrossTheWholeRepository` method; update
   `ScanScope_ExcludesThePlanDocumentWhileItExists_AndNoCitationPinMoves` to read
   `DocsCoveragePins.ExpectedDocumentCount`.
4. `DocReferenceTests.cs`: remove the two constants and the
   `Coverage_IsAccountedForAcrossTheWidenedScope` method.
5. Restore the `ExpectedDocumentReferenceCount` pin (value 362) with its arithmetic
   history comment.

Values (verified by running the extractors):
- ExpectedCitationCount = 114
- ExpectedCitedDocumentCount = 19
- ExpectedDocumentCount = 53
- ExpectedFileCount = 735
- ExpectedCodeCitationCount = 127
- ExpectedDocumentReferenceCount = 362

## Blast radius

- Test files only (`tests/Nivara.Tests/Docs/`). No production code, no extractors.
- Removes 2 tests, adds 1 => net -1 test in the Docs namespace (66 total).
- `ScanScope_...` test still references the document count via the new pins class.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Perturb a `docs/*.md` file, run the Docs filter, confirm two pins from both old
  fixtures report in one run, then revert.

## Planned commits

1. `docs: plan consolidation of docs coverage pins in TODO.md`
2. `test(docs): add DocsCoveragePins and consolidated DocsCoverageTests`
3. `test(docs): remove per-fixture coverage tests and repoint references`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- #677 — consolidate the six docs coverage pins (this work)
