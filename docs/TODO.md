# Plan: Retire legacy `File.cs:NN` citation form (issue #854)

## Problem
Follow-up to #850/#852. Once all citations migrated to `File.cs::MemberName` form, retire support for legacy `File.cs:NN` line-number form in `DocCitationExtractor`. Acceptance:
- Regex only matches `::` form.
- `BrokenCitations.md` and `DocCitationTests` updated.
- Docs gate green.

## Proposed changes

### Code
- `tests/Nivara.Tests/Docs/DocCitations.cs`
  - `CitationPattern` regex drops `oldFile:start-end` alternation.
  - `Scan`: remove legacy branch; all citations are symbol citations.
  - `Faults`: line-range / `PastEndOfFile` branch becomes unreachable but keep enum member for compat.
  - Doc comments updated.

### Fixture
- `tests/Nivara.Tests/Docs/Fixtures/BrokenCitations.md`: convert/remove legacy citations to `::` form.

### Tests
- `tests/Nivara.Tests/Docs/DocCitationTests.cs`: remove range test, update references/counts.
- `tests/Nivara.Tests/Docs/Fixtures/BrokenReferences.md` + `DocReferenceTests.cs`: update legacy citation to `::`.

## Verification
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test ... Docs` filter
- `EveryCitationInTheDocumentation_Resolves`

## Planned commits
1. docs: plan #854 in TODO.md
2. test(docs): retire legacy citation form from gate
3. docs: remove TODO.md

## Blast radius
- Docs gate tests (67). No product code.
- `ExpectedCitationCount` stays 113 (docs already migrated).

## GitHub issues log
- #854 — retire legacy citation form (this plan)
