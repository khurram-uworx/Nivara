# Plan: Symbol-anchored documentation citations (issue #850)

## Problem

`docs/plan/ARROW-REVIEW.md` citations drifted again after #835 zero-copy export: line numbers now resolve to the wrong symbol (e.g. `ArrowInterop.cs:999-1015` lands on `CreateZeroCopyHalf`). Issue #841 generalised this: the citation gate only proves a file exists and a line is in range, so "resolves but describes the wrong symbol" is invisible. Root cause is line-number-based anchors, which are fragile under edits and not greppable by coding agents.

## Proposed changes (scope: ADR + gate + ARROW-REVIEW pilot)

### ADR-008 — symbol-anchored citations
New `docs/adr/008-symbol-anchored-citations.md`. Records: citations anchor to a source symbol (`File.cs::MemberName`) instead of a line number; signature appended only when needed to disambiguate overloads; gate resolves symbols via Roslyn (no regex on source); line-number form retired after migration. Add the ADR index line to AGENTS.md (required by `AdrIndexTests`).

### Gate: `tests/Nivara.Tests/Docs/DocCitations.cs`
- Add a Roslyn-based symbol index (cached per file, parsed once) that enumerates member declarations: class/struct/interface/record/enum, method (with signature), constructor, property, field, event. Each entry: name, normalised signature, kind, start/end line.
- Citation discovery in markdown keeps a regex (the only regex; source resolution uses Roslyn). New form: `File.cs::Symbol` with optional `(paramType, ...)` signature. Old `File.cs:NN` form still accepted during migration.
- Resolve `::` citations against the cached index: file must exist; symbol must exist; if multiple declarations share the name, a signature is required (else `AmbiguousSymbol`); if signature given, it must match (else `MissingSymbol`).
- New fault kinds: `MissingSymbol`, `AmbiguousSymbol`.
- Cache parsed trees in a `Dictionary<string,int>`-style cache keyed by repo-relative path to avoid re-parsing (citation tests are already slow).

### Negative control
`tests/Nivara.Tests/Docs/Fixtures/BrokenCitations.md` + `DocCitationTests.cs`: add `::`-form broken citations (missing symbol, ambiguous symbol) and a resolvable `::` citation; update expected fault count with justification.

### Pilot migration: `docs/plan/ARROW-REVIEW.md`
Retarget every `File.cs:NN` citation in that file to its enclosing symbol (`::MemberName`, signature where ambiguous). Use the Roslyn index to auto-map each old citation to its enclosing member, then review each mapping for intent (this is the #841 "right symbol" check).

### Follow-up GitHub issues (created during execution)
- Migrate remaining plan docs (POLARS-*, AISTACK-*, ARROW-ROADMAP) to symbol form.
- Migrate root docs, blog posts, and the widened-scope `*.cs` citations.
- Retire the `:` line-number form from the gate once all citations are migrated.

## Verification steps
- Build: `dotnet build Nivara.slnx`
- Gate: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running)
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- AdrIndexTests green after AGENTS.md update.

## Planned commits
1. `docs: plan symbol-anchored citations in TODO.md`
2. `docs: add ADR-008 symbol-anchored citations + AGENTS index`
3. `test(docs): add Roslyn symbol index and :: citation resolution to gate`
4. `test(docs): extend negative control and DocCitationTests for :: form`
5. `docs(plan): migrate ARROW-REVIEW.md citations to symbol anchors`
6. (follow-ups as separate issues, not in this PR)

## Blast radius
- `tests/Nivara.Tests/Docs/DocCitations.cs` (core gate logic + new symbol index).
- `tests/Nivara.Tests/Docs/DocCitationTests.cs`, `DocReferenceTests.cs` (counts/fault assertions).
- `tests/Nivara.Tests/Docs/Fixtures/BrokenCitations.md`.
- `docs/plan/ARROW-REVIEW.md` (pilot migration).
- `docs/adr/008-*.md`, `AGENTS.md`.
- `DocsCoveragePins` counts unchanged if migration is one-for-one (verify in G2).

## GitHub issues log
- (to be populated during execution)
