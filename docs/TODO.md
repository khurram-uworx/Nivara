# Plan: Migrate legacy `File.cs:NN` citations to symbol-anchored form (issues #852, #853)

## Problem

ADR-008 introduced symbol-anchored citations (`File.cs::MemberName`). The gate still accepts the legacy
`File.cs:NN` form during migration. Issues #852 and #853 cover the remaining plan docs and root/blog/widened
`*.cs` citations. Once migrated, #854 retires the legacy form. This branch does the migration only.

## Scope (this branch)

- `docs/plan/POLARS-REVIEW.md`, `POLARS-ROADMAP.md`, `AISTACK-REVIEW.md`, `AISTACK-ROADMAP.md`, `ARROW-ROADMAP.md` (#852)
- Top-level `docs/*.md`, `docs/adr/*.md`, `docs/blog/*.md` (#853)
- Widened-scope `*.cs` citations in `src/`, `samples/`, `tests/` (#853)
- Uncovered scanned files: `AGENTS.md`, `CHANGELOG.md`, `samples/NivaraChess/README.md`,
  `tests/Nivara.PerformanceTests/README.md` (required so #854 can retire the legacy form)

## Approach

Retarget each legacy `File.cs:NN` (or `File.cs:NN-MM`) citation to `File.cs::MemberName`, where `MemberName`
is the declaration containing the cited line. Count-preserving: one legacy citation -> one symbol citation.
Verify each retarget resolves via the citation gate. Do not touch the gate's deliberate negative controls
(`BrokenCitations.md`, `BrokenReferences.md`) — those are #854's job.

## Planned commits

1. `docs: plan migration of legacy citations in TODO.md`
2. `docs(plan): migrate POLARS-REVIEW.md citations`
3. `docs(plan): migrate POLARS-ROADMAP.md citations`
4. `docs(plan): migrate AISTACK-REVIEW.md citations`
5. `docs(plan): migrate AISTACK-ROADMAP.md citations`
6. `docs(plan): migrate ARROW-ROADMAP.md citations`
7. `docs: migrate top-level docs/*.md citations`
8. `docs: migrate docs/adr/*.md citations`
9. `docs: migrate docs/blog/*.md citations`
10. `chore: migrate widened-scope *.cs citations`
11. `docs: migrate AGENTS/CHANGELOG/samples/tests README citations`
12. `docs: remove TODO.md — plan executed` (after G2)

## Verification

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask first)
- `EveryCitationInTheDocumentation_Resolves` passes
- `DocsCoverageTests` pins unchanged (count-preserving)

## Blast radius

Touches many markdown files and a few `.cs` comments. No runtime code. Gate pins unchanged if count-preserving.

## GitHub issues log

- #852 — plan docs migration
- #853 — root/blog/cs migration
- #854 — retire legacy form (separate, after this branch)
