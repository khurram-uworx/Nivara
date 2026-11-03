# Plan: Fix issue #860 — Docs: reconcile Arrow notes to close out #857

## Problem

A few Arrow reconciliation notes in `docs/plan/ARROW-REVIEW.md` and `docs/plan/ARROW-ROADMAP.md` still read as stale now that export zero-copy and validity wiring have landed. Specifically:

- `docs/plan/ARROW-REVIEW.md` line 9 (2026-08-02 reconciliation) still says “At that point no public API advertised zero-copy”, which clashes with the 2026-11-03 reconciliation stating export does. Export-only limitation is not first/unambiguous.
- `docs/plan/ARROW-ROADMAP.md` sequencing section must state explicitly that Phase D export landed independently of Phase A, while D-import, Phase F, and chunk-aware handling remain blocked on Phase A.
- “Phase A not started” wording must be explicit in the phase list and acceptance criteria.

## Proposed changes (doc-only)

1. `docs/plan/ARROW-REVIEW.md`:
   - Reorder the two reconciliation blockquotes (2026-11-03 first) and reword 2026-11-03 note so the export-only limitation is first/unambiguous.
   - Reword 2026-08-02 note tail to historical language.
   - Add a lead sentence under §4.1 making the export-only limitation first.
2. `docs/plan/ARROW-ROADMAP.md`:
   - Sequencing rationale wording: Phase C and export half of Phase D “landed independently of Phase A”; D-import, Phase F, chunk-aware handling “remain blocked on Phase A”.
   - Phase A heading + status line + acceptance-criteria bullet with “Phase A not started”.
   - “Where we are today” gets “(Phase A not started)” without dropping existing citation tokens.

## Verification

- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (expect 66/66, no pin deltas)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Inspect git diff for citation token additions/removals.

## Planned commits

1. `docs: plan issue #860 in TODO.md`
2. `docs: reconcile ARROW-REVIEW.md zero-copy notes for #860`
3. `docs: clarify ARROW-ROADMAP Phase A status and sequencing for #860`
4. `docs: remove TODO.md — plan executed`

## Blast radius

Documentation only (`docs/plan/*`). No source/test changes. Docs gate pins must remain unchanged; no `*.cs::Symbol` citation tokens added or removed.

## GitHub issues log

- (none created)
