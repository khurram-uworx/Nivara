# Plan: retire docs/plan/NIVARACHAT-SILK-GAPS.md

**Branch:** khurram/nivarachat
**Date:** 2026-11-21

## Problem

`docs/plan/NIVARACHAT-SILK-GAPS.md` tracked six Silk gaps (S1–S6) for NivaraChat. All
six are now implemented and their issues (#1340, #1341, #1342, #1276, #1274, #1273)
are CLOSED (verified via `gh issue view`). The non-goals in the document are already
captured in `samples/NivaraChat/ARCHITECTURE.md` ("Not doing") and `docs/SILK.md`.
Nothing remains to salvage, so the document should be removed to avoid stale guidance.

## Proposed changes

1. Delete `docs/plan/NIVARACHAT-SILK-GAPS.md`.
2. Update dangling references:
   - `docs/plan/NIVARAINFERENCE-SILK-GAPS.md:10` — remove "The NivaraChat counterpart is
     `docs/plan/NIVARACHAT-SILK-GAPS.md`." (rephrase to note it was retired once all its
     gaps closed).
   - `docs/plan/LORA.md:14` — drop the file path from "Consolidates the design asks in
     `docs/plan/NIVARACHAT-SILK-GAPS.md` (Gap S5, …)".
3. No code/test/build changes.

## Blast radius

- No code, tests, or build artifacts reference the file (verified via repo-wide grep:
  only the two doc references above).
- `docs/plan/` is excluded from the doc-coverage scan (`DocSnippetExtractor.IsScanned`),
  so no `DocsCoveragePins` change is required.
- No `File.cs:NN` citations or ADR entries reference the file.

## Verification steps

- `git grep NIVARACHAT-SILK-GAPS` returns no matches.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` clean.
- Docs snippet gate (`EveryGatedBlock_CompilesWithoutErrors`) green (unchanged scope).

## Planned commits

1. `docs: plan retirement of NIVARACHAT-SILK-GAPS.md in TODO.md`
2. `docs: remove retired NIVARACHAT-SILK-GAPS.md and update dangling references`
3. `docs: remove TODO.md — plan executed` (after G2 review)

## GitHub issues log

- (none — this is a docs cleanup; no deferred work discovered)
