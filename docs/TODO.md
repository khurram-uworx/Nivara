# Plan: Issue #704 — Loosen docs-gate scan scope and pin-by-integer ratchet

## Problem

`DocReferenceExtractor.ScanFiles()` pins a raw file count (`ExpectedFileCount` ~752) over
every `.md`+`.cs` under `src/`, `samples/`, `tests/`, `docs/`, and the repo root. Every
add/remove of any in-scope file moves the pin, forcing pure-arithmetic pin edits. The scans
also sweep the whole `docs/` tree (including `docs/plan/` transient docs), dragging transient
documents into "coverage."

The gate's real safety net (deleted docs still cited, #549/#625) is the missing-target check,
not the ratchet. The ratchet only forces a human to acknowledge any file-set change.

Decisions for this plan (from issue comment):
- `docs/blog/`, `docs/research/` (and any `docs/<subfolder>/`) are **out of the pin**; only
  `docs/*.md` top-level counts. Anything in a subfolder of `docs/` is out.
- Keep the markdown pin (`ExpectedDocumentCount`) as-is for now.
- Deleting a top-level `docs/*.md` requires updating the pin.

## Proposed changes

1. `DocReferenceExtractor.ScanFiles()`:
   - C# files: keep reading all `*.cs` (for `File.cs:NN` citations) as before.
   - Markdown: only repo root `*.md` (non-recursive), `docs/*.md` (top-level, non-recursive),
     `docs/adr/*.md` (recursive), and `README.md`/`AGENTS.md` anywhere.
   - Keep existing exclusions: `docs/plan/`, `samples/data/`, `tests/Nivara.Tests/Docs/`,
     `docs/TODO.md`, build/vendor segments.
   - Drop the `ExpectedFileCount` assertion usage (remove pin).

2. `DocsCoveragePins` / `DocsCoverageTests`:
   - Remove `ExpectedFileCount` pin and its assertion (raw file count no longer pinned).
   - Keep all other pins (`ExpectedCitationCount`, `ExpectedCitedDocumentCount`,
     `ExpectedDocumentCount`, `ExpectedCodeCitationCount`, `ExpectedDocumentReferenceCount`).
   - Update arithmetic doc comments where they reference the removed pin.

3. `AGENTS.md`: update the "Change → test coupling" bullet about `ExpectedFileCount`.

## Blast radius

- `tests/Nivara.Tests/Docs/DocReferences.cs` — `ScanFiles()`, `FileCount()`.
- `tests/Nivara.Tests/Docs/DocsCoveragePins.cs`, `DocsCoverageTests.cs`.
- `tests/Nivara.Tests/Docs/DocReferenceTests.cs` — `ToleratedHistoricalMentions...` still uses
  `ScanExcludedDocuments`; unaffected.
- `AGENTS.md`.
- Downstream: `DocumentReferenceCount()` and `CodeCitationCount()` are still pinned and now
  computed over the narrowed markdown scope; counts will move and must be re-pinned
  deliberately (handled in review step).

## Verification

- Build `tests/Nivara.Tests`.
- Run targeted: `DocsCoverageTests`, `DocReferenceTests`, `DocCitationTests`,
  `DocumentationSnippetTests`.

## Planned commits

1. `docs: plan issue #704 in TODO.md`
2. `refactor(docs-gate): narrow widened scan markdown scope`
3. `test(docs-gate): drop raw file-count pin`
4. `docs: update AGENTS.md pin coupling note`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet)
