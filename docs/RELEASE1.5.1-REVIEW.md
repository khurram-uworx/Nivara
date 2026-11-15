# v1.5.1 Release Retrospective

Observations and improvement ideas from the v1.5.1 release process.
Items below are recorded as retrospective lessons; unfiled suggestions are explicitly
closed as **no action** unless a future issue is created and linked here.

## What went well

- **Last-release issues partially addressed before tagging.** The v1.5.0 retrospective
  items on undocumented `NU5104` suppression, uneven doc refresh, and missing
  post-tag version reset are now covered in `RELEASING.md` (Step 6 doc-refresh
  checklist, Step 10 reset) and in a csproj comment. See "Bugs / Tech Debt" below
  for the items that remain open.
- **Version bump and changelog cut cleanly.** `PackageVersion` moved to `1.5.1` in
  both csproj files, `CHANGELOG.md` `## [Unreleased]` became `## [1.5.1] - 2026-11-15`,
  and the documentation gates stayed green after the pin adjustments.

## Bugs / Tech Debt

### 1. Benchmark run captured for v1.5.1

**Status:** Addressed. With human approval, all release benchmarks were run and their
artefacts committed under `release/v1.5.1/`: core perf harness
(`nivara-perf-v1.5.1.json`), NivaraIncident (`nivaraincident-v1.5.1.txt`),
NivaraInference CPU/Silk/BLAS plus PyTorch reference runs, and NivaraFineTuning
(`nivarafinetuning-v1.5.1.txt`). README tables still need the Prev/Current shift
pass from these captures.

**Recorded suggestions (not filed):**
- Split the harness into a machine-runnable quick mode, or document expected runtimes
  in each `## Release Benchmark` section.
- Cache and commit benchmark JSON from a contributor machine where the weights are
  available.

### 2. `docs/` coverage accounting is mechanical (carried from v1.5.0)

**Status:** Open — deferred. The six pins in `DocsCoveragePins` still move whenever
citations are added or removed, so release bookkeeping looks like reduced coverage.
Splitting the pin into "live docs" and "CHANGELOG" scopes remains a future
improvement; not attempted for this release to avoid destabilising the gates.

### 3. Stable package depends on prerelease dependencies (carried from v1.5.0)

**Status:** Partially addressed. `NU5104` suppression in both csproj files is now
documented with an inline comment explaining the Parquet.Net / .NET 11 RC cause and
the revert condition. The override itself remains; revert once those dependencies
ship stable.

### 4. Docs refresh step applied unevenly (carried from v1.5.0)

**Status:** Addressed for process. Step 6 of `RELEASING.md` now requires a
doc-refresh confirmation listing each reviewed file with a named owner; the actual
per-file review for this release is recorded in the PR description.

### 5. Version arithmetic across the preview line (carried from v1.5.0)

**Status:** Addressed for process. `RELEASING.md` Step 10 documents the post-tag
reset to the next dev line (`1.6.0-preview` or `2.0.0-preview`) on `main`.

## Release Process

### 6. Benchmarks run with approval

Benchmarks were run with explicit human approval, launched via a temporary
`release.cmd` in the background with no timeout, and the artefacts committed under
`release/v1.5.1/`. Core perf table regenerated; inference table partially
refreshed; incident/fine-tuning tables not yet regen'd.

### 7. CI failures during prep

**Status:** Resolved. The documentation reference pin drifted twice because edits
to `RELEASING.md`/`docs/RELEASE1.5.1-REVIEW.md` added or removed backticked
`.md` references; corrected `DocsCoveragePins.ExpectedDocumentReferenceCount` to
490. Lesson: any doc edit that adds/removes a backticked `.md` path moves the
pin — update it in the same commit.

### 8. Missed steps / process friction

- README table regen was only done for the core harness and a few inference rows;
  incident/fine-tuning tables and most inference rows remain stale.
- Doc refresh (Step 2) was applied unevenly; only RELEASING/AGENTS/inference README
  touched, not the full list.
- Shell access was intermittently denied (pwsh spawn), forcing retries.
- Background benchmark runs are long; the canonical `release/run-benchmarks.cmd`
  runner and `release/README.md` were added to reduce friction next time.
- `release/wrap-table.py` added to automate hidden-marker wrapping.

---

*Recorded during the v1.5.1 release prep on 2026-11-15.*