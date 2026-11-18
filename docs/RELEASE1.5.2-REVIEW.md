# v1.5.2 Release Retrospective

Observations and improvement ideas from the v1.5.2 release process.
Items below are recorded as retrospective lessons; unfiled suggestions are explicitly
closed as **no action** unless a future issue is created and linked here.

## What went well

- **Last-release issues addressed before tagging.** The v1.5.1 retrospective items on
  the broken canonical benchmark runner (unescaped `)` inside a parenthesised `if` block
  made `exit /b 2` unconditional), LF-only `.cmd` files that `cmd.exe` cannot parse
  (`.gitattributes` forced `eol=lf` for all text), and the `release/<version>` vs
  `release/v<version>` folder-prefix mismatch are now fixed: the runner's `echo Usage`
  paren is escaped as `^)`, `.cmd`/`.bat` are forced to CRLF via `.gitattributes`,
  and the runner writes to `release\v<version>\`. See "Bugs / Tech Debt" below.
- **Version bump and changelog cut cleanly.** `PackageVersion` moved to `1.5.2` in both
  csproj files, `CHANGELOG.md` `## [Unreleased]` (none present) — top entry is
  `## [1.5.2] - 2026-11-16`, and the documentation gates stayed green after the
  edits (snippet gate + `DocsCoverageTests`).
- **Build and targeted gates green before tagging.** `dotnet build Nivara.slnx` succeeded
  (0 errors, 1 pre-existing NU5104 warning), `EveryGatedBlock_CompilesWithoutErrors`
  passed (1 test), `DocsCoverageTests` passed (1 test), and
  `dotnet format Nivara.slnx whitespace --verify-no-changes` passed.

## Bugs / Tech Debt

### 1. Benchmark runner was unusable as committed (carried, now fixed)

**Status:** Addressed. The committed `release/run-benchmarks.cmd` could never run:
its `if "%~1"==""` block contained `echo Usage: ... (e.g. 1.5.1) 1>&2`, whose
unescaped `)` terminated the parenthesised block early, making `exit /b 2` unconditional.
Additionally `.gitattributes` forced LF line endings on `.cmd` files, which `cmd.exe`
cannot parse. Fixed by escaping the paren (`^)`) and adding `*.cmd`/`*.bat` CRLF rules
to `.gitattributes`. The runner's output folder was also changed from `release\<version>\`
to `release\v<version>\` to match the canonical evidence convention.

**Recorded suggestions (not filed):**
- Add a CI check that runs `release/run-benchmarks.cmd`'s argument-parsing path
  (or at least `cmd /c` parses the file) so a broken batch file is caught before release.
- Document the LF/CRLF rule prominently; it affects every future `.cmd`/`.bat`.

### 2. `docs/` coverage accounting is mechanical (carried from v1.5.0/v1.5.1)

**Status:** Open - deferred. The six pins in `DocsCoveragePins` still move whenever
citations are added or removed, so release bookkeeping looks like reduced coverage.
Not attempted for this release to avoid destabilising the gates.

### 3. Stable package depends on prerelease dependencies (carried)

**Status:** Unchanged. `NU5104` suppression in both csproj files remains, documented
with an inline comment (Parquet.Net / .NET 11 RC cause; revert once those ship stable).

### 4. Docs refresh step applied unevenly (carried)

**Status:** Partially addressed. Step 6 of `RELEASING.md` requires a doc-refresh
confirmation listing each reviewed file with a named owner; the per-file review for this
release is recorded in the PR description. `README.md`, `GETTING-STARTED.md`,
`ARCHITECTURE.md`, `docs/AUTODIFF.md`, `EXAMPLES.md`, `samples/README.md`, `AGENTS.md`,
`RELEASING.md`, and `docs/adr/` should be confirmed reviewed before merge.

### 5. Changelog completeness

**Status:** Partially addressed. The `## [1.5.2]` section was expanded with the major
feature groups landed since v1.5.1 (frame-level dataset operations, public
`TryGetReadOnlyTensorSpan`, native-by-default model loading, Silk native paths, Qwen
parity fixtures, gradient-tracking fixes). A final review against `git log v1.5.1..HEAD`
is still recommended before tagging to catch any user-visible item missed.

## Release Process

### 6. Benchmarks run with approval

Benchmarks were launched with explicit human approval via the (now-fixed) canonical
runner `release/run-benchmarks.cmd 1.5.2` in the background with no timeout, writing
to `release/v1.5.2/` (renamed from the runner's `release/1.5.2/` output). Artefacts
expected: core perf harness (`nivara-perf-v1.5.2.json`), NivaraIncident, NivaraInference
CPU/Silk/BLAS plus PyTorch references, and NivaraFineTuning. README table regen
(Prev/Current shift) to be completed from these captures before tagging.

### 7. CI / local verification during prep

- `dotnet build Nivara.slnx` — passed (0 errors, 1 pre-existing NU5104 warning).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` — passed.
- `EveryGatedBlock_CompilesWithoutErrors` — passed (1 test, ~14 s).
- `DocsCoverageTests` — passed (1 test).
- Full suite (`dotnet test` with `Category!=Performance&...`) — to be run before tagging.

### 8. Missed steps / process friction

- Shell access via `pwsh` spawn was intermittently denied (environment); worked around
  by invoking `cmd.exe` directly and converting batch files to CRLF.
- The runner's `release\<version>` (no `v`) vs canonical `release\v<version>` mismatch
  required a post-run folder rename; fixed in the runner for next time.
- Background benchmark runs are long; the canonical runner remains the preferred entry
  point.

---

*Recorded during the v1.5.2 release prep on 2026-11-18.*