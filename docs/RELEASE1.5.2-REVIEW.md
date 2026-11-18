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
to `release/v1.5.2/`. `status.txt` reads `DONE` and all expected artefacts are present:
core perf harness (`nivara-perf-v1.5.2.json`), NivaraIncident, NivaraInference
CPU/Silk/BLAS plus five PyTorch references, and NivaraFineTuning. The core harness
results table was regenerated from the JSON (Prev/Current shift, header restored,
`Δ%` corrected), and the NivaraIncident, NivaraInference and NivaraFineTuning README
tables were refreshed from the remaining captures on 2026-11-18. Rows without a
same-session comparison leg are marked `—` or `(carried)` rather than divided
across captures.

### 7. CI / local verification during prep

- `dotnet build Nivara.slnx` — passed after the README refresh (0 errors, 1 pre-existing NU5104 warning).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` — passed.
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` — 66/66 passed, including `EveryGatedBlock_CompilesWithoutErrors` and `DocsCoverageTests`; no doc-pin drift from the README edits.
- Full filtered suite (`Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`) — 4,686 passed, 0 failed, 0 skipped (5 m 2 s).

### 8. Missed steps / process friction

- Shell access via `pwsh` spawn was intermittently denied (environment); worked around
  by invoking `cmd.exe` directly and converting batch files to CRLF.
- The runner's `release\<version>` (no `v`) vs canonical `release\v<version>` mismatch
  required a post-run folder rename; fixed in the runner for next time.
- Background benchmark runs are long; the canonical runner remains the preferred entry
  point.
- The canonical `regen_perf_table.py` copy (three identical copies under `samples/*/Python/`)
  had two defects: it dropped the `| Scenario | Prev | … |` header and separator on regen,
  and it computed `Δ%` against the *old* Prev instead of the shifted Prev, so the column
  did not match the Prev/Current pair. All three copies are now fixed (header preserved,
  `Δ% = (Current − Prev)/Prev`, and JSON-only results appended with `Prev = —`). The core
  harness table for v1.5.2 was regenerated with the corrected logic. Troubleshooting the
  core harness JSON also showed three new scenarios (`AutoDiff RMSNormModule fwd+bwd`,
  `Qwen decode block`, `Qwen decode fwd`) that had never been added as table rows; they
  are now present with `Prev = —`.

### 9. Capture coverage gaps in the 2026-11-18 run

**Status:** Documented in the README tables; no re-run in this release.

- The canonical runner invokes BLAS with 2 row-blocks for DistilBERT / SST-2 /
  Laya, while the older swept recommendation in the inference README was 4.
  The tables record the setting actually captured; the 4-block sweep was not
  repeated. Update the runner before the next release if 4 blocks are still the
  intended default.
- The inference sample does not print the OpenCL driver version, so no driver
  string is quoted with the Silk rows.
- PyTorch vision / SST-2 and the SmolLM F32 CPU/PyTorch pair were not rerun in
  this session. Their cells are marked `(carried)` and no ratio crosses sessions.
- The Qwen CPU leg used a 206-token tool prompt while the Silk legs used a
  36-token synthetic prompt, so the artefacts are retained but no same-prompt
  ratio is published.
- The fine-tuning capture prints separate Nivara CPU and Nivara Silk steady-state
  lines (1238.1 and 247.0 ms/batch). The README uses those printed values rather
  than deriving a timing from individual batch lines.

---

*Recorded during the v1.5.2 release prep on 2026-11-18.*