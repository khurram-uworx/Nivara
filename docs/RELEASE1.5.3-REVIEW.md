# v1.5.3 Release Retrospective

Observations and improvement ideas from the v1.5.3 release process.
Items below are recorded as retrospective lessons; unfiled suggestions are explicitly
closed as **no action** unless a future issue is created and linked here.

## What went well

- **Last-release issues addressed before tagging.** The v1.5.2 retrospective items on
  the benchmark runner, CRLF `.cmd` handling, and the `release/v<version>` evidence
  folder are carried forward and verified; the runner was reused unchanged for v1.5.3.
- **Version bump and changelog cut cleanly.** `PackageVersion` moved to `1.5.3` in both
  csproj files, CHANGELOG.md gained a ## [1.5.3] - 2026-11-20 entry, and the
  documentation gates were kept green after the edits.
- **Benchmarks run with approval on AC power.** The canonical runner
  `release/run-benchmarks.cmd 1.5.3` was launched in the background with no timeout,
  writing to `release/v1.5.3/`.

## Bugs / Tech Debt

### 1. `docs/` coverage accounting is mechanical (carried from v1.5.0/v1.5.1/v1.5.2)

**Status:** Open - deferred. The pins in `DocsCoveragePins` still move whenever
citations or reader-facing documents are added or removed; this release adds
docs/RELEASE1.5.3-REVIEW.md, so ExpectedDocumentCount moves from 61 to 62.
Not restructured this release to avoid destabilising the gates.

### 2. Stable package depends on prerelease dependencies (carried)

**Status:** Unchanged. `NU5104` suppression in both csproj files remains, documented
with an inline comment (Parquet.Net / .NET 11 RC cause; revert once those ship stable).

### 3. Docs refresh step applied unevenly (carried)

**Status:** Partially addressed. Step 6 of RELEASING.md requires a doc-refresh
confirmation listing each reviewed file with a named owner; the per-file review for this
release is recorded in the PR description. README.md, GETTING-STARTED.md,
ARCHITECTURE.md, docs/AUTODIFF.md, EXAMPLES.md, samples/README.md, AGENTS.md,
RELEASING.md, and docs/adr/ should be confirmed reviewed before merge.

### 4. Changelog completeness

**Status:** Partially addressed. The `## [1.5.3]` section was drafted from
`git log v1.5.2..HEAD`. A final review against the same range is still recommended
before tagging to catch any user-visible item missed.

## Release Process

### 5. Benchmarks run with approval

Benchmarks were launched with explicit human approval via the canonical runner
`release/run-benchmarks.cmd 1.5.3` in the background with no timeout, writing to
`release/v1.5.3/`. The run exited with code 1 and did not write `status.txt`; the
captured subset and the gaps are documented in "Analysis of the canonical benchmark
run" below. README table refresh from the captured subset is tracked in the PR
description.

### 6. CI / local verification during prep

- `dotnet build Nivara.slnx` — passed (0 errors, 1 pre-existing NU5104 warning).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` — passed.
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` —
  passed (66/66, including `EveryGatedBlock_CompilesWithoutErrors` and `DocsCoverageTests`).
  `DocsCoveragePins.ExpectedDocumentReferenceCount` moved 516 → 518 to account for two
  additional markdown references introduced by this review document.

### 7. Missed steps / process friction

- Shell access via `pwsh` spawn was intermittently denied during prep; worked around by
  retrying and by reading artefacts through the file tools.
- Background benchmark runs are long; the canonical runner remains the preferred entry
  point.
- The review doc itself is a reader-facing docs/*.md, so the document pin moves by one;
  update `DocsCoveragePins.ExpectedDocumentCount` in the same commit.

### 8. Capture coverage gaps

- Silk/OpenCL driver version is not printed by the inference sample, so no driver string
  is quoted with the Silk rows.
- Any model whose run could not be captured in this session is recorded as `—` or
  `(carried)` in the README tables rather than divided across sessions.

## Analysis of the canonical benchmark run (2026-11-20)

The run launched as `release/run-benchmarks.cmd 1.5.3` (background, no timeout) completed
with exit code 1 and did **not** write `release/v1.5.3/status.txt`, so it is treated as
incomplete rather than fully successful.

### What completed

- Core perf harness — `release/v1.5.3/nivara-perf-v1.5.3.json` (57 scenarios) plus `core.txt`.
- NivaraIncident — `nivaraincident-v1.5.3.txt`.
- NivaraInference CPU — `nivarainference-cpu-v1.5.3.txt` (ends at ModernBERT-large).
- NivaraInference Silk — `nivarainference-silk-v1.5.3.txt` (ends at SmolLM BF16).
- NivaraInference BLAS — `nivarainference-blas-v1.5.3.txt`, but only partially:
  minilm, DistilBERT/SST-2, ModernBERT-large (choice1/choice2) and Laya (choice2)
  captured; ModernBERT choice3/choice4 and the rest of Laya did not finish.

### What did not complete

- ModernBERT-large BLAS choice3/choice4 and the remainder of the Laya BLAS run
  (each choice takes ~3.5 minutes).
- PyTorch reference captures (`pytorch-*-v1.5.3.txt`) — none produced.
- NivaraFineTuning capture (`nivarafinetuning-v1.5.3.txt`) — none produced; the
  pre-existing `samples/NivaraFineTuning/benchmark_results.txt` is from 2026-11-18
  and was not refreshed.

### What took the time

The dominant cost is the **BLAS leg of ModernBERT-large and Laya** (≈3.5 min per
choice, several choices per model), followed by the PyTorch reference scripts and
fine-tuning. The core harness, incident, CPU and Silk legs were comparatively fast.

### Decision taken

Given the time cost and the carried-over precedent from v1.5.2 (rows without a
same-session comparison leg are marked `—` or `(carried)` rather than divided across
sessions), this release proceeds with the partial capture set above. The missing
BLAS choices, PyTorch references and fine-tuning numbers are recorded as gaps rather
than re-run. If a fuller capture is wanted later, re-run only
`modernbert benchmark --blas --threads 4` and `laya benchmark --blas --threads 2`
plus the PyTorch scripts and `samples/NivaraFineTuning/benchmark_timing.cmd`, then
append the artefacts under `release/v1.5.3/` without overwriting the existing files.

### Local verification outcome

- `dotnet build Nivara.slnx` — succeeded (0 errors, 1 pre-existing NU5104 warning).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` — passed.
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` —
  passed (66/66). `ExpectedDocumentReferenceCount` adjusted 516 → 518.

---

*Recorded during the v1.5.3 release prep on 2026-11-20.*
