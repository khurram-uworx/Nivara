# Releasing Nivara

This document describes how to cut a new release of the `Nivara` and `Nivara.Extensions`
NuGet packages. Publishing is automated by the tag-triggered CD workflow
(`.github/workflows/cd.yml`), so the main work is updating docs, bumping versions, and
creating the tag.

## Prerequisites

- Push access to `khurram-uworx/Nivara`.
- The `NUGET_API_KEY` secret must be set in the GitHub repository settings so the CD
  workflow can publish to NuGet.
- **.NET SDK** — the same version as the repo target framework (check `global.json` or
  the `.csproj` `TargetFramework`).
- **Python 3 + PyTorch** — required for the NivaraInference and NivaraFineTuning
  benchmarks only. Install from https://pytorch.org/get-started/locally/. Verify with
  `python -c "import torch; print(torch.__version__)"`.
- **Model weights** — download HuggingFace weights before running inference benchmarks
  (see `samples/NivaraInference/README.md` Quick start for `hf download` commands).
- **NivaraIncident dataset** — generate with
  `dotnet run --project samples/NivaraIncident/NivaraIncident.Cli -- generate --dataset ./data/incident-lab --scenario A --scale 1`
  before running NivaraIncident benchmarks.

## Release checklist

### Step 1 — Draft the changelog

In `CHANGELOG.md`, replace the `## Unreleased` heading (if any) with
`## [X.Y.Z] - YYYY-MM-DD` and make sure every user-visible change since the last tag is
captured — breaking changes first, then features/fixes. Use the previous release entry as a
style template.

### Step 2 — Update public docs

Review these files for anything that changed since the last tag and refresh as needed:
- `README.md`
- `GETTING-STARTED.md`
- `ARCHITECTURE.md`
- `docs/AUTODIFF.md`
- `EXAMPLES.md`
- `samples/README.md`
- `AGENTS.md`
- The `docs/adr/` decision records, when a change matches an ADR decision.

### Step 3 — Bump package versions

In both `src/Nivara/Nivara.csproj` and `src/Nivara.Extensions/Nivara.Extensions.csproj`:
- Set `<PackageVersion>` to the new version.
- Update `<PackageReleaseNotes>` to summarize the new release (the previous release's
  notes are not accumulated).

The csproj `PackageVersion` is the source of truth for the repo state; the CD workflow
overrides it at pack time via `-p:PackageVersion=$(VERSION)`, but keeping it in sync means
a plain local `dotnet pack` also produces the right version.

### Step 4 — Capture machine configuration

Before measuring, record the machine config **once** and reuse it across all perf documents.
Include it in each document's recording line using the format:

```
*Recorded YYYY-MM-DD — <CPU model>, <logical cores> logical processors, .NET <version>, <extra context>.*
```

For example:
```
*Recorded 2026-08-21 — Intel Core Ultra 7 255H, 16 logical processors, .NET 10.0.11, 10M records, scenario A.*
```

Record at minimum: CPU model and core count, .NET SDK/runtime version, and any
environment-specific notes (OS, RAM, whether the machine was idle or busy, GPU present
but unused, etc.).

### Step 5 — Run benchmarks and update docs

> **⚠️ You MUST run every benchmark listed below and record actual measured numbers.
> Do NOT update recording dates, machine info, or table values without first running
> the corresponding benchmark command and capturing its output.** If a benchmark cannot
> be run (missing Python, missing model weights, etc.), note it explicitly in the PR
> description and leave the existing numbers in place with a comment — do not fake or
> carry forward stale numbers as if they were fresh.

There are four benchmark documents. Each must be updated with fresh measurements. The
table format uses a rolling Prev/Current history: the previous reading moves to **Prev**
and the fresh measurement becomes **Current**, with **Ratio** (`Current / Prev`) and
**Δ%** (`((Current − Prev) / Prev) × 100`). When there is no prior reading on the same
machine, Current is the first entry and Prev/Ratio/Δ% are left blank.

#### 5a. Core performance harness (pure .NET, no external deps)

```powershell
dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json <path> --runs 3
```

This produces JSON with per-scenario `ops/s`, `B/op`, `gen0/op` (medians of 3
independent child processes). Update `tests/Nivara.PerformanceTests/README.md`:

- Shift the existing **ops/s** column to a new **Prev** column.
- Place the fresh measurements in the **Current** column.
- Add a **Ratio** column (`Current / Prev`) and a **Δ%** column.
- Keep **B/op** and **gen0/op** as-is (stability indicators, not throughput metrics).
- If a scenario is new (no prior reading), leave Prev/Ratio/Δ% blank for that row.
- Update the machine line and recording date.
- Replace the old "point A" policy language with "rolling Prev/Current history".

**Gate:** The JSON file must exist on disk with valid numbers before proceeding. Save it
to a known path (e.g., `tests/Nivara.PerformanceTests/baseline-vX.Y.Z.json`) so it can
be referenced in the PR.

#### 5b. NivaraIncident benchmarks (pure .NET, needs generated dataset)

Requires the incident dataset (see Prerequisites). Run:

```powershell
# Nivara-only analysis timings
dotnet run --project samples/NivaraIncident/NivaraIncident.Cli -c Release \
  -- analyze --benchmark --dataset ./data/incident-lab --scenario A

# Nivara vs Polars (requires Python + Polars — see samples/NivaraIncident/Python/)
dotnet run --project samples/NivaraIncident/NivaraIncident.Cli -c Release \
  -- analyze --benchmark --dataset samples/data/benchmark-1m --scenario A --records 1000000
```

Update `samples/NivaraIncident/README.md`:

- **Nivara-only table:** shift existing timing to **Prev**, place new timings in
  **Current**, add **Ratio** and **Δ%** columns.
- **Polars comparison table:** keep the Nivara/Polars/Ratio structure; update numbers
  and recording line. Add a **Prev** column only if there is a prior same-machine
  measurement.
- Update the machine line and recording date in both tables.

#### 5c. NivaraInference benchmarks (requires Python + PyTorch + model weights)

Requires Python, PyTorch, and HuggingFace model weights (see Prerequisites). Run both
sides in the same session for fair comparison:

```powershell
# Nivara (C#)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark

# PyTorch (Python) — run immediately after Nivara on the same machine
cd samples/NivaraInference/Python
python benchmark.py  # or the equivalent per-model script
```

Update `samples/NivaraInference/README.md`:

- Shift existing timing columns to **Prev (PyTorch)** / **Prev (Nivara)**.
- Place fresh measurements in **Current (PyTorch)** / **Current (Nivara)**.
- Add **Prev Slowdown** (old ratio) and **Current Slowdown** (new ratio) columns.
  Alternatively, if the table is too narrow, keep single PyTorch/Nivara columns and
  add a **Δ%** column for Nivara only.
- Update the machine line, recording date, and prose referencing ratios.

#### 5d. NivaraFineTuning benchmarks (requires Python + PyTorch + model weights)

Requires Python, PyTorch, and DistilBERT SST-2 weights (see Prerequisites). Run:

```powershell
# Nivara fine-tuning (runs both sides via benchmark_timing.cmd)
cd samples/NivaraFineTuning
.\benchmark_timing.cmd
```

The script runs Nivara (Release, Server GC + Tiered PGO) and PyTorch
(`torch_threads = nproc`) side by side and writes results to `benchmark_results.txt`.

Update `samples/NivaraFineTuning/README.md`:

- Apply the same Prev/Current/Ratio/Δ% pattern for the Nivara timings.
- Update the Slowdown column and extrapolated full-run estimates.
- Update the recording line with machine info and date.

### Step 6 — Commit, PR, merge

Commit the release prep (docs + version bump) on a branch, open a PR, and merge to
`main`:

```powershell
git checkout -b release/<version>
# ... edits from steps 1-5 ...
git add .
git commit -m "chore: prepare vX.Y.Z release (version bump, release notes, changelog)"
git push -u origin release/<version>
gh pr create --repo khurram-uworx/Nivara --base main \
  --title "chore: prepare vX.Y.Z release" --body-file pr-body.md
```

The PR description must include:
- A checklist confirming which benchmarks were actually run (with paths to saved JSON
  output where applicable).
- Any benchmarks that could not be run and the reason (missing deps, etc.).
- The CD test results (3200+ tests passing).

### Step 7 — Verify the build before tagging

The CD workflow builds and runs the full test suite, so a broken build or failing tests
will fail the release. Locally, run `dotnet build Nivara.slnx` (and `dotnet test` when
practical) before pushing the tag.

### Step 8 — Tag and push

Annotated tags are preferred:

```powershell
git tag -a vX.Y.Z -m "Nivara vX.Y.Z"
git push origin vX.Y.Z
```

Pushing a `v*` tag to `main` triggers the CD workflow, which builds, tests, packs, and
publishes `Nivara` and `Nivara.Extensions` to NuGet (`--skip-duplicate`).

### Step 9 — Confirm the publish

Watch the CD run under the repo's Actions tab. The workflow packs with the tag-derived
version (tag name minus the leading `v`) and pushes both the `.nupkg` and `.snupkg` for
each package.

## Manual publish (fallback)

The workflow also accepts a `workflow_dispatch` with a `version` input if a tag-based run
is not possible. Select **Actions → CD → Run workflow**, enter the version, and run. All
other steps (build, test, pack, push) are identical.

## Conventions

- Release commits follow the pattern `chore: prepare vX.Y.Z release (version bump, release
  notes, changelog)` (see the v1.1.0 release commit for the reference shape).
- The NuGet `PackageReleaseNotes` should be a self-contained one-paragraph summary — it is
  rendered on nuget.org and is independent of `CHANGELOG.md`.
- Publish only from `main`; tags pushed from feature branches will trigger the workflow but
  should be avoided so NuGet history matches released source.
- Benchmark output JSON should be committed alongside the release prep (or at minimum
  referenced in the PR description) so reviewers can verify measurements were actually
  taken, not carried forward from a different machine or codebase version.
