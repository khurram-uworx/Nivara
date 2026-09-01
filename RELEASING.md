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
  `dotnet run --project samples/NivaraIncident/NivaraIncident.Cli -- generate --dataset ./samples/data/incident-lab --scenario A --scale 1`
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
> description and leave the existing numbers in place — do not fake or carry forward
> stale numbers as if they were fresh.

Each benchmark has a `## Release Benchmark` section in its README with exact commands,
prerequisites, and table-update instructions. Run them in order:

| # | Benchmark | README | Prerequisites |
|---|-----------|--------|---------------|
| 1 | Core perf harness | `tests/Nivara.PerformanceTests/README.md` | .NET SDK only |
| 2 | NivaraIncident | `samples/NivaraIncident/README.md` | Generated dataset (see Prerequisites) |
| 3 | NivaraInference | `samples/NivaraInference/README.md` | Python + PyTorch + model weights |
| 4 | NivaraFineTuning | `samples/NivaraFineTuning/README.md` | Python + PyTorch + model weights |

For each benchmark: **open the README → follow the `## Release Benchmark` section →
run the commands → update the tables with fresh Prev/Current numbers.** Do not skip
any benchmark — if one cannot be run, say so in the PR with the reason.

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
