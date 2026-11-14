# Releasing Nivara

This document describes how to cut a new release of the `Nivara` and `Nivara.Extensions`
NuGet packages. Publishing is automated by the tag-triggered CD workflow
(`.github/workflows/cd.yml`), so the main work is updating docs, bumping versions, and
creating the tag.

## Prerequisites

- Push access to `khurram-uworx/Nivara`.
- **NuGet publishing (OIDC / trusted publishing)** — the CD workflow (`.github/workflows/cd.yml`) publishes via GitHub Actions OIDC: it requests `id-token: write`, uses `NuGet/login@v1` to exchange the workflow OIDC token for a short-lived NuGet API key, and passes that temporary key to `dotnet nuget push`. No long-lived `NUGET_API_KEY` repository secret is required by the workflow. A matching trusted publishing / federated-credential policy must be configured for this repository/workflow in the NuGet account (nuget.org) so the token exchange can succeed.
- **.NET SDK** — the same version as the repo target framework (check the `.csproj` `TargetFramework`, e.g. `src/Nivara/Nivara.csproj`).
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
- `RELEASING.md` (if release/publish/tooling mechanics changed)
- The `docs/adr/` decision records, when a change matches an ADR decision.

For `README.md`, `AGENTS.md`, and `RELEASING.md`, explicitly verify claims about tooling, publishing, dependencies, and measured figures against the current source of truth (`.github/workflows/*`, `*.csproj`, and current test/build artifacts) so drift does not carry into the release.

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
| 3 | NivaraInference (CPU) | `samples/NivaraInference/README.md` | Python + PyTorch + model weights |
| 3a | NivaraInference (Silk/OpenCL) | `samples/NivaraInference/README.md` | OpenCL GPU + AC power + model weights |
| 3b | NivaraInference (BLAS) | `samples/NivaraInference/README.md` (OpenBLAS section) | OpenBLAS DLL (`lib/libopenblas.dll`) + model weights |
| 4 | NivaraFineTuning | `samples/NivaraFineTuning/README.md` | Python + PyTorch + model weights |

For each benchmark: **open the README → follow the `## Release Benchmark` section →
run the commands → update the tables with fresh Prev/Current numbers.** Do not skip
any benchmark — if one cannot be run, say so in the PR with the reason.

**Cross-machine rule:** if the Prev reading was on a different machine (or a
codebase version that changed the measurement), set **Prev to `—`** and note the
machine difference in the table footer. Do not compute ratios across machines —
a Nivara improvement plus a PyTorch improvement from more cores looks like a
regression when it is not.

**BLAS backend (required, non-skippable):** every model that supports `--blas` (minilm,
distilbert, modernbert, laya) must run the BLAS variant in the same session and record a
BLAS column (ms, same units as the CPU column) plus both `--threads` (row-blocks, default 1) and `--blas-threads` (OpenBLAS pool, default 1), following the
`## OpenBLAS acceleration` section of `samples/NivaraInference/README.md` (recommended
thread counts per model). BLAS runs are CPU-only and require `lib/libopenblas.dll`. Do not
skip this line item; if the DLL is unavailable, record the reason in the PR.

**Silk backend (required, non-skippable):** every sample that supports a `--silk`
variant (`minilm`, `distilbert`, `distilbert_sst`, `laya`, `modernbert` where
applicable, and the Qwen generation paths) must run **both** the CPU and the Silk
variant in the same session and record a Silk column (ops/s or ms/op, same units
as the CPU column) plus its ratio. CPU-only numbers are not release-grade once
the Silk backend ships as a first-class path (see issue #984). Silk runs are
AC-power-only and require the OpenCL device/driver to be recorded alongside the
machine line. Do not skip this line item; if an OpenCL device is unavailable,
record the reason in the PR.

**Capture output for review:** when a human runs these benchmarks on behalf of
an agent (or for later re-measurement), put every command in a temporary
`release.cmd` at the repository root that redirects each command's console
output to a working file, run it, and delete the script afterwards (it is
commit noise and must never be committed). Hand back the file paths
rather than pasting output into chat. Example lines inside `release.cmd`:
`dotnet run --project <proj> -c Release -- ... > release\\vX.Y.Z\\<name>-vX.Y.Z.txt 2>&1`
The core harness JSON file should be committed as canonical evidence under
`release/vX.Y.Z/nivara-perf-vX.Y.Z.json` (see "Benchmark artefact policy" below); a
`%TEMP%` copy may also be kept as staging during active prep and cleaned up
manually afterwards. The script may `cd` into sample
directories, but it must end back at the repository root.

**Regenerating benchmark tables safely (hidden-tag technique):** wrap each
results table in the README with HTML comments `<!--hidden-start-->` ... `<!--hidden-end-->`,
then regenerate only the rows between those markers from the captured
JSON/console output using a short Python script. Keep a `.bak` copy, review the
diff, then remove the markers before committing. Preserve the regenerator script
in the sample's `Python/` folder (e.g.
`samples/NivaraInference/Python/regen_perf_table.py --json <file> --readme <file>`,
and the matching copies under `samples/NivaraIncident/Python/` and
`samples/NivaraFineTuning/Python/`) so the next release can reuse it;
NivaraInference/Python is the canonical template and the others may diverge as
their table schemas evolve.

### Benchmark artefact policy (issue #988)

Permanent evidence lives under `release/<version>/`, one folder per release version:

- `release/vX.Y.Z/nivara-perf-vX.Y.Z.json` — core perf harness output.
- `release/vX.Y.Z/<sample>-vX.Y.Z.txt` (and optional `.csv`) — per-benchmark console captures.

Rules:
- Never overwrite or repurpose a prior release folder; history stays addressable by path.
- Committed files under `release/<version>/` are the canonical, permanent artefacts.
- `%TEMP%` files are staging during active prep; they may remain in place while the
  release is in progress and are cleaned up manually afterwards.
- PR attachments are optional review snapshots and may expire; cite the committed
  `release/<version>/` path as authoritative.
- File names always carry the version so a path is unambiguous.
- Machine/recording line format stays as specified in Step 4.

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
- Paths to the committed benchmark evidence under `release/<version>/` (canonical,
  permanent) and, optionally, the PR attachments used for review.
- Any benchmarks that could not be run and the reason (missing deps, etc.).
- The CD test results (3200+ tests passing).

The PR must show a green CI check before merging. Do not merge with `--admin`
until CI passes, and do not merge a docs-only PR without confirming `dotnet build
Nivara.slnx` and the targeted test suite are green locally.

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
publishes `Nivara` and `Nivara.Extensions` to NuGet using OIDC trusted publishing
(`NuGet/login@v1` → short-lived API key, `--skip-duplicate`).

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
- Benchmark output should be committed alongside the release prep under the
  `release/<version>/` folder (canonical, permanent artefact path) so reviewers can
  verify measurements were actually taken, not carried forward from a different
  machine or codebase version. PR attachments are an optional review snapshot.
