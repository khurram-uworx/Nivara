# Releasing Nivara

This document describes how to cut a new release of the `Nivara` and `Nivara.Extensions`
NuGet packages. Publishing is automated by the tag-triggered CD workflow
(`.github/workflows/cd.yml`), so the main work is updating docs, bumping versions, and
creating the tag.

## Prerequisites

- Push access to `khurram-uworx/Nivara`.
- The `NUGET_API_KEY` secret must be set in the GitHub repository settings so the CD
  workflow can publish to NuGet.

## Release checklist

1. **Draft the changelog.** In `CHANGELOG.md`, replace the `## Unreleased` heading (if any)
   with `## [X.Y.Z] - YYYY-MM-DD` and make sure every user-visible change since the last
   tag is captured — breaking changes first, then features/fixes. Use the previous release
   entry as a style template.

2. **Update public docs.** Review these files for anything that changed since the last tag
   and refresh as needed:
   - `README.md`
   - `GETTING-STARTED.md`
   - `ARCHITECTURE.md`
   - `docs/AUTODIFF.md`
   - `EXAMPLES.md`
   - `samples/README.md`
   - `AGENTS.md`
   - The `docs/adr/` decision records, when a change matches an ADR decision.

3. **Bump package versions.** In both `src/Nivara/Nivara.csproj` and
   `src/Nivara.Extensions/Nivara.Extensions.csproj`:
   - Set `<PackageVersion>` to the new version.
   - Update `<PackageReleaseNotes>` to summarize the new release (the previous release's
     notes are not accumulated).

   The csproj `PackageVersion` is the source of truth for the repo state; the CD workflow
   overrides it at pack time via `-p:PackageVersion=$(VERSION)`, but keeping it in sync
   means a plain local `dotnet pack` also produces the right version.

4. **Capture machine configuration.** Before measuring, record the machine config including battery/power status once
   and reuse it across all three perf documents. Include it in each document's recording
   line using the format established in `samples/NivaraIncident/README.md`:
   ```
   *Recorded YYYY-MM-DD — <CPU model>, .NET <version>, <extra context>.*
   ```
   For example: `*Recorded 2026-08-19 — Intel Core Ultra 7 255H Power Plugged In, .NET 10.0.11, 10M records, scenario A.*`
   Record at minimum: CPU model and core count, .NET SDK/runtime version, and any
   environment-specific notes (OS, RAM, whether the machine was idle or busy, GPU present
   but unused, etc.).

5. **Refresh benchmark readings.** Three documents carry machine-specific numbers that go
   stale as hardware and the codebase move on. Re-measure them on the machine running the
   release and update all three. Each document keeps a rolling two-column history: the
   previous reading moves to a **Prev** column and the fresh reading becomes **Current**,
   with a ratio/percentage column showing the delta. When there is no prior reading on the
   same machine, Current is the first entry and Prev/ratio cells are left blank.

   - `tests/Nivara.PerformanceTests/README.md` — the canonical performance table. Run the
     harness (`dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json <path> --runs 3`,
     medians of 3 child processes per its Methodology section). In the Results table:
     - Shift the existing **ops/s** column to a new **Prev** column.
     - Place the fresh measurements in the **Current** column.
     - Add a **Ratio** column showing `Current / Prev` (e.g. `1.05×` or `0.92×`).
     - Add a **Δ%** column showing `((Current − Prev) / Prev) × 100` (e.g. `+5.0%` or `−8.0%`).
     - Keep the **B/op** and **gen0/op** columns as-is (these are stability indicators, not
       throughput metrics, so a ratio column is not useful for them).
     - If a scenario is new (no prior reading), leave Prev/Ratio/Δ% blank for that row.
     - Update the machine line and recording date. Remove the "point A" policy language and
       replace it with "rolling Prev/Current history" — the table now carries its own
       comparison context.

   - `samples/NivaraIncident/README.md` — the Nivara analysis performance table **and** the
     Nivara vs Polars comparison table. For the Nivara-only table, shift the existing timing
     column to **Prev** and place new timings in **Current**, with **Ratio** and **Δ%** columns
     (same formula as above). For the Polars comparison table, keep the current structure
     (Nivara / Polars / Ratio) and only update the numbers and recording line; add a Prev
     column only if there is a prior same-machine measurement to compare against.
     Update the machine line and recording date in both tables.

   - `samples/NivaraInference/README.md` — the PyTorch vs Nivara inference table. Shift
     the existing Nivara and PyTorch timing columns to **Prev (PyTorch)** and **Prev (Nivara)**
     and place fresh measurements in **Current (PyTorch)** and **Current (Nivara)**, with a
     **Prev Slowdown** column (old ratio) and a **Current Slowdown** column (new ratio). If
     the table format is too narrow for four timing columns, keep the single PyTorch/Nivara
     pair and add a separate **Δ%** column for Nivara only (PyTorch times are stable baseline
     on the same machine). Update the machine line, recording date, and the prose that
     references the ratios.

   - `samples/NivaraFineTuning/README.md` — the PyTorch vs Nivara fine-tuning table.
     `benchmark_timing.cmd` runs both sides and tees to `benchmark_results.txt` (Nivara in
     Release with Server GC + Tiered PGO; PyTorch with `torch_threads = nproc`). Apply the
     same Prev/Current/Ratio/Δ% pattern for the Nivara timings. Update the table, the
     Slowdown column, and the extrapolated full-run estimates.

6. **Commit the release prep** (docs + version bump) on a branch, open a PR, and merge it
   to `main`:

   ```powershell
   git checkout -b khurram/v120
   # ... edits from steps 1-5 ...
   git add .
   git commit -m "chore: prepare vX.Y.Z release (version bump, release notes, changelog)"
   git push -u origin khurram/v120
   gh pr create --repo khurram-uworx/Nivara --base main --title "chore: prepare vX.Y.Z release" --body-file pr-body.md
   ```

7. **Verify the build before tagging.** The CD workflow builds and runs the full test suite,
   so a broken build or failing tests will fail the release. Locally, run
   `dotnet build Nivara.slnx` (and `dotnet test` when practical) before pushing the tag.

8. **Tag and push.** Annotated tags are preferred:

   ```powershell
   git tag -a vX.Y.Z -m "Nivara vX.Y.Z"
   git push origin vX.Y.Z
   ```

   Pushing a `v*` tag to `main` triggers the CD workflow, which builds, tests, packs, and
   publishes `Nivara` and `Nivara.Extensions` to NuGet (`--skip-duplicate`).

9. **Confirm the publish.** Watch the CD run under the repo's Actions tab. The workflow
   packs with the tag-derived version (tag name minus the leading `v`) and pushes both the
   `.nupkg` and `.snupkg` for each package.

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
