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

4. **Refresh benchmark readings.** Three documents carry machine-specific numbers that go
   stale as hardware and the codebase move on. Re-measure them on the machine running the
   release and update all three (record the measurement date + machine in each):
   - `tests/Nivara.PerformanceTests/README.md` — the canonical point-A table. Run the
     harness (`dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json <path> --runs 3`,
     medians of 3 child processes per its Methodology section) and replace the Results table.
     No A/B against the old table: readings from other machines are only order-of-magnitude
     comparable, so the new numbers become the rolling point A.
   - `samples/NivaraInference/README.md` — the PyTorch vs Nivara inference table. Download
     the models (`hf download ...`, see the README's Quick start), then measure both sides
     in the same session on the same machine: Nivara via `dotnet run -c Release -- <model> benchmark`,
     PyTorch via `Python/<model>.py` (vision) / `Python/*_benchmark.py` (MiniLM, DistilBERT).
     Update the table, the Slowdown column, and the prose.
   - `samples/NivaraFineTuning/README.md` — the PyTorch vs Nivara fine-tuning table.
     `benchmark_timing.cmd` runs both sides and tees to `benchmark_results.txt` (Nivara in
     Release with Server GC + Tiered PGO; PyTorch with `torch_threads = nproc`). Update the
     table, the Slowdown column, and the extrapolated full-run estimates.

5. **Commit the release prep** (docs + version bump) on a branch, open a PR, and merge it
   to `main`:

   ```powershell
   git checkout -b khurram/v120
   # ... edits from steps 1-4 ...
   git add .
   git commit -m "chore: prepare vX.Y.Z release (version bump, release notes, changelog)"
   git push -u origin khurram/v120
   gh pr create --repo khurram-uworx/Nivara --base main --title "chore: prepare vX.Y.Z release" --body-file pr-body.md
   ```

6. **Verify the build before tagging.** The CD workflow builds and runs the full test suite,
   so a broken build or failing tests will fail the release. Locally, run
   `dotnet build Nivara.slnx` (and `dotnet test` when practical) before pushing the tag.

7. **Tag and push.** Annotated tags are preferred:

   ```powershell
   git tag -a vX.Y.Z -m "Nivara vX.Y.Z"
   git push origin vX.Y.Z
   ```

   Pushing a `v*` tag to `main` triggers the CD workflow, which builds, tests, packs, and
   publishes `Nivara` and `Nivara.Extensions` to NuGet (`--skip-duplicate`).

8. **Confirm the publish.** Watch the CD run under the repo's Actions tab. The workflow
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
