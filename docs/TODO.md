# Release v1.4.0 prep — `release/140` branch

## Problem

We need to cut the v1.4.0 release. The `## Unreleased` section in `CHANGELOG.md` is
substantial (streaming, windowing, expression engine, bug fixes, performance) and the
csproj `PackageVersion` is still at 1.3.0. Per `RELEASING.md`, we must update docs,
bump versions, refresh benchmarks, and get a PR to `main` so a `v1.4.0` tag can be pushed.

## Target version

**v1.4.0** — the Unreleased section is a minor-level bump (no breaking changes from the
public perspective since 1.3.0; the `Unreleased` section contains only Added/Changed/Fixed).

## Plan (steps 1–7 from RELEASING.md)

### Step 1: Draft the changelog
- In `CHANGELOG.md`, replace `## [Unreleased]` with `## [1.4.0] - 2026-08-21`
- Ensure every user-visible change since v1.3.0 is captured (the current Unreleased
  section already covers this; just update the heading)

### Step 2: Update public docs
Review and refresh if needed:
- `README.md`
- `GETTING-STARTED.md`
- `ARCHITECTURE.md`
- `docs/AUTODIFF.md`
- `EXAMPLES.md`
- `samples/README.md`
- `AGENTS.md`
- `docs/adr/` — check if any new ADR was added (ADR-004 for fused expression engine)

### Step 3: Bump package versions
- `src/Nivara/Nivara.csproj`: `<PackageVersion>1.4.0</PackageVersion>`, update `<PackageReleaseNotes>`
- `src/Nivara.Extensions/Nivara.Extensions.csproj`: `<PackageVersion>1.4.0</PackageVersion>`, update `<PackageReleaseNotes>`

### Step 4: Capture machine configuration
Record the machine config once and include it in all perf docs.
Current machine: .NET SDK 11.0.100-preview.7 (but targeting net10.0).
Need to detect CPU model, core count, and environment.

### Step 5: Refresh benchmark readings
Four documents to update with Prev/Current/Ratio/Δ% pattern:
1. `tests/Nivara.PerformanceTests/README.md` — run `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --json <path> --runs 3`
2. `samples/NivaraIncident/README.md` — Nivara-only table + Polars comparison table
3. `samples/NivaraInference/README.md` — PyTorch vs Nivara inference table
4. `samples/NivaraFineTuning/README.md` — PyTorch vs Nivara fine-tuning table

### Step 6: Commit release prep
Commit all doc + version bump changes with message:
`chore: prepare v1.4.0 release (version bump, release notes, changelog)`

### Step 7: Verify build
Run `dotnet build Nivara.slnx` and `dotnet test` to ensure nothing is broken before tagging.

## Deferred (manual, human-controlled)

- Push branch + open PR to `main`
- After PR merge: tag `v1.4.0` and push → triggers CD workflow

## Blast radius

- All changes are docs/config/changelog only (no code changes), so blast radius is zero
  for the source code. Benchmark re-measurement runs existing scenarios.

## GitHub issues log

- (none created — this is a release-prep branch, not a feature branch)
