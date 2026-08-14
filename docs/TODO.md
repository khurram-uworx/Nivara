# Plan: v1.3.0 release prep

Branch: `release/130` (created off `main` @ `10c370f`)
Target: v1.3.0, tagged on `main` after this PR merges.

## Problem

Cut the next release per `RELEASING.md`. Publishing is automated by the tag-triggered CD
workflow, so the work here is: draft the changelog, refresh package metadata, verify the
build, and prepare the release branch for merge + tag.

## Scope / proposed changes

1. **CHANGELOG.md** — replace `## [Unreleased]` with `## [1.3.0] - 2026-08-14`. The
   Unreleased section is already comprehensive (commits landed incrementally during the
   dev cycle); no content addition expected beyond the heading/date.
2. **src/Nivara/Nivara.csproj** — `<PackageVersion>1.2.0` → `1.3.0`; rewrite
   `<PackageReleaseNotes>` as a self-contained one-paragraph v1.3.0 summary.
3. **src/Nivara.Extensions/Nivara.Extensions.csproj** — same version bump + release-notes
   rewrite.
4. **Public docs review** — confirm README / GETTING-STARTED / ARCHITECTURE /
   docs/AUTODIFF.md / EXAMPLES.md / samples/README.md / AGENTS.md have no stale
   release-specific claims. Historical "since 1.2.0" references are fine; only update if a
   doc contradicts the current state.
5. **Verify build** — `dotnet build Nivara.slnx` before the prep is committed/pushed.
   `dotnet test` only with explicit human confirmation.
6. **Commit plan for this branch:**
   - `docs: plan v1.3.0 release prep in TODO.md`
   - `docs: draft v1.3.0 changelog` (CHANGELOG.md)
   - `chore: prepare v1.3.0 release (version bump, release notes, changelog)` (csproj
     version + release notes)
   - docs refresh (only if step 4 finds stale content)
   - `docs: remove TODO.md — plan executed`

## Blast radius

- `CHANGELOG.md`, `src/Nivara/Nivara.csproj`, `src/Nivara.Extensions/Nivara.Extensions.csproj`,
  `docs/TODO.md` — docs/metadata only. No source or test code changes; no API surface.
- Downstream: the NuGet `PackageReleaseNotes` text (rendered on nuget.org) and the version
  reported by local `dotnet pack`. The CD workflow overrides the version at pack time via
  `-p:PackageVersion=$(VERSION)`, so the bump mainly keeps local packs correct.
- Tests: none exercise the csproj metadata or changelog; no test changes expected.

## Verification

- `git diff` review of every commit.
- `dotnet build Nivara.slnx` (asked before running).
- `dotnet test` only if the human wants it (per AGENTS.md, requires explicit confirmation).

## GitHub issues log

- (none so far — this is a docs/metadata-only release prep)
