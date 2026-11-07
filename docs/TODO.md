# Plan: Issue #988 — release benchmark artefact policy (release/ folder)

## Problem
Release benchmark numbers currently live only in ephemeral `%TEMP%` files.
Reviewers cannot verify tables were refreshed. Need a permanent, release-wise
artefact location plus documented policy, retention, and PR checklist step.

## Decisions (human-approved)
- Permanent evidence location: top-level `release/<version>/`.
- Policy recorded in `RELEASING.md` only (no ADR).
- `%TEMP%` evidence left in place during active prep and cleaned manually later.

## Proposed changes
1. Create `release/` tree: `.gitkeep`, `README.md` (policy/layout/naming/recording line/retention).
2. `.gitignore`: add exception for tracked `release/` despite `[Rr]elease/` build-output rule.
3. `RELEASING.md`: update Step 5, Step 6 checklist, conventions, release.cmd example, add "Benchmark artefact policy" subsection.
4. Three sample regen scripts: update `%TEMP%` example path to `release/vX.Y.Z/...`.
5. New `.github/PULL_REQUEST_TEMPLATE/release.md` checklist including committed evidence paths.

## Blast radius
- No src code; mostly docs/ignore/template.
- DocsCoverage tests likely unaffected (release/README.md and .github templates outside scan scope).
- `.gitignore` exception must be verified with `git check-ignore`.

## Verification
- `git check-ignore -v release/vX.Y.Z/foo.json` returns nothing.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Review `git diff` and targeted doc gate if needed.

## Planned commits
1. docs: plan #988 in TODO.md
2. chore(release): create release/ folder and artifact policy README
3. chore(gitignore): track release/ evidence despite Release/ build output
4. docs(releasing): document benchmark artifact policy and update flow
5. docs(samples): point regen helper examples at release/
6. chore(pr-template): add release PR checklist
7. docs: remove TODO.md — plan executed

## GitHub issues log
- (none new yet)
