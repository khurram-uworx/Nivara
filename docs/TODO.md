# Plan: reconcile AISTACK review/roadmap status with current main (issue #1292)

## Problem

`docs/plan/AISTACK-REVIEW.md` and `docs/plan/AISTACK-ROADMAP.md` (revised 2026-11-03) status text no longer reflects `main`. Phases 1/2/4/5 and frame-level dataset ops have since landed, but the docs still state them as open/missing. Non-goals must remain explicit and remaining gaps listed as open follow-ups.

## Proposed changes

Documentation-only edits to the two plan files:

- `docs/plan/AISTACK-ROADMAP.md`: top status line, "Where we are today", Phase 1/2/3/4/5 status notes, §4 definition of done.
- `docs/plan/AISTACK-REVIEW.md`: header reassessment lines, scorecard table, §5 gaps, summary.

## Verification steps

- Re-read both files for consistency with non-goals and evidence paths.
- No build/test impact (plan docs excluded from scan).

## Planned commits

1. `docs: plan #1292 reconciliation in TODO.md`
2. `docs: reconcile AISTACK roadmap status with main`
3. `docs: reconcile AISTACK review scorecard with main`
4. `docs: remove TODO.md — plan executed`

## Blast radius

Plan docs only. No code, no tests, no gate pins.

## GitHub issues log

- (none created yet)
