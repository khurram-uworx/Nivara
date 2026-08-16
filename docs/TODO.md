# Phase 4 review follow-ups

**Branch:** `khurram/phase4-review` · **Base:** `main` · **Status:** executing

## Problem

Phase 4 (Async-First Streaming, `docs/PHASE4.md`) is delivered and CI is green, but the
review surfaced defects and acceptance-criteria gaps. The AC2 perf scenario confirmed a
real cancellation race (issue #280) at runtime.

## Work items

- [x] Create GitHub issues (high/medium/low + backlog) — see issues log below.
- [x] Add AC2 mid-stream cancellation scenario to `tests/Nivara.PerformanceTests`
      (Program.cs + README scenario table).
- [x] Run the harness — cancellation race confirmed (see #280 for the stack trace).
- [ ] Update README with the measured evidence and known-failing note.
- [ ] Delete `docs/PHASE4.md` (plan superseded by STREAMING.md, CHANGELOG, and the issues).
- [ ] Review the branch from a distance, then offer push + PR.

## Verification

- `dotnet build tests/Nivara.PerformanceTests -c Release` — clean.
- `dotnet run --project tests/Nivara.PerformanceTests -c Release` — AC2 scenario currently
  FAILS (expected) until the cancellation race (#280) is fixed in product code; all other
  26 scenarios pass. This branch only adds the probe + docs; the fix is tracked, not made here.

## Blast radius

- `tests/Nivara.PerformanceTests/Program.cs` — adds one scenario + two internal helper
  types (`PerfChunkedSource`, `PerfStreamableOperation`); no public API change.
- `tests/Nivara.PerformanceTests/README.md` — scenario table row + known-failing note.
- `docs/PHASE4.md` — deleted (documentation only; decisions live in STREAMING.md/CHANGELOG).
- No `src/Nivara` product-code changes on this branch — the team fixes #279/#280/#278.

## GitHub issues log

- [x] #279 — high: `AsQueryFrame()` disposal aliasing destroys source frame columns (created while reviewing Phase 4)
- [x] #280 — medium: streaming cancellation race masks OCE with QueryExecutionException; producer unobserved; in-flight frames leak
- [x] #278 — medium/doc: STREAMING.md documents chunk-frame disposal the pipeline never performs
- [x] #281 — low/backlog: AC3 memory-budget/backpressure verification gap; StreamingBufferManager not wired
