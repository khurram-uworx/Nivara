# Plan: issue #1064 — mergeable accumulator for streaming/partitioned metrics (ADR)

## Problem

Metrics require two fully materialised `IColumn` of equal length. Classification is a
monoid (`label -> index` + `long[k*k]` matrix, key-union + cell-wise add), but
`AggregationFunction` has no accumulate/merge contract. Streaming machinery
(`QueryFrame.AsStream`, `StreamingExecutionStrategy`, `ParallelExecutionStrategy`)
exists but cannot feed metrics incrementally.

## Scope (this branch)

Land ADR-009 + AGENTS.md index entry only. No accumulator implementation (separate issue).

## Proposed changes

1. Create `docs/adr/009-metrics-mergeable-accumulator.md` (ADR-009) answering 1-4:
   - Q1: public, stateful accumulators (`ConfusionMatrixAccumulator<TLabel>`,
     `RegressionAccumulator`) with `Add`, `Merge`, `ToFrame()`; not an internal-only
     `...Frame(IAsyncEnumerable<...>)` convenience.
   - Q2: reduction in ascending chunk/partition index order; completion order may not reduce.
   - Q3: refuse-to-stream-rejected; accept mergeable one-pass regression with stated tolerance
     (`1 ulp`, no worse than `1e-12` relative or `1e-9` absolute; SS_res exact-ish from
     non-negative sums).
   - Q4: one diagnostic per finalisation; none per `Add`.
2. Add `- **ADR-009** (...)` line to AGENTS.md ADR list.
3. Update `DocsCoveragePins` if document counts move (new ADR under docs/adr).
4. Possibly update `docs/METRICS.md` (leave out unless needed; keep change minimal).

## Blast radius

- `docs/adr/009-*.md` new file; scanned by snippet/citation/reference gates.
- `AGENTS.md` ADR index (AdrIndexTests).
- `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` (document/reference counts).
- No source code changes; no public API changes.

## Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test ... Docs` namespace (AdrIndexTests, DocsCoverageTests, snippet gate)
- AdrIndexTests both directions.

## Planned commits

1. `docs: plan issue #1064 in TODO.md`
2. `docs: add ADR-009 mergeable metrics accumulator`
3. `docs: register ADR-009 in AGENTS index`
4. `test: adjust DocsCoveragePins for new ADR`
5. `docs: remove TODO.md — plan executed`

## GitHub issues log

- (none yet; capture deferred work during execution)
