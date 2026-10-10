# Plan — #563: DatasetGenerator emits trailing default-valued rows

Branch: `khurram/563` (off `khurram/479`).

> As each task executes, if you find deferred work or a concern (known limitation,
> follow-up, refactor) that is outside this plan, create a tracked issue immediately
> (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the
> **GitHub issues log** below. Do not rely on memory and do not wait until the plan
> finishes — compaction during execution loses items.

## Problem

`DatasetGenerator.GenerateFromRecordCount(datasetPath, scenarioId, totalRecords, rowGroupSize)`
in `samples/Nivara.Samples/Incident/DatasetGenerator.cs` sizes every request array to
`totalRecords` but only fills `floor(totalRecords / 30) * 30` slots: the per-minute loop
stops when `requestsPerMinute` is exhausted, so the trailing `totalRecords % 30` slots keep
their default values (`Timestamp = 0`, null `Service`/`Endpoint`/`Region`/`TraceId`,
`DurationMs = 0`, `StatusCode = 0`, `IsRetry = false`).

Both writers (`ToParquet` and `WriteCsv`) use the full arrays, so both formats carry the
default rows. For `totalRecords = 10_000` the file has **10,000 rows of which the last 10
are defaults**; at 10M (scale 1) the last 10 are defaults. The method name and the CLI help
(`samples/NivaraIncident/README.md`: `--records` = "Exact record count") read as though every
written row is a generated one.

The incident analyses are unaffected — they filter `Timestamp >= incidentStart` before
grouping, and the default rows carry `Timestamp = 0` — but the trap is live for any
whole-file reader (a group-by on `Region` yields a null group; a status-code distribution
counts extra zeros).

## Root cause

`DatasetGenerator.cs:46-73`:

- `requestsPerMinute = (int)(totalRequests / durationMinutes)` is a floor;
- arrays are allocated at `totalRequests`;
- the fill loop bound is `r < requestsPerMinute && requestIdx < totalRequests`;
- both writers use the full arrays.

## Decision (chosen with the human)

**Option 2 — keep the exact requested row count and fill the tail with valid rows**,
distributed evenly across minutes. Rationale: it makes the method name and the
`--records` "Exact record count" promise true, removes the misleading default rows, and
requires **no re-baselining** of the exact-count assertions (`IngestionTests`,
`IncidentLabBenchmark.TestRowCount`) that commits `8532bf01` / `654947be` just settled.

## Proposed changes

### 1. `samples/Nivara.Samples/Incident/DatasetGenerator.cs` (core fix)

```csharp
// line 49 — carry the per-minute remainder
int requestsPerMinute = (int)(totalRequests / durationMinutes);
int remainderRequests = totalRequests - requestsPerMinute * (int)durationMinutes;
```

```csharp
// line 73 — distribute the remainder across the first `remainderRequests` minutes;
// drop the `requestIdx < totalRequests` guard so a future miscalculation throws loudly
// instead of silently defaulting again.
int minuteRequests = requestsPerMinute + (minute < remainderRequests ? 1 : 0);
for (int r = 0; r < minuteRequests; r++, requestIdx++)
```

Identity to hold for every input: `30 * requestsPerMinute + remainderRequests == totalRequests`
(also true when `totalRecords < 30`: `requestsPerMinute == 0`, first `totalRecords` minutes get
one row each). `Generate(path, id, scale)` delegates here, so the scale path is fixed too.

### 2. New test — `tests/Nivara.Tests/Incident/DatasetGeneratorTests.cs`

Executable spec, small record counts only (never 10M), plain `[Test]` methods:

- multiple of 30 (e.g. `9_000`): `RowCount == 9_000`, 300 rows in each of 30 minutes;
- non-multiple (e.g. `10_003`): `RowCount == 10_003`, first 13 minutes have 334, remaining 17 have 333;
- smaller than 30 (e.g. `20`): `RowCount == 20`, one row per minute for 20 minutes;
- no defaults (non-multiple case): every row has non-null `Service`/`Region`/`TraceId`,
  `Timestamp != 0`, `DurationMs > 0`, `StatusCode ∈ {200, 429, 500, 502, 503}`.

### 3. Stale-comment / harness updates

- `tests/Nivara.Tests/Incident/IngestionTests.cs:11-14` — the "last 10 are trailing defaults"
  comment becomes false; rewrite. `TotalRows = 10_000` and `ExpectedChunks = 100` are unchanged.
- `tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs` `TestFieldRanges` — remove the
  `StatusCode == 0` skip and the `defaultValueCount` reporting; with no defaults, `sc == 0`
  must fail the range check.
- `IncidentLabBenchmark.GenerateSmallWithRowGroupSize` — a private duplicate of the same defect
  (5000 → 20 default rows). It only feeds the row-group test, which ignores values, so it is not
  live; fix it the same way so the defect does not survive in a second copy.

### 4. Tracking artifacts

- Issue **#504** body states the wrong row count (`= 9,990 rows`); correct it via
  `gh issue edit 504 --body-file <tmp>` (temp file per AGENTS.md — never inline `--body`).
- Commit `ffefb82e` (#499) body is immutable history; no edit possible. The correction is
  already carried by `8532bf01` and `654947be`.

### 5. Docs

- `samples/NivaraIncident/README.md:67` — `--records ... Exact record count` becomes true under
  this fix; no correction required. Optional one-line clarification that rows are fully populated.
- `CHANGELOG.md` — add a `#563` entry (watch the current branch's CHANGELOG state; stage the
  #563 hunk only).

## Blast radius

| File / symbol | Change | Depends on it |
|---|---|---|
| `DatasetGenerator.GenerateFromRecordCount` | fill loop distributes remainder | `Generate` (scale), CLI `generate`, all Incident fixtures, `IncidentLabBenchmark` |
| `DatasetGenerator.Generate` | none (delegates) | sample CLI |
| `tests/Nivara.Tests/Incident/*` | comment + new test file | CI |
| `tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs` | `FieldRanges` tighten; duplicate small generator fix | manual `--dataset-test` harness |
| `CHANGELOG.md` | new entry | release notes |

Generated data changes (RNG draw order shifts from the distributed remainder). No golden
fixtures depend on it: the Polars references in `gen_reference.py` are hand-authored arrays,
and every Incident test is property-based. Any on-disk dataset must be regenerated (the README
quick-start already implies this). Determinism remains a property: same seed → same bytes for a
given build. `TestDeterminism` compares two fresh runs of the same code.

## Verification

- `dotnet build Nivara.slnx`.
- Targeted NUnit (Release), allowed without asking:
  `dotnet test tests/Nivara.Tests -c Release --filter "FullyQualifiedName~Nivara.Tests.Incident"`
  — capture the test process's own exit status, not a filtered pipeline's.
- Manual harness (heavy; Release only, ask first): 
  `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --dataset-test`.
- No new timing assertions, so no `[Category("Performance")]` needed.

## Planned commits

1. `fix(samples): fill the full requested record count in DatasetGenerator` — `DatasetGenerator.cs`.
2. `test(incident): pin exact row count and absence of trailing defaults` — new
   `DatasetGeneratorTests.cs` + `IngestionTests` comment.
3. `chore(perf): tighten the incident field-range gate` — `IncidentLabBenchmark.cs`.
4. `docs: record #563 in CHANGELOG` — CHANGELOG only.

Then the `gh issue edit 504` body correction.

## Probe / harness lifecycle

No new probe or harness is required. The change is a correctness fix to a sample generator; it
is verified by the NUnit suite and the existing `--dataset-test` mode. No temp harness created.

## GitHub issues log

- [ ] #504 — IngestionTests duplicate generator / wrong row count; body to be corrected as part of #563.

## G1 grounding

- **microsoft-learn** — `Random(Int32)` is documented as generating "a reproducible sequence of
  pseudo-random numbers" (learn.microsoft.com/dotnet/api/system.random.-ctor). The fix preserves
  determinism as a property for a fixed seed; the tests compare two fresh runs, not fixed output,
  so a future .NET `Random` algorithm change cannot silently break the gate. No Tensor/Vector or
  new API surface is touched — the change is plain integer arithmetic and array fill.
- **code-memory** — `GenerateFromRecordCount` has 4 commits (last `db2af4a3` "expose rowGroupSize");
  `Generate` is its only in-file caller. No `GenerateSmallDataset` symbol remains (the #504 swap
  `8532bf01` landed), and `GenerateSmallWithRowGroupSize` is the single duplicate.
- **Blast radius (confirmed)** — external call sites are cross-project and not captured as symbol
  relationships, so confirmed by grep: CLI `samples/NivaraIncident/NivaraIncident.Cli/Program.cs:73`;
  tests `StreamixScenarioTests`, `IngestionTests`, `IncidentSurfaceTests`, `AnalysisTests`,
  `AnalysisResourceTests`; harness `IncidentLabBenchmark`. Matches the blast-radius table above.
- **Decisions / red flags** — none surfaced. Implementation proceeds as written.
