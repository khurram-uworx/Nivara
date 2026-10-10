# #504 — IngestionTests still carries a duplicate dataset generator with hard-coded counts

Issue: https://github.com/khurram-uworx/Nivara/issues/504 (labels: `code-quality`, `sample`)
Branch: `khurram/504`

## Problem

`tests/Nivara.Tests/Incident/IngestionTests.cs` carried a private `GenerateSmallDataset`
(lines 118–268): a reimplementation of `samples/Nivara.Samples/Incident/DatasetGenerator.cs`,
alongside the copy removed from `AnalysisTests.cs` in #499 and a fourth in
`tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs` (`GenerateSmallWithRowGroupSize`).

The issue predicted that swapping the generator re-baselines two hard-coded counts
(line 34 `RowCount == TotalRows`, line 73 `chunkCount == TotalRows / RowGroupSize`).

## Verified facts (measured on the tree, not taken from the issue)

- **The counts do not move, and the issue's premise is wrong.** `GenerateFromRecordCount`
  computes `requestsPerMinute = (int)(totalRecords / 30.0) = 333` and fills only
  `333 x 30 = 9,990` request rows — but it sizes the arrays to `10,000` and
  `NivaraFrame.Create` uses the whole array, so `requests.parquet` carries **10,000 rows**, the
  last 10 being trailing defaults (`StatusCode 0`, null `Service`/`Region`). Confirmed by
  running the fixture: `Expected: 9990, But was: 10000`. The same claim in the #499 commit body
  (`ffefb82e`) is equally wrong; 9,990 is the *populated* row count, not the file's row count.
  Filed as #563.
- `GenerateFromRecordCount` hard-coded `rowGroupSize: 10_000` (`DatasetGenerator.cs:235`).
- Parquet chunking is **row-group aligned and ignores the advisory `chunkSize`**:
  `ParquetLazySource.ReadChunkAsync(chunkIndex, …)` maps index → row group
  (`src/Nivara.Extensions/IO/ParquetDataSource.cs:219,253`). So **chunks == row groups**.
- Therefore, with 10,000 requested records and 100-row groups: 100 row groups → **100 chunks**,
  which is what the old test-local generator produced too. The three streaming tests
  (`YieldsExpectedChunkCount`, `DisposesResources` with `Count == 3`, `CancellationStopsStream`
  with `cancelAfter = 3`) are unaffected — the earlier worry that a 1-row-group file would break
  them does not arise once the row-group size is passed through.
- `GenerateFromRecordCount` writes a superset of the local helper's files (it adds
  `dependencies.parquet`) and adds `affectedRegion` plus a per-event latency ramp.
- `IncidentSurfaceTests.Surface()` keys rows as `{Type}.{MethodName}` (line 246) built from
  `nameof` (line 273). An **optional parameter keeps the name**, so no overload is created and
  the surface/collision gates stay green — confirmed green.
- `GenerateFromRecordCount` has no XML doc, so the added parameter does not trip CS1573.

## Decision (resolved with the human before planning)

**Option C** — add an optional `rowGroupSize` to `DatasetGenerator.GenerateFromRecordCount`
(default `10_000`) and have `IngestionTests` request a 100-row layout. This preserves the
observed chunking shape (100 chunks) without inflating the fixture.

## Grounding (G1)

- Optional/named arguments are source-compatible:
  [Named and Optional Arguments](https://learn.microsoft.com/dotnet/csharp/programming-guide/classes-and-structs/named-and-optional-arguments).
- [IDE0005](https://learn.microsoft.com/dotnet/fundamentals/code-analysis/style-rules/ide0005)
  only reports on build when XML documentation is enabled; no `*.props`/`*.targets`/`*.csproj`
  sets `GenerateDocumentationFile`. Dropping the unused `using Nivara.IO;` is therefore hygiene,
  **not** a build requirement. (Done anyway.)
- Blast radius of the signature change: callers are `DatasetGenerator.Generate`, `AnalysisTests`,
  `AnalysisResourceTests`, `StreamixScenarioTests`, `IncidentSurfaceTests` and
  `samples/NivaraIncident/NivaraIncident.Cli/Program.cs:73` (the code-memory index misses the
  last one; grep found it). All remain source-compatible.

## Blast radius

- `DatasetGenerator.GenerateFromRecordCount` — public member of the `Nivara.Samples` assembly.
  Signature changes, name does not; no binary-compatibility requirement for a sample assembly.
- `IngestionTests` — fixture-only; its five tests are the only consumers.
- Not affected: `AnalysisTests`, `StreamixScenarioTests`, `AnalysisResourceTests`,
  `IncidentSurfaceTests`, `PolarsIncidentCrossValidationTests`, `NivaraIncident.Cli`.

## Verification

1. `dotnet build Nivara.slnx -c Release` after each change — clean.
2. Targeted fixture run (pre-approved under AGENTS.md): `IngestionTests` 5/5 green;
   `IncidentSurfaceTests` 3/3 green (surface registry + no-overload gate).
3. Runtime delta, five rounds per side, **interleaved with the order alternated** (AGENTS.md:
   separate best-of-N blocks are what made a prior gate flaky):
   Release, `--no-build`, filtered to `IngestionTests`, every run exit 0 —
   before 5.575 s median `[5.509, 5.942]`, after 5.905 s median `[5.393, 6.066]`, ratio 1.059x;
   test-host-reported fixture duration 567 ms → 594 ms median. No timing assertion added.

## Commits

1. `docs: plan #504 IngestionTests generator swap in TODO.md` (`a21cf6a2`)
2. `feat(incident): expose rowGroupSize on GenerateFromRecordCount` (`db2af4a3`)
3. `test(incident): generate IngestionTests fixtures with DatasetGenerator` (`8532bf01`)
4. `docs: correct the #504 plan — the counts do not move` (`654947be`)
5. `docs: record the #504 follow-up issues` (this commit)
6. `docs: remove TODO.md — plan executed`

Deferred with the human's agreement (out of #504's scope): folding
`IncidentLabBenchmark.GenerateSmallWithRowGroupSize` into `GenerateFromRecordCount`
(`GenerateFromRecordCount(dir, "A", 5_000, rowGroupSize)` → 4,980 populated + defaults, 5 groups)
is tracked as #564, because it changes what that benchmark writes and so needs its own
measurement rather than riding on a test re-baseline.

## GitHub issues log

- [x] #563 — DatasetGenerator emits trailing default-valued rows and its generated count is
      misreported (created while working on #504; the 9,990 claim in #504 and in the #499 commit
      body is wrong).
- [x] #564 — IncidentLabBenchmark still carries a fourth copy of the dataset generator
      (created while working on #504; deferred by agreement rather than bundled in).
- [x] Comment posted on #504 correcting the 9,990 premise
      (https://github.com/khurram-uworx/Nivara/issues/504#issuecomment-6101484155).
