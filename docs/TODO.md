# #504 — IngestionTests still carries a duplicate dataset generator with hard-coded counts

Issue: https://github.com/khurram-uworx/Nivara/issues/504 (labels: `code-quality`, `sample`)
Branch: `khurram/504`

## Problem

`tests/Nivara.Tests/Incident/IngestionTests.cs` carries a private `GenerateSmallDataset`
(lines 118–268): a third reimplementation of
`samples/Nivara.Samples/Incident/DatasetGenerator.cs`, alongside the copy that used to sit in
`AnalysisTests.cs` and was removed in #499, and a fourth in
`tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs` (`GenerateSmallWithRowGroupSize`).

Unlike `AnalysisTests`, `IngestionTests` asserts **hard-coded absolute counts** against its own
generator (line 34 `RowCount == TotalRows`, line 73 `chunkCount == TotalRows / RowGroupSize`), so
the swap is a re-baseline rather than a straight swap.

## Verified facts (from the current tree, not the issue text)

- `GenerateFromRecordCount(dir, "A", 10_000)` computes
  `requestsPerMinute = (int)(10_000 / 30.0) = 333`, runs 30 minutes and is bounded by
  `requestIdx < totalRequests`, so it emits **9,990 rows**, not 10,000. The issue is right.
- The generator hard-codes `rowGroupSize: 10_000` (`DatasetGenerator.cs:235`), so with 10,000
  records the request file is a **single row group**.
- Parquet chunking is **row-group aligned and ignores the advisory `chunkSize`**:
  `ParquetLazySource.ReadChunkAsync(chunkIndex, …)` maps index → row group
  (`src/Nivara.Extensions/IO/ParquetDataSource.cs:219,253`) and `IQuerySource.ToAsyncEnumerable`
  walks that index (`src/Nivara/Query/IQueryInterfaces.cs:47-63`). So **chunks == row groups**.
- Therefore a literal swap (keep 10,000 records) yields 9,990 rows in one row group = **one chunk**,
  which breaks **three** streaming tests, not the two the issue names:
  - `StreamChunks_YieldsExpectedChunkCount` (line 62, expects 100)
  - `StreamChunks_DisposesResources` (line 76, breaks at 3 and expects `Count == 3`)
  - `StreamChunks_CancellationStopsStream` (line 93, `cancelAfter = 3`, expects 3)
- `GenerateFromRecordCount` writes a superset of the local helper's files (it adds
  `dependencies.parquet`), so the file-level swap is safe.
- `IncidentSurfaceTests.Surface()` keys rows as `{Type}.{MethodName}` (line 246) and `Key()` is
  built from `nameof` (line 273). An **optional parameter keeps the name**, so no overload is
  created and the surface/collision gates stay green.
- `GenerateFromRecordCount` has no XML doc, so an added parameter does not trip CS1573 under
  `TreatWarningsAsErrors` + `EnforceCodeStyleInBuild` (`Directory.Build.props`).

## Decision (resolved with the human before planning)

**Option C** — add an optional `rowGroupSize` to `DatasetGenerator.GenerateFromRecordCount`
(default `10_000`) and have `IngestionTests` request a 100-row layout. This keeps the observable
chunking shape (100 chunks) so the three streaming tests keep their meaning, without inflating the
fixture to the 30,000 records option B would need.

## Why the chunk assertion must change

With 9,990 rows and 100-row groups: 99 groups of 100 + one group of 90 = **100 row groups →
100 chunks**. The current expression `TotalRows / RowGroupSize` passes only by coincidence of the
old generator (10,000 / 100 exactly). The honest baseline is `ceil(9,990 / 100) = 100`; plain
integer division would give the **wrong** 99. This is exactly the "re-derive from the row-group
boundaries rather than divide" point in the issue.

## Proposed changes

### Commit 1 — `feat(incident): expose rowGroupSize on GenerateFromRecordCount`

`samples/Nivara.Samples/Incident/DatasetGenerator.cs`

- line 37:
  `public static void GenerateFromRecordCount(string datasetPath, string scenarioId, long totalRecords, int rowGroupSize = 10_000)`
- line 235: `var parquetOptions = ParquetWriteOptions.Default.With(rowGroupSize: rowGroupSize);`

Green on its own: the default preserves every existing caller (`Generate`, `AnalysisTests`,
`StreamixScenarioTests`, `IncidentSurfaceTests`, `AnalysisResourceTests`, `NivaraIncident.Cli`),
and the surface gate sees the same member name.

### Commit 2 — `test(incident): generate IngestionTests fixtures with DatasetGenerator (re-baseline)`

`tests/Nivara.Tests/Incident/IngestionTests.cs`

- Constants reworked so the moved baseline is visible, not silent:

  ```csharp
  const int RequestedRecords = 10_000;
  // GenerateFromRecordCount emits (totalRecords / 30) rows per minute over 30 minutes, so the
  // integer truncation drops the remainder: 10,000 records -> 333 x 30 = 9,990 rows.
  const int TotalRows = RequestedRecords / 30 * 30;
  const int RowGroupSize = 100;
  // Parquet chunks align to native row groups (the chunkSize argument is advisory), so the chunk
  // count is the row-group count. 9,990 / 100 truncates to 99; the real count is 100.
  const int ExpectedChunks = (TotalRows + RowGroupSize - 1) / RowGroupSize;
  ```

- `OneTimeSetUp` (line 19) →
  `DatasetGenerator.GenerateFromRecordCount(tempDir, "A", RequestedRecords, RowGroupSize)`
- line 34 → `Is.EqualTo(TotalRows)` (now evaluates to 9,990)
- `StreamChunks_YieldsExpectedChunkCount` → assert `chunkCount == ExpectedChunks` **and**
  `sum(chunk.RowCount) == TotalRows`, so the row-group partition is pinned and not just counted
- delete the private helper (lines 118–268)
- drop the now-unused `using Nivara.IO;` (required by `EnforceCodeStyleInBuild` +
  `TreatWarningsAsErrors`)

### Commit 3 (optional, separate reason) — fold the fourth generator

`tests/Nivara.PerformanceTests/IncidentLabBenchmark.cs`: delete
`GenerateSmallWithRowGroupSize` (line 207) and call
`DatasetGenerator.GenerateFromRecordCount(dir, "A", 5_000, rowGroupSize)` at line 150
(5,000 records → 4,980 rows → 5 groups of 1,000 > 1 ✓). Out of #504's stated scope; only with the
human's say-so, otherwise filed as a follow-up issue.

## Blast radius

- `DatasetGenerator.GenerateFromRecordCount` — public member of the `Nivara.Samples` assembly.
  Adding an optional parameter changes its signature but not its name; source-compatible for all
  six existing call sites and for the `nameof`-based surface registry. No binary-compatibility
  requirement (sample assembly).
- `IngestionTests` — fixture-only change; its five tests are the consumers of the regenerated
  fixture. No other file calls the private helper (confirmed via code-memory call graph: the only
  caller is `IngestionTests.OneTimeSetUp`).
- Not affected: `AnalysisTests`, `StreamixScenarioTests`, `AnalysisResourceTests`,
  `IncidentSurfaceTests`, `PolarsIncidentCrossValidationTests`, `NivaraIncident.Cli`.
- Test files that cover the touched behaviour: `IngestionTests` (fixture lifecycle),
  `IncidentSurfaceTests` (sample public surface), `ParquetStreamingTests` and `AsyncStreamingTests`
  (row-group-aligned chunking, independent coverage).

## Verification steps

1. `dotnet build Nivara.slnx -c Release` after each commit.
2. Targeted fixture run (pre-approved under AGENTS.md): 
   `dotnet test -c Release --no-build --filter "FullyQualifiedName~Nivara.Tests.Incident.IngestionTests"`.
3. Runtime delta, mirroring #499's `ffefb82e`: 5 timed runs per side of the filtered
   `IngestionTests` fixture in Release, on `--no-build`, capturing `$LASTEXITCODE` of the test
   process itself (not of a pipeline); report medians and ranges in commit 2's body. No timing
   assertion is added.
4. Confirm `IncidentSurfaceTests` still passes (surface registry + no-overload gate).

## Planned commits

1. `docs: plan #504 IngestionTests generator swap in TODO.md`
2. `feat(incident): expose rowGroupSize on GenerateFromRecordCount`
3. `test(incident): generate IngestionTests fixtures with DatasetGenerator (re-baseline)`
4. `docs: remove TODO.md — plan executed`
5. (optional) `test(perf): fold IncidentLabBenchmark generator into DatasetGenerator`

## Runtime delta (measured)

- before (test-local generator, 10,000 rows / 100 groups / ~200 instance rows): _pending_
- after (DatasetGenerator, 9,990 rows / 100 groups / ~28,800 instance rows): _pending_

## GitHub issues log

- [ ] _(none yet)_
