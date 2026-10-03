# #496 — `Incident.AnalysisTests` teardown file-lock makes the full suite exit 1 despite all tests passing

Branch: `khurram/496`. Issue: https://github.com/khurram-uworx/Nivara/issues/496

## Problem

`dotnet test -c Release --filter "Category!=Performance"` reports
`Failed: 0, Passed: 3755` yet exits 1. `Incident.AnalysisTests.OneTimeTearDown`
throws `IOException: The process cannot access the file 'requests.parquet' because
it is being used by another process` from its recursive `Directory.Delete`.

### The issue body's diagnosis is wrong on two counts

1. **"Give each parallel fixture its own unique temp directory (per-run GUID)" is
   already in place.** `AnalysisTests.cs:15` builds
   `Path.Combine(Path.GetTempPath(), $"inc-analysis-{Guid.NewGuid():N}")` — added in
   `05cf08b`, long before the issue was filed. There is no shared path to collide on.
2. **Nothing runs in parallel.** No `[Parallelizable]`, `[NonParallelizable]`,
   `FixtureLifeCycle`, `LevelOfParallelism`, `.runsettings` or `maxcpucount` exists
   anywhere under `tests/`. NUnit's default is sequential.

The real cause is a same-process, undisposed-file-handle leak.

### Root cause

`ParquetDataSource.CreateReader()` (`src/Nivara.Extensions/IO/ParquetDataSource.cs:396`)
opens the file as:

```csharp
var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
```

There is **no `FileShare.Delete`**, so a single live handle makes any attempt to
open the file with `DELETE` access fail — which is what `Directory.Delete` does.
The handle is released only by `ParquetLazySource.Dispose()`
(`ParquetDataSource.cs:494`), which is reachable only via
`QueryFrame.Dispose()` (`src/Nivara/Query/QueryFrame.cs:1111-1128`). `NivaraParquetReader.ScanAsQueryFrame`
documents exactly this contract (`NivaraParquetReader.cs:122-127`): "the file handle
stays open until the returned frame is disposed … important before deleting or replacing it."

`Analysis.AnalyzeGroupedAggregation` never disposes the `QueryFrame` it opens.
That frame is a method-local, so **no caller can release it** — the method returns
only materialized `NivaraFrame` data:

```csharp
var frame = Ingestion.LoadParquet(Path.Combine(datasetPath, "requests.parquet")); // handle opened
var result = frame.Filter(...).Filter(...).Collect();                              // handle never closed
return NivaraFrame.Create(...);                                                   // caller gets data only
```

`AnalysisTests.GroupedAggregation_NonEmptyResults` calls it four times (scenarios
A-D), leaking one handle per scenario's `requests.parquet`. Teardown then fails on
the first locked file it reaches.

### Why it is intermittent, and why "34/34 in isolation" proves nothing

The leaked `FileStream`'s `SafeHandle` is released only by the finalizer, so
teardown success depends on GC timing. The opt-in `NivaraResourceManager` tracker
that could reclaim abandoned lazy frames is toggled **process-globally** by
`ResourceManagementPropertyTests.cs:20` / `:27`, and is *off* while `AnalysisTests`
runs — so nothing reclaims the handle deterministically. A GC-timing-dependent pass
is not a fix, and the issue's isolation run is exactly that kind of evidence.

### Same defect, already hidden elsewhere

`StreamixScenarios.cs:35`, `:72` and `:143` each build
`var query = Ingestion.LoadParquet(...)` and never dispose it. That leak is why
`StreamixScenarioTests.cs:24-25` currently swallows the teardown failure:

```csharp
try { Directory.Delete(tempDir, true); }
catch (IOException) { /* file lock from Parquet reader — acceptable */ }
```

A blanket `catch` that hides a real defect is itself a gate reduction; fixing the
leak lets that catch go.

## Proposed changes

### 1. `samples/Nivara.Samples/Incident/Analysis.cs`

- `AnalyzeGroupedAggregation` (L217, L221): `using var frame = …` and
  `using var result = …`.
- `AnalyzeGroupedAggregationWithTypedLinq` (L258, L259): `using var frame = …` and
  `using var collected = …`.
- Optional tidy: `AnalyzeDeploymentCorrelation` has both `using var deployments`
  (L76) and an explicit `deployments.Dispose()` (L135). Harmless today (the
  `disposed` guard makes the second call a no-op); drop the explicit call only if
  it does not muddy the diff.

### 2. `samples/Nivara.Samples/Incident/StreamixScenarios.cs`

`using var query = …` at L35, L72, L143. Method-scope disposal is correct for the
epoch-reusing loop at L143.

### 3. `tests/Nivara.Tests/Incident/AnalysisTests.cs:107`

Add the missing `using` on `Analysis.AnalyzeGroupedAggregation`.

### 4. `tests/Nivara.Tests/Incident/StreamixScenarioTests.cs:22-27`

Delete the `try`/`catch (IOException) { /* …acceptable */ }` so the fixture gates
again.

### 5. New `tests/Nivara.Tests/Incident/AnalysisResourceTests.cs` — deterministic gate

The flake is GC-timing-dependent, so the gate must not be. New fixture with its own
GUID temp dir and `DatasetGenerator.GenerateFromRecordCount(tempDir, "A", 10_000)`
(already used by `StreamixScenarioTests`, so no third copy of the 180-line
`GenerateSmallDataset`). Plain teardown, no swallow.

One test per `Analysis.*` entry point (6): call it, dispose what the API returns,
then probe the file:

```csharp
static void AssertUnlocked(string path)
{
    using var probe = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None);
}
```

`FileShare.None` fails if **any** other handle to the file is open, so it detects the
leak deterministically and without deleting shared data.

**Baseline probe before the call as well**, so a red verdict is attributable: a
green baseline with a red post-call probe means *this* call leaked; a red baseline
means an earlier test leaked. One failure reason per verdict.

Expected: the two grouped-aggregation tests are **red before** the fix, green after.

## G1 grounding outcome

**Official docs (`FileShare`, `File.Delete`) confirm the mechanism:**

- `FileShare.Delete` — "Allows subsequent deleting of a file." It is a distinct flag
  (value 4) that `FileShare.Read` (value 1) does not include, so
  `ParquetDataSource.CreateReader`'s `FileShare.Read` withholds delete permission
  from other handles.
- `File.Delete` throws `IOException` — "The specified file is in use." That is the
  exact exception and message shape the issue reports.
- `FileShare.None` — "Declines sharing of the current file. Any request to open the
  file (by this process or another process) will fail until the file is closed."
  This validates the probe in the new gate: `FileShare.None` fails while *any*
  other handle is open, so it detects the leak deterministically.

Note the message says "another process" even when the holder is the current process
— which is why a same-process leak was misread as a cross-process race.

**Codebase navigation (`code-memory` + reads):**

- `QueryFrame.Dispose()` (`src/Nivara/Query/QueryFrame.cs:1111`) disposes the shared
  `source`; every derived frame (`Filter`/`Select`/window ops) shares that one
  source, so disposing the returned frame is sufficient and the intermediate frames
  need no disposal of their own.
- `ParquetLazySource.Dispose()` (`ParquetDataSource.cs:494`) is idempotent via its
  `disposed` guard, so double-dispose is a no-op.
- `CsvLazySource` (`CsvDataSource.cs:583`) opens the same way and holds the handle
  for the frame's lifetime, but `ParquetCsvConvergence_SameAnalysisSameResults`
  already uses `using`, so no CSV leak remains to expose after the Parquet fix.
- `NivaraResourceManager` tracking (the mechanism that could reclaim abandoned lazy
  frames) is opt-in and process-global, toggled by `ResourceManagementPropertyTests`
  — off while `Incident.*` runs, so no deterministic backstop exists today.
- `DatasetGenerator.GenerateFromRecordCount` writes the same six instance columns
  and the same four parquet files the test-local generator does, so the new gate can
  reuse it without behaviour drift.

**Considered and rejected:** adding `FileShare.Delete` to
`ParquetLazySource.CreateReader` would make the symptom disappear without fixing the
leak, and contradicts the documented contract at `NivaraParquetReader.cs:122-127`
("the file handle stays open until the returned frame is disposed"). Out of scope;
the leak is fixed at the source instead.

**Blast radius:** as tabulated above — two sample files, two test files, one new
test file. No public API, kernel, tensor or AutoDiff surface changes. The CLI's six
call sites into `Analysis.*` get correct handle lifetimes as a side effect.

## Verification steps — all executed, with results

1. **Gate red before the fix.** `AnalysisResourceTests` against unfixed code:
   `Failed: 4, Passed: 2`, `dotnet test` exit status **1**. The shape was not the
   predicted "exactly 2 red" — `AnalyzeGroupedAggregation` failed on its own
   post-call probe (*"AnalyzeGroupedAggregation left the Parquet file handle
   open"*) and the three alphabetically later tests then failed on their
   **baseline** probe. One root cause, four red tests; the baseline probe is what
   made the attribution unambiguous.
2. **A second, pre-existing bug surfaced.** `AnalyzeGroupedAggregationWithTypedLinq`
   kept failing after the disposal fix — with `SchemaValidationException`, not a
   lock. It maps `RequestRow`, which declares `DurationPercentRank`, but its
   pipeline never created that column. It has **zero callers** outside the new gate
   and had never been executed. Confirmed pre-existing by reproducing it on a
   `git stash`ed pristine tree. Fixed by adding the same `PercentRank` step
   `AnalyzeRegionalPartitioning` already uses (human-approved; not a handle defect).
3. **Gate green after the fix.** `AnalysisResourceTests`: `Passed: 6, Failed: 0`,
   exit status **0**.
4. **No incident regression.** `FullyQualifiedName~Nivara.Tests.Incident`:
   `Passed: 62, Failed: 0`, exit status **0** — including with the
   `StreamixScenarioTests` swallow removed.
5. **Full suite, 3 consecutive runs**, exit status captured from the `dotnet test`
   process itself with no pipeline in the way:

   | Run | Exit | Result |
   | --- | --- | --- |
   | 1 | 0 | Passed: 3761, Failed: 0, Skipped: 14, Total: 3775 |
   | 2 | 0 | Passed: 3761, Failed: 0, Skipped: 14, Total: 3775 |
   | 3 | 0 | Passed: 3761, Failed: 0, Skipped: 14, Total: 3775 |

   3/3 at exit 0, against a 1-in-N reported symptom.

Note on method: an initial gate run reported `EXITCODE=0` while printing four
failures, because `$LASTEXITCODE` after a pipe is the *pipe's* status. Every
result above re-runs without a pipeline. This is the exact trap AGENTS.md warns
about, and the same one that made #496 hard to read.

## Blast radius

| Change | Files | Downstream |
| --- | --- | --- |
| Dispose leaked `QueryFrame`s | `samples/Nivara.Samples/Incident/Analysis.cs`, `StreamixScenarios.cs` | `samples/NivaraIncident/NivaraIncident.Cli/Program.cs` (6 call sites), `tests/Nivara.Tests/Incident/AnalysisTests.cs`, `StreamixScenarioTests.cs`. Behaviour-preserving: `NivaraFrame` results are materialized before dispose, and `ParquetLazySource.Dispose` is idempotent (`disposed` guard). |
| Test `using` | `AnalysisTests.cs` | none |
| Remove teardown swallow | `StreamixScenarioTests.cs` | none — it can now fail if a leak returns, which is the point |
| New gate fixture | new file only | none |

No public API or contract change. No kernel, tensor or AutoDiff surface is touched.

## Planned commits — as landed

1. `docs: plan #496 in TODO.md` — `366ada2`
2. `docs: record #496 G1 grounding outcome in TODO.md` — `4e3fba7`
3. `test(incident): gate Parquet file-handle release per Analysis entry point` — `9813a9c`
   (committed **red**, before the fix, so the gate's value is evidenced rather than asserted)
4. `fix(incident): dispose Parquet-backed QueryFrames in Analysis and StreamixScenarios` — `24d10e1`
5. `test(incident): remove StreamixScenarioTests teardown IOException swallow` — `5015257`

Planned order was gate-after-fix; the gate was moved ahead of the fix so its red
state could be observed and recorded. Commits 3 and 4 remain separately
reviewable, and the fix commit's message states the before/after counts.

## Deferred work discovered during execution

- `AnalyzeGroupedAggregationWithTypedLinq` had **no test coverage and had never
  run**. Fixed here (human-approved). The wider gap — sample methods in
  `samples/Nivara.Samples/Incident/` being callable-but-unverified — is logged as
  a follow-up issue below.
- `ParquetLazySource` opens with `FileShare.Read` and no `FileShare.Delete`, so
  *any* future leak in *any* consumer blocks deletion on Windows. The new gate
  covers the six `Analysis` entry points only; other `ScanAsQueryFrame` consumers
  (`Csv`, `Json`, `AsyncStreamingTests`, `NivaraParquetReader.ScanQuery`) have no
  equivalent handle-release gate. Logged as a follow-up issue below.

## GitHub issues log

- [x] #496 — this work (created while verifying #494)
- [ ] #498 — no handle-release gate for non-Incident `ScanAsQueryFrame` consumers (created while fixing #496)
- [ ] #499 — `samples/Nivara.Samples/Incident` methods are callable but unverified (created while fixing #496)
- [x] #494 — BatchNorm eval mode produces no weight/bias gradient (merged, PR #497)

> As each task executes, if you find deferred work or a concern, create a tracked
> issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its
> number in the log above — don't rely on memory or wait until the end of the plan,
> as compaction can lose it.