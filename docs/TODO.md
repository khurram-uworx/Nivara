# Plan — #499: `samples/Nivara.Samples/Incident` methods are callable but unverified

Branch: `khurram/499` (off `khurram/498`). PR target: `khurram/498`.

## Problem

`Analysis.AnalyzeGroupedAggregationWithTypedLinq` had **zero callers** and had never been
executed. It maps `RequestRow`, which declares `DurationPercentRank`, but its pipeline never
created that column, so the first call threw:

```
Nivara.Exceptions.SchemaValidationException : Property 'DurationPercentRank' on row type 'RequestRow'
does not map to any column in the frame schema.
```

#496 fixed that method and added `AnalysisResourceTests` to gate handle release. But the
underlying issue stands: **the Incident sample's public surface is broader than anything that
runs**, and nothing makes the gap visible.

### Measured inventory of the public surface

| Member | Real caller | Executed by a test? |
|---|---|---|
| `Analysis.AnalyzeDegradationOrdering` | CLI (3 modes) | yes |
| `Analysis.AnalyzeDeploymentCorrelation` | CLI | yes |
| `Analysis.AnalyzeSaturationOrdering` | CLI (2 modes) | yes |
| `Analysis.AnalyzeRegionalPartitioning` | CLI (2 modes) | yes |
| `Analysis.AnalyzeGroupedAggregation` | CLI | yes |
| **`AnalyzeGroupedAggregationWithTypedLinq`** | **none** | `AnalysisResourceTests` only — a `FileShare.None` probe that asserts nothing about results |
| **`Analysis.DeploymentRow`** | **none** | **none** |
| **`Schema.RequestTelemetry`** | **none** | **none** |
| **`Schema.DeploymentEvent`** | **none** | **none** |
| **`Schema.ServiceDependency`** | **none** | **none** |
| **`Schema.InstanceState`** | **none** | **none** |
| **`DatasetGenerator.Generate(path, id, scale)`** | CLI + `Nivara.PerformanceTests` | **not in NUnit** (10M records — see below) |
| **`Scenarios.Get(null)`** | — | **none**; throws `NullReferenceException`, not `ArgumentException` |
| **`ServiceEvent.Timestamp` / `.Service` / `.Magnitude`** | — | **none** (only `.EventType` is asserted) |
| **`WindowResult.Empty`** (`StreamixScenarios.cs:199`) | internal path only | **never reached** |
| `StreamixScenarios.Run*` ×3 | CLI | yes, but only `> 0` / `>= 0` |

`Schema.cs` is **entirely dead** — all four records have zero references repo-wide, including the
CLI and README. `RequestTelemetry` duplicates `RequestRow`; `InstanceState` duplicates
`InstanceRow`.

### The sixth analysis is not in the CLI

`README.md` says "Run all five analyses" and the CLI really runs five
(`Program.cs:144-176`). `AnalyzeGroupedAggregationWithTypedLinq` is a **sixth analysis with no
caller at all**, including in `--benchmark`.

### A coverage-presence gate would NOT have caught the original bug

`AnalysisResourceTests.cs:82-88` **already called** the broken method and passed, because it
only probed the file handle. So the gate cannot be "is this member named in a test?" — it has to
be **execute the member and look at what it returns**. That distinction decides the gate's shape.

### Making the members `internal` would not have caught it either

C# emits **no** diagnostic for an unused `internal` member, and the repo's `IDE0051` is
`suggestion` and only covers `private` (`.editorconfig`). `Nivara.Samples.csproj:31-38` also
already grants `InternalsVisibleTo` to `Nivara.Tests` and `Nivara.PerformanceTests`, so
`internal` removes no test access. The issue's option (b) is recorded here as **rejected on
mechanism**, not on taste.

### #498 already owns handle release — do not duplicate it

`khurram/498` landed `tests/Nivara.Tests/IO/FileHandleProbe.cs` and
`ScanAsQueryFrameHandleTests.cs`, and repointed `AnalysisResourceTests` at the shared probe.
This issue's gate is about **execution coverage**; handle release is already gated and stays
there. `FileHandleProbe` is reused for any disposal the new exercises need.

## Decisions (confirmed with human before execution)

1. **Smoke gate + delete dead code**, rather than only adding tests. A new untested public member
   must fail the suite, not merely be absent from a hand-written list.
2. **Delete the five zero-reference types** (`Schema.cs` ×4, `DeploymentRow`). They are
   documented nowhere and referenced by nothing; keeping them means keeping shapes that
   nothing validates — which is the bug class.
3. **Wire the sixth analysis into the CLI** (`analyze` + `--benchmark`). It is the showcase for
   the typed LINQ `GroupBy` capability the README advertises, and it returns a distinct column
   set from `AnalyzeGroupedAggregation`.
4. **`DatasetGenerator.Generate` goes in the gate's skip list with a reason.** It writes
   `10_000_000 * scale` records, so it cannot run in CI. It *is* verified by
   `Nivara.PerformanceTests/IncidentLabBenchmark.cs:70-75`, which is a manual
   `dotnet run -c Release` harness that CI never invokes. Per `GUIDELINES.md`
   §"A Gate Must State Its Own Coverage", this is reported as skipped-with-reason, never
   silently dropped.
5. **Fix `Scenarios.Get(null)` to `ArgumentNullException`** (one-line `ThrowIfNull`), since it is
   public sample API driven by CLI args.
6. **Consolidate the duplicated dataset generator in this change** — the human's call, against
   `AGENTS.md`'s "do not alter a test's inputs inside a commit about something else". Mitigated
   by making it **its own commit whose sole stated reason is the re-baseline**, so the baseline
   move is a commit headline rather than something buried in an unrelated one.

## Proposed changes

### 1. Delete the dead types

- Delete `samples/Nivara.Samples/Incident/Schema.cs` (four records, zero references).
- Delete `Analysis.DeploymentRow` (`Analysis.cs:22-28`).
- `RequestRow` / `InstanceRow` stay — both are `Query<T>()` targets.

### 2. Wire the sixth analysis into the CLI

- `samples/NivaraIncident/NivaraIncident.Cli/Program.cs`: add a
  `=== Typed LINQ Grouped Aggregation ===` block after `:175`, and a
  `RunBenchmarkIteration("Typed LINQ Grouped Agg", …)` after `:137`.
- `samples/NivaraIncident/README.md`: five → six (quick-start, mode table, "What it exercises").

### 3. New gate — `tests/Nivara.Tests/Incident/IncidentSurfaceTests.cs`

Modelled on `tests/Nivara.Tests/AutoDiff/OpNodeInputContractTests.cs:704`
(`AssertEveryMultiTensorOpCovered`), which is the repo's existing "reflect a surface and fail on
uncovered members" precedent.

**Rows** = each public type in `Nivara.Samples.Incident` plus each public declared method,
public static property, and public static field. Nested types (`StreamingSummary`, `WindowResult`,
`WindowedAnalyticsSummary`, `AutoDiffSummary`) are **excluded** — they are DTOs constructed by
the registered `Run*` exercises, and `WindowResult.Empty` gets a direct test in step 4.

**Each row resolves to exactly one of:**

| Verdict | Meaning |
|---|---|
| `exercised` | a registered closure invokes it |
| `covered-by <exercise>` | transitively covered; the gate names the exercise |
| `skipped: <reason>` | explicit, non-empty reason required |

**Two tests, because presence and success are different failures:**

1. `IncidentSurface_HasNoUnregisteredMembers` — every reflected row resolves; fails listing gaps.
2. `IncidentSurface_AllExercisesSucceed` — runs every closure once, **one failure reason per
   member**, and prints `N of M executed` to `TestContext`.

**POCO rows and the original bug class.** `RequestRow` / `InstanceRow` / `ServiceEvent` /
`IncidentScenario` are one row each, not one row per property. `RequestRow` / `InstanceRow` are
satisfied transitively by the registered analyses that call `Query<T>()` — and that binding is
exactly the operation that threw the original `SchemaValidationException`, so the row is bound
tightly to the bug. Note `InstanceRow.PeakQueueDepth` is **not** in `instances.parquet`; it is
created by `RollingMax` at `Analysis.cs:158` before `Query<InstanceRow>()`, so
`AnalyzeSaturationOrdering` must stay a registered exercise for that row to hold.
`ServiceEvent` / `IncidentScenario` get a direct construction exercise touching every property.

**Fixture.** `OneTimeSetUp` generates one dataset via
`DatasetGenerator.GenerateFromRecordCount(dir, "A", 10_000)` into a GUID temp dir, matching
`AnalysisResourceTests.cs:29-32`. Every exercise disposes its frames — reuse `FileHandleProbe`
where a handle is opened. Unique dir per run avoids the #496 parallel-teardown race.

### 4. Edge cases and weak assertions

- `Scenarios.cs:86`: add `ArgumentNullException.ThrowIfNull(id)`.
- `ScenarioTests.cs`: `Get(null)` → `ArgumentNullException`; `Get("")` → `ArgumentException`;
  assert `ServiceEvent.Timestamp` / `.Service` / `.Magnitude`, which no test reads today.
- `StreamixScenarioTests.cs`: the three cases assert only `> 0` / `>= 0`. Replace with real
  invariants — `WindowEnd >= WindowStart`, `ErrorRate ∈ [0,1]`,
  `TotalRows == Σ WindowResult.RowCount`, finite `FinalLoss`. Add a direct shape test for the
  `WindowResult.Empty` sentinel.

### 5. Consolidate the duplicated generator

`AnalysisTests.GenerateSmallDataset` (`AnalysisTests.cs:362-543`) is a 180-line reimplementation
of `DatasetGenerator`, duplicating the service/region tables, the per-service profiles, and the
instance generation. So every `AnalysisTests` case runs against a dataset **production never
produces**, and `AffectedServices` is applied differently in the two
(`AnalysisTests.cs:394` vs `DatasetGenerator.cs:83-85`, where the real one also honours
`affectedRegion` and the per-event latency ramp).

Replace the helper with `DatasetGenerator.GenerateFromRecordCount(dir, sid, 10_000)`.

**All `AnalysisTests` assertions were checked against the swap and are property-based**
(`Does.Contain`, range checks, quantile ordering, determinism) — **no hard-coded absolute
counts**. Request row count is identical: both compute `10_000 / 30 = 333` per minute × 30
minutes = **9,990**. The three scenario-story assertions each have a matching special case in
the real generator:

| Assertion | Survives? | Why |
|---|---|---|
| `ScenarioB_DeploymentCorrelation_DeployAtMinute17` | yes | `DatasetGenerator.cs:158-165` has the identical `scenarioId == "B" && i == 0` override |
| `ScenarioD_RegionalPartitioning_ApSouth1Present` | yes | `DatasetGenerator.cs:51` sets `affectedRegion = "ap-south-1"` for D |
| `ScenarioA_DegradationOrdering_OrdersFirst` | yes | name is misleading — the body (`:151-154`) only asserts four services are *present* |

**Known cost:** the real generator emits ~28,800 instance rows vs the test-local 800
(`DatasetGenerator.cs:184-188`: 8 services × ~12 instances × 10 regions × 30 snapshots), ~36× more
through `AnalyzeSaturationOrdering`'s partitioned `RollingMax` + 4 quantiles + stddev across 4
scenarios. To be measured and reported, not assumed.

## Verification steps

1. **G1** — ground the plan, state blast radius, clear with the human before implementing.
2. `dotnet build Nivara.slnx` — no new warnings (`TreatWarningsAsErrors` is on repo-wide).
3. Targeted runs, `-c Release`: `Incident` namespace fixtures, `-c Release`
   (`AGENTS.md`: Debug compares an unoptimized kernel against ReadyToRun framework code).
4. **Full suite** `-c Release --filter "Category!=Performance"`, capturing the **process exit
   status**, not a pipeline filter's (`#496`'s lesson).
5. Measure and report the saturation-ordering runtime delta from step 5.
6. Re-run the new gate fixture to confirm it is stable, not flaky.
7. **G2** — two reviews before deleting this file.

## Planned commits

1. `docs: plan #499 in TODO.md`
2. `refactor: delete dead Incident sample types with zero references`
3. `feat(cli): wire the typed LINQ grouped aggregation into analyze and benchmark`
4. `test: gate every Incident sample public member by execution`
5. `fix: reject null scenario id and cover Scenarios/Streamix edge cases`
6. `test: generate AnalysisTests fixtures with DatasetGenerator (re-baseline)`
7. `docs: remove TODO.md — #499 plan executed`

Commit 6 is deliberately last and separately stated: it is the only commit that moves an
existing baseline, so it must be identifiable as such.

## Blast radius

- **Sample surface only.** No `src/` change, no public library API change, no behavior change.
- `Schema.cs` deletion: zero references confirmed by grep across `.cs`/`.md`/`.slnx`/`.csproj`.
  `Nivara.Samples` is not packable/published; its 8 referencing projects are all in-repo.
- `Analysis.DeploymentRow` deletion: zero references confirmed.
- CLI change: additive `Console.WriteLine` blocks only; `analyze` grows from 5 to 6 analyses.
  Existing flags, defaults, and output format for the first five are untouched.
- `Scenarios.Get` change: `NullReferenceException` → `ArgumentNullException` for a `null` input.
  No in-repo caller passes `null` (CLI passes `args[++i]`, already non-null-checked by the parse).
- New gate: test-only, new file, no existing fixture modified. It duplicates no existing gate —
  handle release is #498's, this is execution.
- `AnalysisTests` fixture swap: **the one real behavior change in this plan.** Re-baselines 19
  `AnalysisTests` cases onto the production generator. Assertions verified property-based; runtime
  to be measured.
- **Not covered:** the gate proves each registered member *executes*, not that its output is
  *correct*. Correctness stays with the per-fixture assertions in step 4/5. It also cannot run
  `DatasetGenerator.Generate`, so that one member's coverage is reported, not verified, here.

## GitHub issues log

- [ ] #499 — `samples/Nivara.Samples/Incident` methods are callable but unverified (this work)

> As each task executes, if you find deferred work or a concern outside this plan, create a
> tracked issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its
> number above — do not rely on memory, as compaction during execution can lose items.