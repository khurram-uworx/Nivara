# #545 — CI runs the full suite with only one test class opting out of `Category!=Performance`

Branch: `khurram/545` (worktree `E:\khurram-uworx\Nivara-545`, based on `main` @ `27cb5f36`).

## Problem

`.github/workflows/ci.yml:30` and `.github/workflows/cd.yml:42` both run

```
dotnet test --no-build --configuration Release --verbosity normal --filter "Category!=Performance"
```

`Category!=Performance` is the only thing keeping long-running work off shared GitHub runners,
and it is opt-in per test. A slow test that forgets its category does not fail — every push and
PR pays for it. The filter's coverage is implicit and nothing measures it.

Two of the issue's static premises are wrong, and correcting them changes the work:

| Issue says | Actually |
|---|---|
| `RequireOptimizedBuildForTiming()` is called from 3 files | **1 file** — `private static` in `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs:707`. Nine other files measure wall-clock or allocations and call nothing. |
| No per-test durations have been measured | A stale trx exists (`tests/Nivara.Tests/TestResults/net11run.trx`, 3276 tests). Stale and off-CI-shape, so it cannot serve as the report — but it establishes the threshold choice. |

Verified on `main`:

```
tests listed            : 3877
listed under the filter : 3875
excluded by the filter  : 2
  PropagateNullMask_PerformanceProbe_IsFasterThanReferenceTripleLoopForSparseMasks
  Transpose_PerformanceProbe_TiledKernelBeatsBclViewMaterialization
```

Category census across `tests/Nivara.Tests`: 133 `[Category]` attributes — 103 free-form
`Feature: …`, 27 `Integration`, 1 `Stress`, **2 `Performance`**. The filter excludes exactly
the two `Performance` tests; every free-form and `Integration` category still runs.

## What the stale measurement shows (provisional — step 1 replaces it)

`net11run.trx`, 3276 tests, Release-equivalent, no `[Parallelizable]` anywhere in the suite,
so per-test sum ≈ wall clock (233.4 s vs 236 s actual):

```
median 1.6ms   p75 4.2ms   p90 15ms   p95 36ms   p99 0.23s   max 138.9s
>=0.5s: 18    >=1s: 11    >=2s: 10    >=5s: 6    >=10s: 3

138.88s  MLNetPipeline_WorksCorrectly                     uncategorised
 15.94s  PartitionedWindow_ScatterEngine_AllocationBound    uncategorised
 13.03s  DistilBert_Inference_Latency                        uncategorised
  8.79s  RankKernel_RowNumber_NoPerCompareBoxing            uncategorised
  7.36s  GroupBy_TypedKeys_AllocationBound                   uncategorised
  5.69s  MiniLm_Inference_Latency                            uncategorised
  4.36s  EmbeddingGather_OneHotMatMul_Vs_Gather              uncategorised
  4.01s  RollingSum_NullFreeFastPath_AllocatesLessThan…      uncategorised
  2.60s  FromArrowTable_CopyPath_DoubleColumn_Throughput_1M  uncategorised
  2.18s  FromArrowTable_CopyPath_IntColumn_Throughput_1M     uncategorised
  1.04s  CreateBoundChannel_UnderLoad_ProducerBlocks…        uncategorised

  0.71s  Transpose_PerformanceProbe_…                    <- categorised
  0.26s  PropagateNullMask_PerformanceProbe_…            <- categorised
```

Three things follow.

1. **The categorisation is inverted relative to cost.** The only two tests the filter excludes
   are the two *cheapest* of the timing-shaped set.
2. **There is no long flat tail**, which answers the issue's step 5 with a measurement rather
   than an opinion: the 3266 tests outside the top 10 sum to ~30 s of the 233 s. Splitting into
   per-area jobs would have each job pay restore + build on a shared runner to parallelise ~30 s
   of work. Step 5 is closed as measured-negative.
3. **`MLNetPipeline_WorksCorrectly` is 59% of the run and is a correctness test**, not a timing
   test — it trains on 5 rows and asserts the prediction. Its cost is ML.NET first-use JIT /
   native load. Human decision: categorise it `Performance` (loses the ML.NET integration
   correctness check on PRs, keeps it in CD) — **decided: categorise**.

## Probe lifecycle (step 0 of the skill, recorded)

The measurement needed here is "per-test durations for the NUnit suite as CI runs it".

- `tests/Nivara.SimdProbe` modes: `correctness`, `benchmark`, `scalar`, `transpose`,
  `tensor-api`, `all`. All are kernel A/B or correctness probes over `src/Nivara` kernels.
- `tests/Nivara.PerformanceTests` flags: throughput / allocation / memory A/B for library
  kernels, plus the `--compare` no-regression gate.

Neither hosts a measurement of the NUnit suite itself, so no existing mode covers the question
and no scratch harness is warranted: the measurement belongs in the workflow as a trx logger,
which is also the durable artefact. No temp project will be created.

## Proposed changes

### 1. Measure — trx logger + artefact (acceptance criterion 1)

Add to `ci.yml` after the Test step:

```yaml
      - name: Upload test durations
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-durations
          path: tests/Nivara.Tests/TestResults/*.trx
```

and extend the Test step with `--logger "trx;LogFileName=ci.trx"`. Add the same logger to
`cd.yml:42` (no upload step needed — CD already gates on exit status).

`TestResults/` is already gitignored (`.gitignore:26`, `[Tt]est[Rr]esult*/`), so the trx write
does not dirty the tree.

Then run the suite locally in Release under the same filter to produce the report the issue asks
for: median, p90, worst offenders, and the wall-clock total. Publish as an issue comment.

**This must be a fresh run.** The trx above is stale and ran *without* the filter.

### 2. State the 2-second threshold in `AGENTS.md` (acceptance criterion 2)

Amend the existing bullet under "Performance & Optimization Thresholds":

> **Timing assertions need an optimized-build guard.** … and must carry
> `[Category("Performance")]` so CI's `--filter "Category!=Performance"` excludes it.

to add the duration rule and the command a contributor runs to re-measure:

> A test taking more than **2 seconds** on a shared runner is `Performance`. The suite's median
> is ~2 ms and p99 is ~0.2 s, so 2 s is far above the noise floor rather than a judgement call.
> Re-measure with:
> `dotnet test -c Release --filter "Category!=Performance" --logger "trx;LogFileName=d.trx"`
> then read `duration` per `UnitTestResult`.

2 s catches the 10 tests holding ~87% of suite time and leaves only
`CreateBoundChannel_UnderLoad_…` (1.04 s) uncategorised. Also update the pointer from
`TensorsHelperTests.cs` to the new shared helper (step 4).

`AGENTS.md` is not in `DocSnippetExtractor.GatedDocuments` and carries no ```csharp fences, so
this edit does not move the snippet-coverage counts in `DocumentationSnippetTests`.

### 3. Categorise the offenders (acceptance criterion 3)

Threshold 2 s, from the fresh measurement. **Per method, not per file** — `NivaraRowTests.cs` has
24 `[Test]`s but one allocation measurement; `TensorInteropTests.cs` has 50.

Every method that measures wall-clock gets both `[Category("Performance")]` **and**
`RequireOptimizedBuildForTiming()`. See the open decision below for the allocation-measuring
ones.

`MLNetPipeline_WorksCorrectly` gets `[Category("Performance")]` with a comment recording that it
is a correctness check moved off PRs for cost, so the gap is a documented decision rather than a
silent hole.

### 4. Promote `RequireOptimizedBuildForTiming()` to a shared helper

AGENTS.md rule 8 (consolidating duplicate logic): move the method out of `TensorsHelperTests` into
`tests/Nivara.Tests/TimingGuards.cs` as `internal static`, delete the private copy, update the two
existing call sites and the newly-categorised methods to call the shared one. Keeps the guard
available to the gate in step 5.

### 5. `SlowTestCategorisationTests` — the durable gate (acceptance criterion 4)

The gap the issue names is that the filter decays back to today's state. The one-time
categorisation in step 3 is not enough on its own.

Model it on `tests/Nivara.Tests/Exceptions/TypeNameUniquenessTests.cs`, which is the repo's
established shape for a repo-scanning gate:

- the scan lives in its own `internal static` method so negative controls can drive it;
- two negative controls drive it with a genuine violation and with a genuine non-violation;
- a `<remarks>` block states the gate's own coverage.

Scan `tests/Nivara.Tests/**/*.cs` for `[Test]`-attributed methods whose body touches
`Stopwatch`, `ElapsedMilliseconds` / `ElapsedTicks` / `GetTotalMilliseconds`, or
`GC.GetAllocatedBytes`, and fail when such a method lacks `[Category("Performance")]`.

#### Grounded requirements for the scan (NUnit docs, G1)

Two facts from <https://docs.nunit.org/articles/nunit/writing-tests/attributes/category.html>
change how this must be implemented:

1. **"Categories are inherited — a test inherits all categories from its fixture and assembly."**
   So a fixture-level or assembly-level `[Category("Performance")]` excludes every test beneath
   it. The scan must therefore resolve the *effective* category set — method-level **plus**
   enclosing type **plus** assembly-level — and only report a violation when `Performance` is
   absent from all three. A method-level-only scan would produce false positives the moment
   someone sensibly puts the attribute on a whole fixture.
   Today the repo has 237 `[TestFixture]`s, 133 method-level `[Category]`s and **0**
   assembly-level ones, so the inherited path is currently unexercised — which is exactly why
   the negative control has to drive it rather than trust it.

2. **Category names may not contain `,`, `+`, `-` or `!`** — these are operator characters in
   NUnit's own `cat == …` expression language. This explains the repo's free-form
   `Feature: …, Property: …` categories: they are inert with respect to VSTest's `Category!=`
   filter, which is why the 103 of them never exclude anything. No action needed, but the
   fixture's remarks should say the gate is not asserting those categories mean anything.

**Use Roslyn, not regex.** `Microsoft.CodeAnalysis.CSharp` 5.9.0 is already referenced by
`Nivara.Tests.csproj` (added for `DocSnippetCompiler`), so a real syntax tree costs no new
dependency. A regex over `.cs` text would have to re-implement attribute scoping and would be
fooled by `[Category("…")]` appearing in a comment or a string literal — the same class of
"the doc names an API that does not exist" defect that issues #524–#532 record.

**Stated coverage limit, to be written in the fixture's remarks:** this gate is
construct-driven, so it cannot catch a slow test that measures nothing — which is exactly the
`MLNetPipeline_WorksCorrectly` case. The duration rule in step 2 plus the trx artefact from step 1
is what covers that class; no in-repo mechanism can auto-detect it without a wall-clock
assertion on a shared runner, which is the flake factory AGENTS.md warns about.

### 6. Answer step 5 in writing

Post the serialisation numbers on the issue and close it as measured-negative.

## Decision (human, after G1)

**Categorise the allocation-bound guards `Performance`.** One uniform rule for every
measurement-bearing test. `WindowAllocationTests` (the #251 pooled-scratch / typed-kernel
allocation reduction) leaves PR runs. It still runs locally and in any unfiltered run. The
guards assert `GC.GetAllocatedBytesForCurrentThread` deltas, not wall-clock, so they get the
category and **not** `RequireOptimizedBuildForTiming()` — that guard exists for timing
assertions, which these are not.

Conditional commit 7 (cut the sample count) is dropped.

## Blast radius

**Filter semantics, grounded (G1).** Per Microsoft's `dotnet test` docs, for NUnit `Category`
and `TestCategory` are equivalent and `!=` is "not exact match", so `Category!=Performance`
excludes a test whenever `Performance` is in its *effective* category set — including via
inheritance. VSTest lookups are case-insensitive; NUnit category names themselves are
case-sensitive. Verified empirically on `main` by diffing `--list-tests` with and without the
filter: exactly 2 of 3877 tests are excluded.

| Change | Blast radius |
|---|---|
| `ci.yml`, `cd.yml` | CI/CD only, no library surface. New artefact + logger; exit-status behaviour unchanged. |
| `AGENTS.md` | Documentation only. Not snippet-gated; no coverage counts move. |
| `tests/Nivara.Tests/TimingGuards.cs` (new) | Internal to the test assembly. Two existing call sites plus the newly-categorised methods. |
| `TensorsHelperTests.cs` | Private helper removed, call sites redirected. The 2 already-categorised probes must stay passing. |
| 9 measurement-bearing test files | **Reduces what CI runs.** Each `[Category("Performance")]` added is a test removed from PR runs. This is the risk to weigh, quantified per the open decision above. |
| `SlowTestCategorisationTests.cs` (new) | New gate. Its own failures are the intended signal. Must not flag the repo's own helper or the gate itself. |

No `src/` file is touched. No public API changes. Nothing here can affect a shipped package.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` — clean.
2. **Ask before running.** `dotnet test -c Release --filter "Category!=Performance"` with the trx
   logger; report the distribution, worst offenders, and wall-clock total. Confirm exit status of
   the process, not of a pipeline filter.
3. Confirm the two already-categorised probes are still excluded, and that every newly
   categorised method is excluded — by diffing `--list-tests` with and without the filter, not by
   reading attributes.
4. Run the gate's negative controls and confirm they **fail for the right reason** — a gate never
   observed failing is not known to work.
5. Run the full suite once unfiltered to confirm the newly categorised tests still pass in Release
   (they are no longer covered by step 2's filtered run).
6. `dotnet test -c Release --filter "FullyQualifiedName~DocumentationSnippetTests"` — confirm the
   snippet-coverage counts are unmoved by the `AGENTS.md` edit.

## Planned commits

1. `docs: plan #545 in TODO.md`
2. `ci: record per-test durations as a trx artefact`
3. `test: state the 2-second shared-runner threshold in AGENTS.md`
4. `test: promote RequireOptimizedBuildForTiming to a shared TimingGuards helper`
5. `test: categorise the tests over the 2-second threshold`
6. `test: gate against a slow timing test added without its category`
7. *(conditional, only if open decision 1 is taken)* `test: cut WindowAllocationTests sample count to fit the threshold`
8. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #545 — this work.
- [ ] (to create during step 3) — the ML.NET integration correctness check no longer runs on PRs
  after `MLNetPipeline_WorksCorrectly` is categorised; track re-establishing it in a cheaper form.
- [ ] (to create during step 1) — if the fresh measurement shows a test whose cost is first-use
  JIT / native load rather than real work, track moving that cost to a `[OneTimeSetUp]` warm-up
  so the test can stay in CI.

Reminder: as each task executes, if deferred work or a concern appears that is outside this plan,
create the issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record the number
above — do not rely on memory, since compaction during execution can lose it.