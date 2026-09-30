# Fix flaky timing gates in `TensorsHelperTests` (#482)

## Problem

`Transpose_PerformanceProbe_TiledKernelBeatsBclViewMaterialization`
(`tests/Nivara.Tests/Tensors/TensorsHelperTests.cs:464`) fails 3/5 runs on
unmodified `main`. Trial 5 failed on a **0.2%** margin (tiled 88029 vs bcl
87888). A sub-1% margin timing assertion cannot hold on a shared machine.

`PropagateNullMask_PerformanceProbe_IsFasterThanReferenceTripleLoopForSparseMasks`
(`TensorsHelperTests.cs:119`) shares the same `MeasureBestOfFive` helper and the
same single-comparison structure. Not observed failing, but carries the identical
fragility.

### Root cause (three compounding defects in the measurement, not the kernel)

1. **Single comparison.** `MeasureBestOfFive` is applied *inside* each route, then
   the two resulting numbers are compared once. Two noisy summaries meet and one
   number decides.
2. **Sequential, tiled-first, no warmup.** CPU frequency / thermal / background
   load drift across the measurement window; the route measured first absorbs the
   drift. This is *systematic bias*, not variance, and it always lands on `tiled`.
3. **Best-of-5 biases the estimate low**, unevenly, because the two routes have
   different variance (cache-blocked kernel vs BCL view materialization).

### Evidence that the claim itself may be false

In **every failing trial in the issue, `tiled` was SLOWER than BCL**
(ratio = tiled/bcl > 1: 1.032, 1.084, 1.002). A gate that fails only when the
measured ratio exceeds 1.0 is asserting an ordering the data does not support.
Before choosing between "fix the measurement" and "drop the claim", the probe
must establish which way the routes actually compare.

### Secondary finding — inconsistent CI hygiene

`ci.yml:30` filters `--filter "Category!=Performance"`.

- `Transpose_PerformanceProbe_*` — **has** `[Category("Performance")]` → excluded from CI.
- `PropagateNullMask_PerformanceProbe_*` — **has no category** → **runs in CI**, carrying
  the identical fragile structure with no escape hatch.

So the probe CI actually exercises is the untriaged one.

## Grounding (G1)

- `docs/TENSORS.md:180-181` carries the claim this gate underwrites: "Parity +
  performance regression gates in `TensorsHelperTests` fail if the BCL
  view-materialization route ever beats the tiled kernel, signalling a
  re-evaluation." Any change to the gate must move this claim in the same change.
- `CHANGELOG.md:139` records the #136 swap-target verification.
- Repo's established answer to "hand kernel vs BCL baseline" is a standalone
  probe with a **median-of-trials** harness, not a unit-test assertion:
  `tests/Nivara.SimdProbe` (`Benchmark.cs:63` warms up 20 passes, 7 trials, takes
  the median) and `tests/Nivara.PerformanceTests` (`CpuGemmProbe.cs`,
  `GateEvaluator.cs` with explicit per-leg floors and a documented
  `BandwidthBoundMinOpsFraction` for machine-state drift).
- `GateEvaluator.cs` is the precedent for shared, unit-testable gate policy:
  pure decision logic in a linked source file, compiled into both the harness and
  the test project, with exact-value unit tests.
- Microsoft guidance on microbenchmarking: prefer **median** over mean, discard
  outliers, warm up before timing, and expect run-to-run variance on real
  hardware — a single-shot ordering assertion is not a supported measurement.

## Proposed changes

### 1. Add a `transpose` mode to `tests/Nivara.SimdProbe` (new file `TransposeKernelProbe.cs`)

Standalone, self-contained A/B of the two #136 routes at the gate's shapes,
using the project's existing median-of-trials harness style:

- warm up both routes before timing
- **interleave** the two routes and alternate which goes first, so
  first-measured drift cancels instead of always penalising `tiled`
- report per-route median, per-round ratio distribution, and win rate
- run **tiers/shapes** so the comparison is a distribution, not one number
- emit a clear verdict line, e.g. `VERDICT: tiled wins N/M rounds (median ratio X)`

Wire it into `Program.cs` as `"transpose"` and into the README's mode table.

### 2. Decide the gate's fate from the probe's data (human decision, see below)

- **If tiled wins by a margin comfortably above the noise floor** → the claim is
  real but the *measurement* was broken. Keep a timing assertion only if it can be
  made robust (interleaved + median-of-ratios); otherwise move it to the probe
  and keep only the deterministic parity test in the unit suite.
- **If the routes are within noise** → the honest finding is "comparable". Delete
  the ordering assertion, keep `Transpose_MatchesBclViewMaterialization_AcrossShapes`
  (deterministic, valuable), and update `docs/TENSORS.md` to say the routes are
  comparable rather than that tiled wins.

Explicitly **not** doing: widening the loop count, adding a retry, or wrapping the
0.2% comparison in a tolerance band — the issue rules all three out as keeping an
assertion the data shows is below the noise floor.

### 3. Fix `PropagateNullMask_PerformanceProbe_*` (same pass)

Same `MeasureBestOfFive` helper, same single comparison. Reference is an O(n³)
triple loop vs an O(n²) optimized path at 160×160, so the expected margin is
large and this gate is probably genuinely load-bearing — but it must be measured
the same robust way, and it must be given `[Category("Performance")]` so CI stops
running an un-triaged timing gate by accident.

### 4. Update the claims that carry the measurement

- `docs/TENSORS.md:180-181` — restate to match whatever step 2 lands.
- `CHANGELOG.md` — record the gate change.

## Blast radius

**Changes are confined to tests and docs. No `src/` code changes.**

| File | Change | Risk |
|---|---|---|
| `tests/Nivara.SimdProbe/TransposeKernelProbe.cs` | new | none — standalone, manual-run, not in CI |
| `tests/Nivara.SimdProbe/Program.cs` | add `transpose` mode + `all` | none |
| `tests/Nivara.SimdProbe/README.md` | document mode + results | none |
| `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs` | rework 2 probes | removes/strengthens a timing assertion; correctness tests untouched |
| `docs/TENSORS.md`, `CHANGELOG.md` | claim text | none |

Not touched: `src/Nivara/Tensors/TensorsHelper.cs` (the tiled kernel itself), any
public API, any runtime behaviour. The only production-code consumer of the tiled
transpose is `GradKernels.Transpose` (`src/Nivara/AutoDiff/Operations/GradKernels.cs:754`),
which is covered by `GradKernelsTests` — unchanged.

## Verification steps

1. `dotnet build Nivara.slnx` clean.
2. `dotnet run -c Release --project tests/Nivara.SimdProbe -- transpose` — record
   the verdict; this is the evidence for the step-2 decision.
3. Run the two reworked probes **10 consecutive times**; require 0 failures
   (baseline today: ~60% failure rate for the transpose probe).
4. Full suite green, with and without `--filter "Category!=Performance"`.
5. Confirm no new CI failures introduced by the category change.

Asking before running `dotnet test` / `dotnet run`.

## Planned commits

1. `docs: plan #482 flaky timing gates in TODO.md`
2. `test: add transpose route A/B probe to SimdProbe` (+ Program.cs + README)
3. `test: rework TensorsHelper timing gates to interleaved median comparison`
4. `docs: restate #136 transpose claim to match measured evidence`

Additive fix commits allowed if a review surfaces a gap.

## GitHub issues log

As tasks execute, deferred work and concerns get an issue created **immediately**
via `gh issue create --repo khurram-uworx/Nivara` and recorded here — not held
in memory.

- [ ] #482 — this work (flaky timing gates in `TensorsHelperTests`)
- [ ] (to fill) — follow-ups discovered during execution