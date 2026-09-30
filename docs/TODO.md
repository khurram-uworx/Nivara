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

### Root cause — measured, not assumed

**The gate fails because it is usually run in Debug, and in Debug the comparison
is meaningless.** `CONTRIBUTING.md:147` documents plain `dotnet test`, which
defaults to Debug. In Debug, `Nivara.dll` is compiled unoptimized, so the
handwritten tiled kernel runs slow; but `Tensor.Transpose`/`FlattenTo` live in
`System.Numerics.Tensors`, which ships **ReadyToRun** and stays fully optimized
regardless of the consuming project's build configuration. The gate therefore
compares a Debug-compiled kernel against optimized framework code.

Measured at 1024x1024 on this machine (Core Ultra 7 255H, Release vs Debug,
identical source, `dotnet run -c <cfg>`):

| Build | tiled (median) | bcl (median) | ratio | Result |
|---|---|---|---|---|
| Release | 3.5–4.1 ms | 9.0–9.2 ms | **0.38–0.45** | tiled wins 2.2–3.2x |
| Debug   | 9.8 ms        | 8.9 ms      | **1.04–1.23** | fails **5 of 5** runs |

The Debug ratios (1.105, 1.165, 1.166, 1.147, 1.232) reproduce the issue's
failing trials (1.032, 1.084, 1.002) including the ~9.8 ms tiled absolute time.
**The tiling kernel is genuinely faster; the gate was measuring an artifact of
build configuration.**

### Hypotheses tested and ruled out

Grounding ran the measurements rather than assuming, and three plausible
explanations were **disproved**:

1. **Contention / machine load — ruled out.** Under a 16-thread memory-pressure
   generator, the interleaved probe still reported tiled winning 149/150 rounds
   (median ratio 0.220). Contention widens the gap; it does not collapse it.
2. **JIT tier-0 / missing warmup — ruled out as the cause.** Measured per-call
   timings for the first 40 calls: the tier-0 penalty is ~2.1x on tiled and
   ~2.45x on bcl — it hits *both* routes, so the ratio stays ~0.3. Warmup is
   still correct practice, but it does not explain the failures.
3. **Generic-vs-concrete dispatch penalty — ruled out.** `Transpose<T> where
   T : struct, INumber<T>` measured identical to a concrete `float`
   specialization (4.13 ms vs 4.11 ms, 1.00x penalty). The generic
   instantiation is not the problem.

### Secondary measurement defects (real, but not the cause)

These compound the above and should still be fixed — they are why the gate could
never be trustworthy once running:

1. **Single comparison.** `MeasureBestOfFive` is applied *inside* each route, then
   the two summaries are compared once. Two noisy numbers meet and one decides.
2. **Sequential, tiled-first, no warmup.** Load drift across the window always
   landed on `tiled` — systematic bias, not variance.
3. **Best-of-5 biases low**, unevenly, because the routes have different variance.

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

### 2. Gate on Release only, and compare distributions

The claim survives — tiled is 2.2–3.2x faster in Release — so the gate should
stay, but only where it measures something real:

- **Skip unless the assembly is optimized.** Detect the test assembly's
  `DebuggableAttribute` (or `#if DEBUG`) and `Assert.Ignore` outside Release.
  Rationale: a Debug comparison against ReadyToRun framework code is not a
  weaker measurement, it is a *wrong* one. The test already carries
  `[Category("Performance")]`; an explicit configuration skip makes the exclusion
  machine-independent instead of relying on someone remembering `--filter`.
- **Interleave, alternate order, warm up, compare medians** — the probe design
  from step 1. Assert on the median ratio with a generous floor (measured Release
  ratio is ~0.38, so a bound near 0.9 sits far above noise and still catches a
  genuine regression).

A Debug skip is honest: it reports "not measured in this configuration" rather
than inventing a pass or a fail.

Explicitly **not** doing: widening the loop count, adding a retry, or wrapping the
comparison in a tolerance band tuned to the 0.2% case — the issue rules all three
out as keeping an assertion below the noise floor.

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
2. `dotnet run -c Release --project tests/Nivara.SimdProbe -- transpose` — confirm
   the verdict holds.
3. `dotnet run -c Debug --project tests/Nivara.SimdProbe -- transpose` — confirm
   the probe *also* detects and reports the Debug condition rather than silently
   reporting the kernel as slow (this is the #482 defect).
4. Run the two reworked unit probes **10 consecutive times in Release**; require
   0 failures (baseline today: ~60% failure rate in Debug).
5. Run them in **Debug**; require the configuration skip, not a red test.
6. Full suite green, with and without `--filter "Category!=Performance"`.

## Probe lifecycle

Per the iterative-work probe harness rule, the diagnosis ran in a temp
directory first and only the durable part was promoted:

| Temp harness | Purpose | Disposition |
|---|---|---|
| `opencode/transpose-probe` | first interleaved A/B scratch harness | superseded by `TransposeKernelProbe`; delete |
| `opencode/tierprobe` | per-call timings, calls 1–40, to test the tier-0 hypothesis | one-off diagnostic; delete |
| `opencode/genericprobe` | generic vs concrete `Transpose<float>` dispatch penalty | one-off diagnostic; delete |
| `opencode/reproprobe` | faithful #482 gate reproduction in Release **and** Debug — this is what found the root cause | one-off diagnostic; delete |
| `opencode/loadgen` | 16-thread memory pressure to test the contention hypothesis | one-off diagnostic; delete |

Promoted to `tests/Nivara.SimdProbe/TransposeKernelProbe.cs` because the A/B is
**reusable**: it is the standing evidence for the #136 swap decision and must be
re-run whenever the BCL `Tensor.Transpose` contract changes. The Debug/Release
contrast is folded into the probe's own output so the next person sees both
numbers instead of rediscovering the trap.

Temp directories are left in place for the human to clean up.

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
