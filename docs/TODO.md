# Plan — #901: `--benchmark` mode for NivaraClassifier (CPU vs Silk)

## Problem

Issue #901 asks for measured evidence on the Silk classifier training/prediction paths:

1. Does the GPU forward (FC projections) give measurable speedup at realistic batch sizes?
2. Is OpenCL fast enough for inference/predict to justify Silk in this sample?
3. Are there avoidable host↔device transfers left in the hot path?

The sample has no way to produce those numbers today: `--command train`/`predict` only report
accuracy and per-epoch wall clock, and there is no CPU-vs-Silk comparison on one machine.

Acceptance criteria from the issue:

1. Documented measurements for the CPU vs Silk classifier paths.
2. A clear default heuristic for when Silk is worth enabling.
3. No unnecessary transfers/readbacks left in the hot path.

## Approach

Add `--benchmark` to `samples/NivaraClassifier`, following the measurement discipline
`tests/Nivara.PerformanceTests` already uses (`LayaForwardProbe`: one warmup, three readings,
median primary; `ApplyMaskProbe`: interleaved A/B rounds because best-of-N is what made #482
flaky).

### Protocol

Per leg, per batch size:

1. Calibrate/warm: 5 throwaway iterations, measure per-iteration cost.
2. `N = clamp(ceil(20 ms / perIterMs), 1, 2000)` — one round is ~20 ms of work, so a
   sub-millisecond forward still yields a defensible median after dividing by N.
3. Three rounds, each a loop of N iterations divided by N.
4. Report median (primary), min, max, iterations, and ops/s.

Deviations, both deliberate:

- `epoch` and `construct` run `N = 1`: one sample is already measurable.
- Round order alternates between the CPU and Silk leg of the same measurement (CPU first on
  round 1, Silk first on round 2, CPU first on round 3). The side measured first on every round
  absorbs load drift — the #482 lesson.

### Legs

| leg | purpose |
| --- | --- |
| `cpu forward` / `silk forward` | AC 1 baseline at the swept batch size |
| `cpu predict` / `silk predict` | real predict path, including input build + argmax |
| `cpu epoch` / `silk epoch` | AC 1 training baseline (`epochs: 1`) |
| `silk construct` | per-instance OpenCL `BuildProgram` — decides whether a short-lived process should pick Silk at all |
| `silk transfer` | upload `B×E` + download `B×2` floats, nothing else: the AC 3 evidence |

`silk transfer / silk forward` is the AC 3 answer without touching `src/`: if the ratio is high
the readback is the story, and per-phase profiling hooks in `GpuTextLinearClassifier` become a
follow-up issue rather than part of this change.

One weight set feeds both sides: build the CPU model, read
`GetParameters().Values[i].Tensor.Data.ToArray()`, push the same arrays through
`GpuTextLinearClassifier.SetWeightsCpuLayout`, then measure.

CPU epoch uses the real path: `FrameBuilder.BuildDocumentClassificationFrame` once,
`TensorDataset` → `DataLoader` at the swept batch size → `TrainingLoop<float>` with `epochs: 1`.

### CLI

- `--benchmark` sets `Command = "benchmark"`; `--command benchmark` also accepted.
- Honours `--mode`, `--batch-size`, `--embedding-dim`, `--hidden-dim`, `--max-seq-len`,
  `--num-samples`, `--epochs`, `--lr`, `--seed`.
- Default batch sweep `{1, 8, 32, 128}`; an explicitly passed `--batch-size` pins one value (needs
  an internal `BatchSizeExplicit` flag so the default `32` is not read as explicit).
- `--silk` guard at `Program.cs` widens from `("train" or "predict")` to
  `("train" or "predict" or "benchmark")`; the message moves with it.
- Debug builds refuse to measure (exit 2), mirroring `LayaForwardProbe`: an unoptimised
  `Nivara.dll` against ReadyToRun BCL code is not evidence.
- No OpenCL device: print the runtime's message and run CPU-only, exit 0.
- Output: human table only. No `--json`, no `--compare`.

### Docs

`samples/NivaraClassifier/README.md` only — CLI row, protocol, the measured table, and the
heuristic derived *after* the run. No `docs/` changes, so no `DocsCoveragePins` movement.

## Proposed changes

| file | change |
| --- | --- |
| `samples/NivaraClassifier/BenchStats.cs` (new) | `Summarize`, `IterationCount`, `Verdict` — pure arithmetic on ticks/doubles |
| `samples/NivaraClassifier/Benchmark.cs` (new) | legs + table |
| `samples/NivaraClassifier/Program.cs` | `--benchmark` parse, guard message, dispatch, help |
| `samples/NivaraClassifier/NivaraClassifier.csproj` | `InternalsVisibleTo Nivara.Tests` (repo convention, cf. `Nivara.Samples.csproj`) |
| `tests/Nivara.Tests/Samples/BenchStatsTests.cs` (new) | median/min/max, iteration count, verdict band |
| `tests/Nivara.Tests/Samples/ClassifierCliArgumentTests.cs` (new) | one spawn: `--silk --command generate` rejection |
| `samples/NivaraClassifier/README.md` | benchmark commands, measured table, heuristic |

`BenchStats` sketch:

```csharp
internal readonly record struct TimingSummary(double MedianMs, double MinMs, double MaxMs, long Ticks);

internal static class BenchStats
{
    internal static TimingSummary Summarize(long[] samples, double ticksToMs);
    internal static int Iterations(long perIterTicks, long budgetTicks, int min, int max);
    internal enum Verdict { Faster, Slower, WithinNoise }
    internal static Verdict Classify(double ratio, double noiseBand);
}
```

## Blast radius

- `samples/NivaraClassifier/Program.cs` — CLI surface. Nothing outside the sample references
  `Options`; `tests/Nivara.Tests` references the project only for `DataGenerator`
  (`OnnxExportClassifierTests`).
- New `samples/NivaraClassifier` types — no `src/` change, so no library consumer is affected.
- `NivaraClassifier.csproj` — adds an assembly attribute only.
- `tests/Nivara.Tests` — two new fixtures; both are pure/fast, no wall-clock assertions, so
  neither needs `[Category("Performance")]`.
- `src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs`, `GpuTextConvClassifier.cs` — **read only**.
- `docs/` — untouched; `DocsCoveragePins` stays at its current values.

## Verification

1. `dotnet build Nivara.slnx -c Release`
2. `dotnet run -c Release --project samples/NivaraClassifier -- --benchmark` (~3–5 s)
3. Same with `--silk` on this machine (Intel Iris Xe iGPU)
4. `dotnet run --project samples/NivaraClassifier -- --benchmark` in Debug → refusal, exit 2
5. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~BenchStatsTests|FullyQualifiedName~ClassifierCliArgumentTests"`; time the spawn case
6. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
7. Targeted Docs namespace run if any `docs/` file is touched (it should not be)

## Planned commits

1. `Add BenchStats timing helpers with unit tests` — the pure statistics, plus its fixture.
2. `Add --benchmark mode to NivaraClassifier` — legs, table, CLI wiring, guard, help,
   `InternalsVisibleTo`, CLI test.
3. `docs: benchmark NivaraClassifier Silk vs CPU in README` — measured table + heuristic.

## GitHub issues log

(nothing captured yet)

## Reminder

As each task executes: if deferred work or a concern appears that is outside this plan,
create a GitHub issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record
its number here. Do not rely on memory and do not wait until the end.
