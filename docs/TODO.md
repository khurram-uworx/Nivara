# Plan: issue #1003 — paired CPU vs Silk baseline for DistilBERT fine-tuning shapes

## Problem

Issue #961 allows documenting that the CPU (reverse-mode) path is the better fit at
B≤8 rather than spending on optimisation, but we only have device per-batch numbers for
B=2/4/8, L=128. There is no paired CPU measurement for the same shapes, and the CPU
sample path does not yet print a steady-state ms/batch (first batch excluded) the way
`RunTrainSilk` does.

## Proposed changes

1. `samples/NivaraFineTuning/Program.cs` — `RunTrain`: record each batch's elapsed time
   and print `Steady-state ms/batch` at epoch end (batches 2..N averaged), matching
   `RunTrainSilk`'s wording so the two runs are directly comparable.
2. `tests/Nivara.PerformanceTests/CpuDistilBertTrainingProfileProbe.cs` (new) +
   `--cpu-distilbert-train-profile` flag — shape- and workload-matched counterpart to
   `SilkDistilBertTrainingProfileProbe`: real DistilBERT weights, synthetic tokens at
   B=2/4/8, L=128, 1 warmup + 5 timed batches, median reported. Reuses the same RNG
   and label scheme as the Silk probe.
3. Wire the new flag into `PerformanceTests/Program.cs` ParseArgs/Main.
4. Run both probes/samples (Release) to obtain paired numbers, then update
   `docs/SILK.md` "Performance benchmarks" with the paired table + metadata.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet build samples/NivaraFineTuning`
- `dotnet build tests/Nivara.PerformanceTests`
- measurements reported below; gradient-parity gate `SilkDistilBertTrainerTests` unchanged
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` (note: Debug skip irrelevant)

## Planned commits

1. docs: plan #1003 in TODO.md
2. feat(samples): per-batch steady-state reporting in RunTrain
3. feat(perf): CPU DistilBERT training profile probe
4. docs: paired CPU vs Silk baseline table (filled from measurements)
5. docs: remove TODO.md — plan executed

## Blast radius

- Touches sample + performance probe only; no public library API. Downstream callers
  of `RunTrain` output change only (added steady-state line). PerformanceTests gains a
  new flag following existing `--silk-distilbert-train-profile` pattern.
- Risk: synthetic probe must stay workload-matched to Silk probe (weights, shapes,
  optimizer hyperparameters, L=128).

## GitHub issues log

- (none yet)
