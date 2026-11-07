# Plan: refactor(samples): simplify RunTrainSilk training loop (issue #966)

## Problem
`RunTrainSilk` calls `trainer.Train(..., epochs: 1)` inside an outer `for (epoch...)` loop, while
`SilkDistilBertTrainer.Train` supports multi-epoch internally. Loss reporting and shuffle seeding
are split awkwardly between trainer and CLI.

## Goal
Pass `cli.Epochs` to a single `trainer.Train(..., epochs: cli.Epochs, ...)` call, and have the CLI
handle dev eval + model materialization per epoch via a per-epoch hook.

## Proposed changes

### 1. `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`
Add optional `onEpochCompleted` callback to `Train`:
`Action<int, int, float, TimeSpan>? onEpochCompleted = null`
Invoke after each epoch's result is recorded: `onEpochCompleted?.Invoke(epoch + 1, epochs, results[^1].Loss, stopwatch.Elapsed);`

### 2. `samples/NivaraFineTuning/Program.cs` (`RunTrainSilk`)
Replace outer loop with single call. Move per-epoch bookkeeping into callbacks:
- `onBatchCompleted`: detect epoch change, reset accumulators, print epoch header, accumulate, print batch line.
- `onEpochCompleted`: print training complete + steady-state ms/batch, materialize weights, Eval, Evaluate, print dev, Train().

Seed: use `1` (see decisions). Moments persist across epochs (trainer internal).

## Blast radius
- Trainer Train signature gains optional trailing param (backward compatible).
- Sample only consumer of multi-epoch Silk training in CLI.
- Tests: `SilkDistilBertTrainerTests` uses Train with epochs=8, onBatchCompleted not used; unaffected.

## Verification
- `dotnet build Nivara.slnx`
- Targeted Gpu tests: `SilkDistilBertTrainerTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan #966 in TODO.md
2. feat(silk): add onEpochCompleted hook to SilkDistilBertTrainer.Train
3. refactor(samples): simplify RunTrainSilk to single Train call
4. docs: remove TODO.md — plan executed

## GitHub issues log
- None yet.

## Decisions for human (G1)
1. Seed handling: current uses `seed: epoch` per call (different RNG each epoch). Single call uses one Random. Propose `seed: 1`. Confirm or adjust.
2. Optimizer moments: current resets per epoch (because Train resets at start). Single call persists. Propose persist (more correct). Confirm.
