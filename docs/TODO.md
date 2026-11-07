# Plan: issue #963 — per-batch timing in RunTrainSilk

## Problem

`RunTrainSilk` delegates the whole epoch to `SilkDistilBertTrainer.Train`, which loops over
mini-batches internally and only returns an epoch-level `SilkTrainEpoch` (loss + total elapsed).
There is no per-batch timing or per-batch loss to print, and no way to exclude the JIT-warmup
batch from a steady-state average. CPU path (`RunTrain`) prints per-batch times; Silk path does not.

## Proposed changes

### 1. `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs` (minimal, backward-compatible)

Add an optional progress callback to `Train`:

```csharp
public IReadOnlyList<SilkTrainEpoch> Train(
    int[] tokenIds, float[] attentionMask, int[] labels,
    int batchSize, int seqLen, int epochs, float learningRate, float weightDecay, int seed,
    Action<int, int, int, float, TimeSpan>? onBatchCompleted = null)
```

Inside the existing `for (b ...)` loop, wrap each batch in a `Stopwatch`, compute `batchLoss` once,
and invoke the callback after `AdamUpdateAll`:

```csharp
var batchSw = System.Diagnostics.Stopwatch.StartNew();
...
float batchLoss = SilkLoss.CrossEntropy(batchLogits, batchLabels, n, numClasses);
lossSum += batchLoss;
...
batchSw.Stop();
onBatchCompleted?.Invoke(epoch + 1, b + 1, batchesPerEpoch, batchLoss, batchSw.Elapsed);
```

Training math, shuffling, optimizer, moments, and return type are unchanged. Default `null`
preserves all existing callers.

### 2. `samples/NivaraFineTuning/Program.cs` — `RunTrainSilk`

Pass a callback that:
- tracks running loss sum per epoch (reset when epoch changes),
- prints `  Batch {batch}/{total} [{elapsed.TotalSeconds:F1}s] - loss: {runningAvg:F4}`,
- records each batch's elapsed time in a list.

After `Train` returns, compute steady-state ms/batch = average of recorded elapsed (excluding the
first batch) * 1000, and print `  Steady-state ms/batch: {steady:F1}`.

## Verification steps

- `dotnet build Nivara.slnx -c Release`
- Targeted tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkDistilBertTrainer"` (GPU-gated, skips without OpenCL)
- Acceptance run on GPU box: `dotnet run -c Release --project samples/NivaraFineTuning -- --mode train --silk --max-examples 25 --epochs 1`

## Planned commits

1. docs: plan #963 in TODO.md
2. feat(silk): add optional per-batch progress callback to SilkDistilBertTrainer.Train
3. feat(samples): per-batch timing in RunTrainSilk
4. docs: remove TODO.md — plan executed

## Blast radius

- `SilkDistilBertTrainer.Train` signature gains a trailing optional parameter (source-compatible).
- `RunTrainSilk` output gains per-batch lines and a steady-state line.
- No change to kernels, math, or public trainer surface beyond the callback.

## GitHub issues log

- (none yet)
