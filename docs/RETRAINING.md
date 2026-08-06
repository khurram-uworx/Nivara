# Online & Incremental Retraining in Nivara

`docs/AUTODIFF.md` documents the training stack — modules, loss functions,
optimizers, `DataLoader`, `TrainingLoop`, and serialization — for training from
scratch. This guide covers what happens *after* deployment: keeping a deployed
model learning from new validated examples as they arrive, entirely in .NET.

The idea, stated up front:

> Retrain incrementally, don't retrain from scratch. Warm-start the deployed
> weights, add the new validated examples to the original dataset, fine-tune at a
> lower learning rate, and carry the optimizer's state across sessions via a
> checkpoint. A fine-tune, not a re-train.

It assumes you have read `docs/AUTODIFF.md` and covers only what that document
does not.

## When you need this

- **Distribution drift** — the serving inputs shift and accuracy decays.
- **Feedback loops** — human review, LLM correction, or A/B results produce
  `(input, corrected-label)` pairs you can trust.
- **Scheduled refreshes** — periodic retraining on accumulated data without the
  cost (and forgetting) of a full re-train.

## The pattern

| Step | What you do | Nivara API |
|------|-------------|------------|
| 1. Warm-start | Load the deployed weights and optimizer state | `TrainingLoop.LoadCheckpoint` (weights-only fallback: `ModelSerializer.Load`, AUTODIFF.md §Serialization) |
| 2. Buffer | Accumulate validated `(input, label)` examples | your code — a list, frame, or file |
| 3. Append | Retrain on original data **plus** the buffer | `TensorDataset` + `DataLoader` (AUTODIFF.md §Training) |
| 4. Fine-tune | Lower learning rate, few epochs | `TrainingLoop` + `Continue(additionalEpochs)` |
| 5. Persist | Re-save weights + a fresh checkpoint | `ModelSerializer.Save` + `TrainingLoop.SaveCheckpoint` |

## Why optimizer state matters

Adam (and momentum-based SGD) keep per-parameter buffers — `m`, `v`, and the
bias-correction step `t`. A fresh optimizer on every retrain resets those
buffers to zero, so the first epochs are a cold start: the fine-tune wastes
epochs and can disturb a well-tuned model.

A checkpoint persists the optimizer state alongside the weights, so a resume
picks up exactly where training left off. Two properties make this clean:

- **The learning rate is constructor state**, not part of the optimizer state.
  Your resume loop can apply a *lower* fine-tune LR (say `0.0005` vs `0.001`)
  while the restored moments carry over.
- **The epoch counter travels with the checkpoint.** After `LoadCheckpoint`, the
  loop's `MaxEpoch` reflects the saved epoch, so `Continue(additionalEpochs)`
  resumes at `savedEpoch + 1` instead of renumbering from 1.

## Retraining-specific APIs

These are the only members `docs/AUTODIFF.md` does not document. Everything else
(`ModelSerializer`, `Checkpoint<T>`, `DataLoader`, `TensorDataset`, optimizer
registration) is unchanged from that guide.

| API | What it adds |
|-----|--------------|
| `TrainingLoop<T>.SaveCheckpoint(path, epoch, loss)` | Persists weights + optimizer state + epoch (`nivara-ckpt-v2`) |
| `TrainingLoop<T>.LoadCheckpoint(path)` | Restores weights + optimizer state + the epoch counter |
| `TrainingLoop<T>.Continue(int additionalEpochs)` | Resumes training, continuing epoch numbering past the saved epoch |
| `TrainingLoop<T>.Run(int startEpoch = 1)` | Train from an arbitrary starting epoch (used by `Continue`) |

## A self-contained example

Model, data, and loader are built exactly as in `docs/AUTODIFF.md` examples 7
and 12 — the retraining flow adds only the steps below. The first full training
is assumed to have ended with `SaveCheckpoint(...)`, so the deployment is
resumable from day one.

```csharp
// Retrain on original data + new validated examples at a lower LR.
using var optimizer = new Adam<float>(learningRate: 0.0005f);
optimizer.AddParameterGroup(model.GetParameters().Values);

var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 5);

// 1. Warm-start: weights AND optimizer state from the checkpoint.
loop.LoadCheckpoint("churn_checkpoint.json");

// 2. Fine-tune: continues at savedEpoch + 1.
var result = loop.Continue(additionalEpochs: 5);
result.PrintSummary();
// Epoch   6 | Loss:   0.041200 | Batches:  1 | Time: 0.02s
// Epoch  10 | Loss:   0.037100 | Batches:  1 | Time: 0.02s

// 3. Persist: updated weights + a fresh checkpoint for the next round.
ModelSerializer.Save(model, "churn_model.json");
loop.SaveCheckpoint("churn_checkpoint.json", result.Epochs[^1].Epoch, result.Epochs[^1].Loss);
model.Eval();   // back to inference mode
```

Two caveats that are easy to get wrong:

- `TrainingLoop` implements `IDisposable` and disposes the model and optimizer.
  If you keep using the model after training (the normal serving flow), do **not**
  dispose the loop; dispose the model yourself when you're done.
- `LoadCheckpoint` restores both weights and optimizer state, so no separate
  `ModelSerializer.Load` is needed when a checkpoint exists. If you only have a
  weights file (e.g., an older model directory), fall back to
  `ModelSerializer.Load` + a fresh optimizer — just be aware the optimizer
  starts cold.

## Design decisions worth preserving

These trade-offs were validated in a production sample (an LLM-feedback intent
classifier). You should have reasons to deviate.

- **Append, don't replace.** Retraining runs on the original dataset **plus** the
  new examples. Training on feedback alone causes *catastrophic forgetting* of
  the seed distribution.
- **Lower learning rate on retrain.** A small step (`0.0005` vs `0.001`) so new
  examples refine rather than overwrite what the model learned.
- **Keep preprocessing stable across sessions.** For text models, save the
  tokenizer with the model and reload it on retrain; vocab indices must not shift
  between the original dataset and the feedback examples.
- **Buffer, then batch.** Collect validated examples into a buffer and retrain in
  batches when the buffer reaches a threshold, rather than one-example-at-a-time
  updates.

## Where you've seen this in action

- **`EXAMPLES.md` Act 8b** — a worked FraudNet example (the same model from
  Act 8, later in production) covering this exact pattern.
- **`samples/NivaraChat --online-learning`** — the full feedback loop end to end:
  classify → low-confidence input routed to an LLM for a corrected label →
  buffered → incremental retrain → keep serving.
- **`docs/AUTODIFF.md`** — the underlying training stack this guide builds on.
