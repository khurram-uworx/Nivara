# Nivara Retraining (Online Learning) — How It Works

Reference for how NivaraChat's `--online-learning` mode performs incremental
retraining of a deployed model. Intended to inform the Act 8b example in
`EXAMPLES.md` and any future docs on continuous learning.

## The problem

Deployed classifiers drift as inputs change. Retraining from scratch is expensive
and discards the serving model's learned behavior. Online learning instead:

1. Warm-starts from the saved model weights.
2. Accumulates validated `(text, label)` examples into a buffer.
3. Periodically retrains on the original dataset **plus** the buffer at a lower
   learning rate (fine-tune, not fresh training).
4. Re-saves the model and keeps serving.

## The feedback loop (`--online-learning`)

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel<float>, 5 classes
    │
    ├── confidence >= threshold ──> Return Nivara classification (no LLM needed)
    │
    └── confidence < threshold  ──> [Ollama LLM]
                                        │
                                        v
                                  LLM provides corrected intent
                                        │
                                        v
                                  Add (text, intent) to training buffer
                                        │
                                        v
                                  Buffer full (retrainThreshold)?
                                        │
                                   yes ──> IntentTrainer.TrainIncremental()
                                        │   loads existing model, trains 5 epochs
                                        │   at lr=0.0005, saves updated model
                                        v
                                  Continue with updated model
```

Mechanics:

- `FeedbackCollector.ClassifyAsync` (`samples/NivaraChat/FeedbackCollector.cs:42`)
  runs the classifier; only when confidence is below the threshold does it call
  the LLM for a corrected intent and append it to the buffer.
- The buffer threshold for the mode is 10 (`samples/NivaraChat/Program.cs:370`);
  the `FeedbackCollector` default is 50.
- When the buffer fills, `FeedbackCollector.Retrain` (`FeedbackCollector.cs:92`)
  flushes it into `IntentTrainer.TrainIncremental`, then the serving process
  rebuilds the collector with the fresh model (`Program.cs:393-402`).

## The retrain step

### Variant A — current implementation: fresh optimizer warm-start

`IntentTrainer.TrainIncremental` (`samples/NivaraChat/Training/IntentTrainer.cs:90`):

1. Load the saved tokenizer (`TextTokenizer.Load`) and reconstruct the model
   architecture; `ModelSerializer.Load` restores the weights.
2. Build the appended dataset: the original seed-500 intent set **plus** the
   feedback buffer, re-tokenized with the *saved* tokenizer (vocab/indices stay
   stable across sessions).
3. Build a `TensorDataset`/`DataLoader` from the combined frame.
4. Construct a **fresh** `Adam<float>` at `lr = 0.0005` and run a new
   `TrainingLoop` for `additionalEpochs` (5).
5. `ModelSerializer.Save` the updated weights and return the model.

Limitation: a fresh optimizer means Adam's moment buffers (`m`, `v`, bias-correction
step `t`) are reset on every retrain. This is the "warm-start weights, cold
optimizer" form — fine for coarse fine-tunes, but it discards training dynamics.

### Variant B — target (this branch): checkpoint resume

Same warm-start + appended dataset, but the optimizer state carries across
sessions via a checkpoint:

1. `IntentTrainer.Train` also calls `TrainingLoop.SaveCheckpoint(...)` after the
   initial full training, persisting weights **and** `Adam<float>.StateDict()`
   (`TrainingLoop.cs:135-140`; checkpoint format `nivara-ckpt-v2`).
2. `TrainIncremental` constructs its loop with the lower-LR Adam, then
   `TrainingLoop.LoadCheckpoint(path)` restores both weights and optimizer state
   (`TrainingLoop.cs:142-152`).
3. `TrainingLoop.Continue(additionalEpochs)` resumes training instead of
   restarting, keeping Adam moments and (with the `_maxEpoch` fix) correct epoch
   numbering.
4. Re-save model + updated checkpoint.

Key property: the learning rate is **constructor state**, not part of
`Optimizer.StateDict()`. So the incremental loop can apply `lr = 0.0005` while
the restored moments carry over — exactly the fine-tuning behavior we want.

## Core APIs involved

| API | Location | Role in retraining |
|-----|----------|--------------------|
| `TrainingLoop.Run(int startEpoch = 1)` | `src/Nivara/AutoDiff/Training/TrainingLoop.cs:81` | Initial training and arbitrary-start runs |
| `TrainingLoop.Continue(int additionalEpochs)` | `TrainingLoop.cs:125` | Resume training after a checkpoint, continuing epoch numbering |
| `TrainingLoop.SaveCheckpoint(path, epoch, loss)` | `TrainingLoop.cs:135` | Persist weights + optimizer state |
| `TrainingLoop.LoadCheckpoint(path)` | `TrainingLoop.cs:142` | Restore weights + optimizer state |
| `Optimizer<T>.StateDict()` / `LoadStateDict(...)` | `src/Nivara/AutoDiff/Optimizer/Optimizer.cs:93,95` | Raw optimizer-state serialization (SGD, Adam, AdamW) |
| `ModelSerializer.Save` / `Load` | `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs:17,28` | Weights-only JSON round-trip (`nivara-ss-v2`) |
| `ModelSerializer.SaveCheckpoint` / `LoadCheckpoint` | `ModelSerializer.cs:68,83` | Checkpoint JSON round-trip incl. optimizer state (`nivara-ckpt-v2`) |
| `Checkpoint<T>` / `ParameterData<T>` | `src/Nivara/AutoDiff/Serialization/Checkpoint.cs` | Checkpoint payload shape (Epoch, Loss, Parameters, OptimizerState) |
| `Adam<float>` | `src/Nivara/AutoDiff/Optimizer/Adam.cs` | Bias-corrected adaptive optimizer; state restored on resume |

## Design decisions worth preserving

- **Tokenizer is fixed across sessions** — saved once, reloaded on retrain, so
  vocab indices never shift between the original dataset and feedback examples.
- **Lower LR on retrain** — `0.0005` vs `0.001` initial; a small step so new
  examples refine rather than overwrite the learned distribution.
- **Appended dataset, not pure feedback** — retraining runs on original data +
  buffer to avoid catastrophic forgetting of the seed distribution.
- **Checkpoint format includes optimizer state** — `Optimizer.StateDict()` gives
  plain `Dictionary<string, T[]>` buffers (Adam: `m`, `v`, `t` per parameter
  group), which serialize directly.

## Files

- `samples/NivaraChat/FeedbackCollector.cs` — LLM fallback + feedback buffer
- `samples/NivaraChat/Training/IntentTrainer.cs` — initial train + `TrainIncremental`
- `samples/NivaraChat/Program.cs` — `--online-learning` orchestration (lines ~350-408)
- `src/Nivara/AutoDiff/Training/TrainingLoop.cs` — loop, `Continue`, checkpoints
- `src/Nivara/AutoDiff/Optimizer/Adam.cs` — optimizer state
- `src/Nivara/AutoDiff/Serialization/ModelSerializer.cs` / `Checkpoint.cs` — persistence

## Relationship to EXAMPLES.md Act 8b

Act 8b ("Online training — keep a deployed model learning") will showcase this
pattern in `EXAMPLES.md` after Act 8: warm-start via `ModelSerializer.Load` +
`TextTokenizer.Load`, appended feedback dataset → `TensorDataset`/`DataLoader`,
lower-LR `TrainingLoop`, and `SaveCheckpoint`/`LoadCheckpoint`/`Continue` for
Adam-moment carryover. It will cross-link to `--online-learning`
(`samples/NivaraChat/README.md:259`). Planned only — not yet written.
