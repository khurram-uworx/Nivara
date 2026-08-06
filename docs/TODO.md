# TODO — NivaraChat online-learning checkpoint-resume + Act 8b

Branch: `khurram/nivarachat`. Parent: `main` (5b3443c).

## Background / motivation

`samples/NivaraChat/README.md` documents the `--online-learning` mode as using
`Optimizer.StateDict()/LoadStateDict()`, `TrainingLoop.Run(startEpoch)`, and
optimizer-state-in-checkpoints (see `samples/NivaraChat/README.md:296,494-496`).
The actual implementation does **not**: `IntentTrainer.TrainIncremental`
(`samples/NivaraChat/Training/IntentTrainer.cs:90-148`) builds a fresh
`Adam<float>` and runs a new `TrainingLoop`, so Adam's moment buffers never carry
across retrain sessions. The README overstates current behavior.

Fix: implement the checkpoint-resume path so the documented claims are true.
That makes it possible to add **Act 8b (Online training)** to `EXAMPLES.md` as a
checkpoint-resume showcase, backed by a working sample.

## Task 1: Core — restore `_maxEpoch` in `TrainingLoop.LoadCheckpoint`

### Priority

High

### Goal

`TrainingLoop.LoadCheckpoint` restores the epoch counter so `Continue()` resumes
with correct epoch numbering.

### Why this exists

`TrainingLoop.LoadCheckpoint` (`src/Nivara/AutoDiff/Training/TrainingLoop.cs:142-152`)
restores model parameters and optimizer state but leaves `_maxEpoch` untouched.
After a load, `Continue(additionalEpochs)` computes `startEpoch = _maxEpoch + 1`
(`TrainingLoop.cs:125-133`) and re-numbers epochs from 1, making checkpoint
resume reports misleading.

### Scope

- Set `_maxEpoch = checkpoint.Epoch` in `LoadCheckpoint`.
- Keep behavior deterministic for the `Run(int startEpoch)` path.

### Acceptance criteria

- A `TrainingLoop` that saves a checkpoint at epoch N, then `LoadCheckpoint` +
  `Continue(additionalEpochs)`, reports epochs starting at N+1.
- New regression test in `tests/Nivara.Tests`.

### Files likely involved

- `src/Nivara/AutoDiff/Training/TrainingLoop.cs`
- `src/Nivara/AutoDiff/Serialization/Checkpoint.cs` (read-only reference)
- `tests/Nivara.Tests/` (regression test for checkpoint resume)

## Task 2: `IntentTrainer.Train` — save a checkpoint with optimizer state

### Priority

High

### Goal

Full training persists a checkpoint containing weights **and** Adam state.

### Why this exists

Incremental retrain needs optimizer state to resume from. Today only
`model.json` (weights + tokenizer) is saved.

### Scope

- After `loop.Run()` in `IntentTrainer.Train`, call
  `loop.SaveCheckpoint(Path.Combine(saveDir, "intent_checkpoint.json"), lastEpoch, lastLoss)`.
- Keep existing `ModelSerializer.Save(model, ...)` / tokenizer save for
  backward compatibility and the normal `--intent` load path.

### Acceptance criteria

- `--intent-train` produces `models/intent_checkpoint.json` alongside
  `intent_model.json` / `intent_tokenizer.json`.

### Files likely involved

- `samples/NivaraChat/Training/IntentTrainer.cs`

## Task 3: `IntentTrainer.TrainIncremental` — checkpoint-resume path

### Priority

High

### Goal

Incremental retrain loads the checkpoint (weights + Adam moments) and resumes
training with `Continue`, honoring the lower learning rate.

### Why this exists

This is the documented behavior in `samples/NivaraChat/README.md` and the API
story Act 8b will showcase.

### Scope

- Load tokenizer + reconstruct model (unchanged from today).
- Build appended dataset (original seed 500 + feedback buffer) + `DataLoader`
  (unchanged).
- Construct `TrainingLoop` with `Adam<float>` at `lr 0.0005`.
- Call `loop.LoadCheckpoint(checkpointPath)` (restores weights + optimizer state),
  then `loop.Continue(additionalEpochs)` instead of `loop.Run()`.
- Re-save updated `model.json` + updated checkpoint.
- `lr` is constructor state, not part of `StateDict` — the lower LR applies while
  moment buffers carry over (desired fine-tune behavior).

### Acceptance criteria

- A scripted `--online-learning` session triggers a retrain at buffer 10 and the
  resumed run reports epochs continuing after the saved epoch.
- README's `Uses:` line and API table claims match implementation.

### Files likely involved

- `samples/NivaraChat/Training/IntentTrainer.cs`

## Task 4: Verify NivaraChat online-learning end-to-end

### Priority

High

### Goal

The `--online-learning` mode runs with the checkpoint-resume path and behaves
correctly.

### Scope

- `dotnet build Nivara.slnx`.
- `dotnet run --project samples/NivaraChat -- --intent-train`.
- Scripted `--online-learning --ollama` session (or, if no Ollama at runtime,
  verify `TrainIncremental` directly via a small harness/test) confirming epoch
  continuity and loss progression across retrain.

### Acceptance criteria

- Build clean; retrain path observed working; commit.

### Files likely involved

- Build/sample verification only.

## Task 5: Act 8b — Online training in `EXAMPLES.md`

### Priority

Medium

### Goal

Add Act 8b after Act 8 (`EXAMPLES.md:586`) documenting online/incremental
training with the checkpoint-resume pattern.

### Scope

- Title + blurb: warm-start from saved weights, retrain on a small buffer of
  validated examples at a lower LR, re-save — drift correction for deployed models.
- Short PyTorch (Python) counterpart.
- Nivara snippet: `ModelSerializer.Load` + `TextTokenizer.Load` warm start,
  appended feedback dataset → `TensorDataset`/`DataLoader`, lower-LR
  `TrainingLoop`, `SaveCheckpoint`/`LoadCheckpoint`/`Continue` (Adam-moment
  carryover), `ModelSerializer.Save` + `model.Eval()`.
- "What this adds over Act 8" table.
- Cross-link to `samples/NivaraChat --online-learning`
  (`samples/NivaraChat/README.md:259`).

### Acceptance criteria

- Act 8b is consistent with the fixed NivaraChat implementation.

### Files likely involved

- `EXAMPLES.md`

## Notes / decisions taken

- Variant B (checkpoint-resume) chosen over variant A (fresh-optimizer warm start)
  for Act 8b; the NivaraChat fix (Tasks 1-3) makes it real.
- Task 1 is a small, deliberate core change on this branch; it is what makes
  `Continue()` correct after `LoadCheckpoint`.
- Commits are local-only per iterative-commit skill; push/PR is human-controlled.
