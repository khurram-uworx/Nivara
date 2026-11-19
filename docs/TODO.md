# Plan: issue #1275 — NivaraChat online learning Silk training path

## Problem
`--online-learning` retrains the intent classifier on CPU only via `IntentTrainer.TrainIncremental()`,
even when `--silk` is set. Core already has Silk text-classifier training (`GpuTextLinearClassifier.Train`,
wired into `IntentTrainer.Train` by #1266), but online learning never uses it.

## Acceptance criteria
- `--online-learning --silk` either runs incremental retrain on device or prints that online learning
  is CPU-only (no silent fallback).
- Existing incremental retrain behavior (lr 0.0005, 5 epochs, buffer of 10) preserved.

## Proposed changes
1. `IntentTrainer.TrainIncremental`: add `useSilk` (and `seed`) params; on device, seed a
   `GpuTextLinearClassifier` from the saved CPU model, train `additionalEpochs` at lr 0.0005, export
   CPU layout, save model + tokenizer. On OpenCL failure, print CPU-only and fall through to CPU path.
2. `FeedbackCollector`: accept `useSilk` + optional `SilkTextClassifier`; use Silk inference when available;
   pass `useSilk` to `TrainIncremental`.
3. `OnlineLearningMode`: create/dispose `SilkTextClassifier` around load and retrain; pass through to collector.
4. README note (optional).

## Blast radius
- `samples/NivaraChat/Training/IntentTrainer.cs`
- `samples/NivaraChat/Helpers/FeedbackCollector.cs`
- `samples/NivaraChat/Modes/OnlineLearningMode.cs`
- `samples/NivaraChat/README.md` (docs)

## Planned commits
1. docs: plan #1275 in TODO.md
2. feat: Silk incremental retrain in IntentTrainer.TrainIncremental
3. feat: plumb --silk through FeedbackCollector and OnlineLearningMode
4. docs: update README online learning --silk note
5. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)
