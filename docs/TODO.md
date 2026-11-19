# Plan: Issue #1341 — NivaraChat Laya adaptive-head Silk training

## Summary

Wire `SilkLayaHeadTrainer` into `RagAdaptiveEvalMode` so that, under `--silk`,
the adaptive Laya head trains on the OpenCL device when available, falling back
to CPU AutoDiff otherwise. Record which backend trained the head in the
adaptive report. f32 only for this release.

## GitHub issues log

- (none yet)

## Proposed changes

1. `AdaptiveFeedbackCollector`: optional `updateAction` delegate (default CPU).
2. `RagAdaptiveEvalMode`: backend selection probe, `SilkUpdate`, `HeadToCheckpointTensors`.
3. `SilkLayaHeadTrainer`: add `ZeroGradients()`.
4. `AdaptiveReport`: add `HeadTrainingBackend` to Silk record and input, render/JSON.

## Verification

- `dotnet build Nivara.slnx`
- sample build
- run `--rag-adaptive-eval` (cpu) and `--rag-adaptive-eval --silk` if device available.

## Planned commits

1. docs: plan issue #1341 in TODO.md
2. feat: updateAction hook in AdaptiveFeedbackCollector
3. feat: ZeroGradients in SilkLayaHeadTrainer
4. feat: HeadToCheckpointTensors + SilkUpdate + backend selection in RagAdaptiveEvalMode
5. feat: HeadTrainingBackend in AdaptiveReport
6. docs: remove TODO.md