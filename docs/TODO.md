# Plan: ModernBERT head-only fine-tune path (issue #1280)

## Problem

ModernBERT-large is inference-only in the repo: no training path on CPU or Silk.
`SilkDistilBertTrainer` is BERT-shaped and does not cover ModernBERT's hybrid
(full + sliding-window) attention, and there is no AutoDiff model wrapper for
fine-tuning. Goal: add a head-only fine-tune path (frozen encoder, trainable
classification head on `[CLS]`), mirroring the Laya head pattern, so users can
adapt ModernBERT to a downstream task without retraining the encoder.

## User story (showcase)

As a .NET developer, adapt ModernBERT-large to a 4-class intent-routing task
(search / code / chat / tool) by training only a small head on precomputed,
frozen `[CLS]` features on CPU, exporting and reloading the head weights — reusing
the head-only training infrastructure that NivaraChat already uses for Laya, so
no training-loop code is duplicated across samples.

## Proposed changes

### A. Promote shared head-training infra to `samples/Nivara.Samples`
- Move `AdaptiveTraining`, `AdaptiveTrainingData`, `AdaptiveTrainingExample` from
  `samples/NivaraChat/Helpers/` into `samples/Nivara.Samples/`.
- Generalize the training loop so it no longer hard-codes `LayaDecisionHead<float,float>`:
  use a forward delegate `Func<ReverseGradTensor<float>, IReadOnlyList<int>, int, ReverseGradTensor<float>>`.
- NivaraChat keeps thin wrappers (no behavior change).

### B. Add ModernBERT classification head (`samples/Nivara.Samples`)
- `ModernBertClassificationHead<TModel,TWeight>`: Linear(hidden→hidden)+ReLU+Linear(hidden→numClasses)
  on the `[CLS]` row (index 0), following `LayaDecisionHead` Module pattern
  (skipWeightInitialization, GetParameters, Eval/Train).
- `LoadWeights` / export keyed `head.*`.

### C. CPU training + showcase entry point
- `ModernBertHeadTraining` helper: precompute `[CLS]` features via
  `ModernBertEncoder.Forward` (inference, no grad), then head-only Adam training
  on frozen features, exporting the head.
- Add `modernbert train-head` mode in `NivaraInference` printing baseline→trained
  accuracy and asserting encoder weights byte-identical before/after (frozen guarantee).

### D. Acceptance (matches #1280)
- Encoder params unchanged after training (hash check).
- Head params changed; exported head reloads and classifies.
- F32 CPU only this pass.

## Verification steps
- Build `Nivara.Samples`, `NivaraChat`, `NivaraInference`.
- Run targeted tests for affected areas.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Docs snippet gate if docs touched (none planned).

## Planned commits
1. docs: plan #1280 in TODO.md
2. (grounding) — microsoft-learn MCP grounding notes
3. feat(samples): promote adaptive head training to Nivara.Samples (generalized)
4. feat(samples): add ModernBertClassificationHead
5. feat(samples): add ModernBertHeadTraining helper
6. feat(inference): add modernbert train-head mode
7. test: head-only training + frozen encoder + reload
8. docs: update README / samples if needed
9. docs: remove TODO.md — plan executed

## Blast radius
- `samples/Nivara.Samples` gains new files; referenced by NivaraChat + NivaraInference (already).
- `samples/NivaraInference/Program.cs` + new mode file.
- No core library changes. No test changes except SupportedModes allow-list.
- Note: promotion of NivaraChat's Laya-specific `AdaptiveTraining` into Nivara.Samples was deferred;
  the new ModernBERT training lives in Nivara.Samples without duplicating that code.

## GitHub issues log

- [x] #1351 — Silk counterpart follow-up for ModernBERT head training (created while working on #1280).
