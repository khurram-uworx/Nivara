# Plan: train and calibrate the Laya action head from routing feedback (#1205)

## Problem

Issue #982 trained the Laya relevance-choice path from labeled retrieval feedback via
`LayaDecisionHead.ForwardChoiceLogits`, leaving the `act_head` weights unchanged. The act head still
reads ~1.0 on the shipped checkpoint, and `LayaCalibration.Decode` documents `ActionProbability` as
an upstream limitation callers must not gate on. There is no differentiable act path and no trusted
action-label contract, so routing behavior cannot be trained or measured.

## Scope

- Define trusted action labels for routing outcomes (answer/proceed vs. abstain/fallback).
- Train the Laya act-head path with Nivara AutoDiff on frozen ModernBERT features.
- Measure baseline vs. trained routing behavior on held-out examples with calibration metrics.
- Refit/validate calibration separately from the fixed pretrained choice temperatures.

Non-goals: no ModernBERT encoder fine-tuning; no Silk trainer work (#1148); no changes to the
relevance-choice training path (#982); no joint representation learning (tracked separately).

## Proposed changes

### Core model (src/Nivara.Extensions/AI/LayaHeadModel.cs)

Add `ForwardActLogits(hidden, type, markerPositions, validLength)` (plus `ForwardActionLogits` alias)
returning `ReverseGradTensor<TModel>` shape `[1, ActionClassCount]`. It reuses `ScoreChoices`, computes
the same four act features via `LayaHeadScoring`, detaches the pooled row and feature row before
concatenation, and runs only `actInput`/`actOut`. Gradients therefore reach only the act-path parameters.

### Core tests (tests/Nivara.Tests/AutoDiff/LayaDecisionHeadTests.cs)

Add a gradient-isolation test mirroring `ForwardChoiceLogits_PreservesInferenceValuesAndTracksOnlyChoicePath`:
shape, softmax class 0 matches `Forward(...).ActionProbability`, and after backward
`typeEmb`/head layers/scorer grads are null while `actInput`/`actOut` grads are non-null.

### Sample action-label contract (samples/NivaraChat/Helpers)

- `AdaptiveActionExample` record with frozen features + `ActionTarget` (0 = proceed/answer, 1 = abstain/fallback).
- Builder mapping trusted routing outcomes to labels.
- `AdaptiveActionTraining` with `UpdateAct`, `TrainActAndCompare`, `EvaluateAct`, using only act parameters.
- Metrics/calibration records and baseline-vs-trained comparison.

### Calibration

Keep action calibration distinct from choice temperatures; fit/validate a separate scale on act logits.

### Docs

Update `docs/LAYA.md` to distinguish base-checkpoint act_head saturation from a trained adaptive act head.

## Blast radius

- `LayaDecisionHead<TModel,TWeight>` gains one public method + alias.
- No changes to choice path, encoder, storage, or persistence format.
- Existing tests remain valid; new tests added.
- DocsCoveragePins may need updates if docs/citations change.

## Planned commits

1. docs: plan #1205 in TODO.md
2. feat: add ForwardActLogits to LayaDecisionHead
3. test: gradient isolation for act path
4. feat: add adaptive action-label training helpers
5. docs: update LAYA.md act-head caveat
6. docs: remove TODO.md — plan executed

## GitHub issues log

- [ ] #1205 — train and calibrate the Laya action head from routing feedback (this issue)
- [ ] #1260 — optional joint representation learning for Laya act + choice paths (created during #1205 execution)
