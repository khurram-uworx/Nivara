# Plan: issue #1065 — Samples: replace hand-counted accuracy with Nivara.Metrics

## Problem

Three sample sites hand-count classification accuracy (and one reinvents the per-class
breakdown) instead of using `Nivara.Metrics`. Samples are the module's only real-world
callers, so using it there both demonstrates the API and surfaces friction while the
surface is still soft.

Sites:
1. `samples/NivaraClassifier/Program.cs` — `int correct` loop, then `(double)correct / testLabels.Length`.
2. `samples/NivaraChat/Training/IntentTrainer.cs` — `classCorrect[5]` / `classTotal[5]`.
3. `samples/NivaraChat/Training/AgentsValidatorTrainer.cs` — `int correct` loop.

## Proposed changes

Add `using Nivara.Metrics;` where needed.

- Classifier: build `predicted`/`actual` `int[]` while encoding/predicting, create
  `NivaraColumn<int>` pair, call `ConfusionMatrixFrame`, derive
  `correct = Sum(Correct)`, `accuracy = correct / testLabels.Length`.
  Apply to all three prediction branches in `RunTraining` (conv/Silk, Silk-linear, CPU).
- IntentTrainer: collect predicted/actual arrays, call `PerClassMetricsFrame`, derive
  overall accuracy as `Sum(Tp) / testLabels.Length`, and per-intent counts as
  `Tp` and `Tp + Fn`. Print unchanged lines; fall back to 0 when a class is absent
  from the frame.
- AgentsValidatorTrainer: same as classifier.

## Verification

- `dotnet build` the two sample projects in Release.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Confidence checks: manually inspect diff; numbers unchanged by construction
  (same denominator, same correctness predicate).

## Planned commits

1. `docs: plan issue 1065 in TODO.md`
2. `samples: use Nivara.Metrics in NivaraClassifier`
3. `samples: use Nivara.Metrics in IntentTrainer and AgentsValidatorTrainer`
4. `docs: remove TODO.md — plan executed` (after G2)

## Blast radius

- sample-only code; no core/test/API changes. `DocsCoveragePins` unaffected
  (no docs change, no citation change).
- Downstream: console output strings and fractional accuracy identical.

## GitHub issues log

(none created yet)
