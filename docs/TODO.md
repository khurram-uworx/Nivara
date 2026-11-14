# Issue #982 — train and update the Laya decision head

**Status:** in progress · **Tracker:** https://github.com/khurram-uworx/Nivara/issues/982 · **Branch:** `khurram/982`

## Problem and confirmed direction

Train and incrementally update the Laya relevance-choice path using labeled retrieval examples over frozen ModernBERT features. Persist updated head weights and verify exact reload. No ModernBERT training, core Nivara changes, Silk training, or ONNX work.

The current branch already contains the #981 decision-layer commits (`ae879571`, `b1fb2c8e`); #981 remains open until the combined work is tested and landed. The user confirmed #981 and #982 work will land together on `main`, and the issues close after testing.

The current `LayaDecisionHead.Forward` copies choice logits to `float[]` and builds detached host-side features. #982 therefore adds a differentiable choice-forward API in `Nivara.Extensions`, sharing the canonical head/scorer computation while preserving inference behavior. Training is limited to the type embedding, Laya head layers, and relevance scorer. Keep the act-head weights unchanged because the feedback labels do not contain action labels; do not gate on `ActionProbability`.

## Operational choices

- Binary labels: grade `0` is negative; grades `1` and `2` are relevant. Use only resolved #980 labels; invalid or unresolved examples must not silently become negatives.
- Feedback batch threshold: 10 examples, matching `OnlineLearningMode`.
- Optimizer: Adam at `0.0005`, five epochs over each completed feedback batch, matching the existing NivaraChat incremental-training defaults.
- Calibration: retain the existing temperature values for #982; report baseline-versus-trained choice/routing metrics, and do not claim post-update confidence remains calibrated. Broader evaluation is #983.
- Missing files: load the pretrained base Laya head when no adaptive file exists. If both adaptive and base weights are absent, preserve the current explicit retrieval-only fallback.
- Persistence: save head weights with `ModelSerializer.Save/Load`; recreate optimizer state on process restart. Optimizer checkpoint continuation is not required by #982.
- Feature extraction stays outside `GradientUtils.Grad()`; only the choice-path head forward/loss/backward/update is tracked.

## Blast radius and grounding

- `src/Nivara.Extensions/AI/LayaHeadModel.cs`: add differentiable choice logits while preserving existing `Forward`.
- `samples/NivaraChat/Helpers/AdaptiveTrainingData.cs` (new): construct frozen-feature examples and apply the verified grade mapping.
- `samples/NivaraChat/Helpers/AdaptiveTraining.cs` (new): deterministic baseline/evaluation and batched AutoDiff training.
- `samples/NivaraChat/Helpers/AdaptiveFeedbackCollector.cs` (new): collect labeled feedback, update at threshold, and persist after success.
- `samples/NivaraChat/Helpers/LayaRetrievalDecisionHead.cs`: load adaptive weights over the pretrained baseline; retain explicit fallback if both model files are absent.
- `tests/Nivara.Tests/AutoDiff/LayaDecisionHeadTests.cs` and new tests under `tests/Nivara.Tests/Samples/`: differentiable output, label/feature reproducibility, batch lifecycle, baseline comparison, and bit-for-bit state/decision reload.
- Existing source evidence: `LayaDecisionHead.Forward` (`src/Nivara.Extensions/AI/LayaHeadModel.cs:288-362`); `LayaCalibration.Decode` action-probability caveat (`src/Nivara.Extensions/AI/LayaCalibration.cs:233-263`); `GradientUtils.Grad()` (`src/Nivara/AutoDiff/Utilities/GradientUtils.cs:12-29`); `ModelSerializer.Save/Load` (`src/Nivara/AutoDiff/Serialization/ModelSerializer.cs:29-56`).
- Relevant constraints: ADR-001/002/003; `AGENTS.md` inference-default and explicit `Grad()` requirements; .NET 11 projects. The choice path uses existing rank-2/fused operations; no new rank-N primitive.
- Code-memory and Microsoft Learn MCPs are unavailable in this environment. Direct source and official Microsoft Learn .NET 11 pages are recorded in the session plan at `C:\Users\khurram\.copilot\session-state\9d2089fa-b576-4803-8d46-0409c3c7b6e1\plan.md`.

## Execution order

1. Add the differentiable training-forward API and tests. Preserve exact inference output and verify the relevant choice-path parameters receive gradients.
2. Add deterministic frozen-feature training examples and baseline/holdout metrics. Keep ModernBERT outside the graph.
3. Implement 10-example feedback batching and the five-epoch Adam update at 0.0005. Make failed updates explicit and do not discard unconsumed examples.
4. Add adaptive save/restore over the pretrained head, exact parameter/decision reload tests, and the explicit both-files-missing fallback.
5. Expose the updater for #983 without taking over its end-to-end loop or full evaluation.

Tasks are sequential because the sample data contract depends on the new differentiable API and persistence depends on the updater. No parallel implementation.

## Verification

- Build after each logical code step: `dotnet build Nivara.slnx`.
- Targeted Release coverage:
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.LayaDecisionHeadTests|FullyQualifiedName~Nivara.Tests.Samples.AdaptiveTrainingTests|FullyQualifiedName~Nivara.Tests.Samples.AdaptiveFeedbackCollectorTests"`
- Serialization regression:
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.SerializationTests"`
- Include focused #981 routing tests in the same relevant sample-test run if added; do not close #981/#982 until the combined target tests pass.
- Before final review/landing: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`. No full test suite without the required explicit approval.

## Planned commits

1. `docs: plan adaptive Laya head training in TODO.md`
2. `feat(ai): expose differentiable Laya choice logits`
3. `feat(samples): train Laya head from labeled retrieval examples`
4. `feat(samples): persist adaptive Laya head updates`
5. After both branch reviews pass, remove this plan in its own commit.

## GitHub issues log

- #981 — decision-layer implementation is already committed on this branch; validate with #982, then close after testing/landing.
- #982 — current training/update/persistence implementation; close only after all acceptance criteria and targeted tests pass.
- #983 — downstream end-to-end feedback loop and broader evaluation; remains outside this implementation.
- #1148 — separate Silk Laya head trainer; remains outside this implementation.
- #1205 — follow-up to train/calibrate the act head from explicit routing/action labels; created during execution after confirming relevance labels do not supervise that branch.

As each task executes, create a tracked GitHub issue immediately for any new deferred concern outside this scope and add it to this log.
