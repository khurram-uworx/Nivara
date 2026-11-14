# Issue #982 — train and update the Laya decision head

**Status:** implementation complete; G2 review in progress · **Tracker:** https://github.com/khurram-uworx/Nivara/issues/982 · **Branch:** `khurram/982`

## Problem and confirmed direction

Train and incrementally update the Laya relevance-choice path using labeled retrieval examples over frozen ModernBERT features. Persist updated head weights and verify exact reload. No ModernBERT training, core Nivara changes, Silk training, or ONNX work.

The current branch already contains the #981 decision-layer commits (`ae879571`, `b1fb2c8e`); #981 remains open until the combined work is tested and landed. The user confirmed #981 and #982 work will land together on `main`, and the issues close after testing.

The current `LayaDecisionHead.Forward` copies choice logits to `float[]` and builds detached host-side features. #982 therefore adds a differentiable choice-forward API in `Nivara.Extensions`, sharing the canonical head/scorer computation while preserving inference behavior. Training is limited to the type embedding, Laya head layers, and relevance scorer. Keep the act-head weights unchanged because the feedback labels do not contain action labels; do not gate on `ActionProbability`.

The #981 inference path currently tokenizes `query` as both the question and the Laya state; it never includes the retrieved chunk's content in ModernBERT input (`LayaRetrievalDecisionHead.cs:112-125`). Correct this before training so the relevance labels supervise actual query/chunk pairs, and pin the prompt behavior with a sample test.

## Operational choices

- Binary labels: grade `0` is negative; grades `1` and `2` are relevant. Train on the resolved judge labels from #980; do not use leakage-prone auto/heading-derived labels for the baseline comparison. Invalid or unresolved examples must not silently become negatives.
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

1. Fix Laya prompt construction to encode retrieved chunk content and add a regression test that different chunks produce different prompt states for the same query.
2. Add the differentiable training-forward API and tests. Preserve inference output and verify intended choice-path parameters receive gradients.
3. Add deterministic frozen-feature examples from judge labels and baseline/holdout metrics. Keep ModernBERT outside the graph and train/eval splits query-disjoint.
4. Implement 10-example feedback batching and the five-epoch Adam update at 0.0005. Make failed updates explicit and do not discard unconsumed examples.
5. Add adaptive save/restore over the pretrained head, exact parameter/decision reload tests, and the explicit both-files-missing fallback.
6. Expose the updater for #983 without taking over its end-to-end loop or full evaluation.

Tasks are sequential because the sample data contract depends on the new differentiable API and persistence depends on the updater. No parallel implementation.

## Verification

- Build after each logical code step: `dotnet build Nivara.slnx`.
- Targeted Release coverage:
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.LayaDecisionHeadTests|FullyQualifiedName~Nivara.Tests.Samples.AdaptiveTrainingTests|FullyQualifiedName~Nivara.Tests.Samples.AdaptiveFeedbackCollectorTests"`
- Serialization regression:
  `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.SerializationTests"`
- Include focused #981 routing tests in the same relevant sample-test run if added; do not close #981/#982 until the combined target tests pass.
- Before final review/landing: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`. No full test suite without the required explicit approval.

## Implementation commits (chronological)

1. `b1fb2c8e`, `ae879571` — existing #981 decision-layer commits retained on this branch
2. `e5aa1612` — `fix(samples): include chunk content in Laya decisions`
3. `d8d1b45d` — `feat(ai): expose differentiable Laya choice logits`
4. `42c204a4` — `feat(samples): train Laya head from labeled retrieval examples`
5. `c5cc7243` — `feat(samples): persist adaptive Laya head updates`
6. `d32cbad5` — `style: fix adaptive training test formatting`
7. `b4aed006` — `test(samples): cover adaptive RAG decision routing`
8. `b88de9f0` — `test(samples): assert bit-exact decoded adaptive decisions`
9. `9fbaa3ba` — `test(samples): assert adaptive update changes choice logits`

## Implementation and verification status

- [x] #981 prompt correction: the shared prompt includes query instructions and retrieved chunk content; the focused regression test distinguishes chunk inputs for the same query.
- [x] Differentiable choice logits: inference still uses the shared head/scorer computation; tests cover exact inference parity and gradients only through the choice path.
- [x] Training data and updater: resolved judge grades map to binary choice targets; frozen features are extracted outside `Grad()`; query splits are deterministic and disjoint; training updates the choice parameters only.
- [x] Feedback and persistence: updates wait for a complete 10-example batch, save atomically, restore the prior in-memory head on failure, and retain buffered examples when an update/save fails. Adaptive state is loaded over baseline weights when present.
- [x] G2 test-coverage review: added deterministic accepted-choice/fallback tests without model files and bit-exact decoded-decision assertions after persistence reload.
- [x] `dotnet build Nivara.slnx` — passed; one existing NU5104 warning for the prerelease Parquet.Net dependency.
- [x] Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` — passed after the formatting correction.
- [x] `dotnet test tests\Nivara.Tests\Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.AutoDiff.LayaDecisionHeadTests|FullyQualifiedName~Nivara.Tests.AutoDiff.LayaCalibrationTests|FullyQualifiedName~Nivara.Tests.AutoDiff.SerializationTests|FullyQualifiedName~Nivara.Tests.Samples.Adaptive|FullyQualifiedName~Nivara.Tests.Samples.LayaRetrievalDecisionHeadTests|FullyQualifiedName~Nivara.Tests.Samples.RagDecisionRoutingTests"` — passed, 68/68; covers Laya head/calibration/serialization, chunk-conditioned prompt/routing, adaptive data/training, and feedback persistence.
- [x] G2 branch-wide and plan-to-implementation reviews — passed; the discovered routing-test and decoded-decision verification gaps were closed in focused test commits.
- Full solution test suite not run. End-to-end training with judge labels was not run because `samples/data/nivarachat/eval/relevance.tsv` is absent; the local Laya checkpoint exists. Synthetic deterministic tests cover training and persistence without downloading model weights.
- No push or issue closure has occurred; #981 and #982 remain open pending combined landing.

## GitHub issues log

- #981 — decision-layer implementation is already committed on this branch; validate with #982, then close after testing/landing.
- #982 — current training/update/persistence implementation; close only after all acceptance criteria and targeted tests pass.
- #983 — downstream end-to-end feedback loop and broader evaluation; remains outside this implementation.
- #1148 — separate Silk Laya head trainer; remains outside this implementation.
- #1205 — follow-up to train/calibrate the act head from explicit routing/action labels; created during execution after confirming relevance labels do not supervise that branch.

As each task executes, create a tracked GitHub issue immediately for any new deferred concern outside this scope and add it to this log.
