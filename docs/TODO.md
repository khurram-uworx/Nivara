# Plan: NivaraChat domain text models gain a Silk inference/training path (#1266)

## Problem

The four domain models (sentiment, entity, validator, intent) are always CPU in
NivaraChat. `--silk` only switches MiniLM embeddings and (for SmolLM/Qwen) the
LLM, so the console "Backend: Silk" line overstates coverage. Core already ships
Silk text classifiers (`GpuTextLinearClassifier`, `GpuTextConvClassifier`) with
training, used by `samples/NivaraClassifier`, but they are not wired into
NivaraChat.

## Proposed changes

### Phase 0 — Truthful reporting
- `samples/NivaraChat/Program.cs` help line: state domain models remain CPU unless wired.
- Trainers/modes note Silk coverage explicitly.

### Phase 1 — Generalize Silk classifiers to N classes (backward-compatible)
- `src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs`: add `numClasses` ctor
  param (default 2); size fc2/logits/backward/Adam moments accordingly; use
  general `SilkLoss.CrossEntropy(...,numClasses)` and `InputGradient(...,numClasses,...)`.
- Same for `GpuTextConvClassifier.cs`.
- `samples/Nivara.Samples/TextClassifierModel.cs`: store `numClasses`; fix
  `SetWeights` fc2 first dim from hardcoded `2` to `numClasses`.

### Phase 2 — Silk training path in trainers (document classifiers)
- `SentimentTrainer`, `ValidatorTrainer`, `AgentsValidatorTrainer`,
  `IntentTrainer`: add `useSilk` param. When set, train via
  `GpuTextLinearClassifier(numClasses=...)`, export CPU layout, populate
  `TextClassifierModel`, save CPU JSON, dispose GPU. CPU fallback on
  `InvalidOperationException`.
- `EntityTrainer` stays CPU (token-level; no Silk token classifier). Documented.

### Phase 3 — Silk inference path
- `ModeHelpers.Load*Model`: add `useSilk` param; when set reconstruct
  `GpuTextLinearClassifier` from loaded CPU weights and return a small adapter
  `SilkTextClassifier` exposing `Predict(int[])` and `Forward`-compatible logits.
- `ModelInferenceHelper`: add Silk overloads of `RunClassifier*` taking the adapter.
- Executors (`SentimentExecutor`, `IntentClassifier`) and `NivaraToolFunctions`:
  accept either CPU model or Silk adapter; branch internally. Keep CPU fallback.

### Phase 4 — Reporting
- `--silk` console output lists each component's backend.

## Verification steps
- Build solution: `dotnet build Nivara.slnx`
- Build sample: `dotnet build samples/NivaraChat/NivaraChat.csproj`
- Targeted tests: GpuText*ClassifierTests, TextClassifierModelTests
- Manual smoke: `--train --silk`, `--intent-train --silk`, `--tools --silk`

## Planned commits
1. docs: plan #1266 in TODO.md
2. docs: truthful --silk reporting (Phase 0)
3. feat(silk): generalize text classifiers to N classes (Phase 1)
4. fix: TextClassifierModel.SetWeights numClasses (Phase 1)
5. feat(sample): Silk training path in NivaraChat trainers (Phase 2)
6. feat(sample): Silk inference path in NivaraChat (Phase 3)
7. docs: --silk component coverage reporting (Phase 4)
8. docs: remove TODO.md

## Blast radius
- Silk classifiers (core Extensions): constructor signature gains optional param
  (backward-compatible via default).
- TextClassifierModel (sample): SetWeights behavior change for numClasses != 2
  (currently broken for N-class anyway).
- Trainers: new optional param; existing callers unaffected.
- ModeHelpers/executors: new optional params; callers updated.
- Tests: GpuText*ClassifierTests construct with default; should still pass.

## GitHub issues log
- (none yet)
