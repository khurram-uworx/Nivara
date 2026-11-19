# Plan: Issue #1340 — Silk token classifier for NivaraChat entity model

## Problem
Entity (token-level) classifier has no Silk inference/training path. `EntityTrainer.Train` has no `useSilk`, `EntityTextModel` has no silk field, modes never attempt a Silk entity runner, `TrainingMode` reports "CPU for entity". Core ships only document-level Silk classifiers; no `GpuTokenClassifier`.

## Proposed changes
1. Core primitive `GpuTokenClassifier` (src/Nivara.Extensions/Silk/GpuTokenClassifier.cs) mirroring GpuTextLinearClassifier but token-level: no pooling, per-token logits, per-token labels.
2. Adapter `SilkTokenClassifier` (samples/NivaraChat/Helpers/SilkTokenClassifier.cs).
3. `EntityTrainer.Train` accepts `useSilk`.
4. `EntityTextModel` accepts optional Silk runner.
5. `NivaraToolFunctions` entity silk support.
6. Modes wiring (AgentsMode, ToolsMode, QwenMode, TrainingMode).

## GitHub issues log
(none yet)

## Verification
- Build Nivara.Extensions and NivaraChat.
- Targeted tests.

## Planned commits
- docs: plan
- feat(silk): add GpuTokenClassifier
- feat(chat): SilkTokenClassifier adapter
- feat(chat): EntityTrainer useSilk
- feat(chat): EntityTextModel silk runner
- feat(chat): NivaraToolFunctions entity silk
- feat(chat): wire entity silk into modes + reporting