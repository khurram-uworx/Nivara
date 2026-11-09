# Plan: finish #1012 partial/frozen Silk training (remaining items)

## Problem

Core `SilkDistilBertTrainer` partial/frozen behavior is implemented and parity-tested,
but the feature is not reachable from the sample, not documented, and not profiled.

## Proposed changes

1. Add `--trainable-layers` CLI to `samples/NivaraFineTuning`.
2. Add partial-mode support/output to `SilkDistilBertTrainingProfileProbe`.
3. Document the API/workflow in `docs/SILK.md` and auto-diff/Laya partial-training guidance.
4. Run targeted parity tests and whitespace gate.
5. Run G2 reviews, then PR.

## Verification

- Build Extensions and FineTuning sample
- targeted SilkDistilBertTrainerTests
- whitespace verification
- profile before/after (human confirmation before long run)
