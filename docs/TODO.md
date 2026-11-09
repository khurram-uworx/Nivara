# Plan: feat(silk): opt-in partial/frozen training for BERT/Laya fine-tuning (#1012)

## Problem

Silk's `SilkDistilBertTrainer` trains every tracked parameter every step, including
frozen-prone backbone layers, embedding tables, and lower transformer layers. For
fine-tuning use cases (DistilBERT SST-2, Laya decision head on frozen ModernBERT),
only the head and perhaps the top N encoder layers need gradients. Training all
layers wastes backward compute and optimizer state/motion traffic.

## Proposed changes

1. `SilkBertWeights` gains an optional `trainableLayerCount` (default `-1` => all).
   - `SilkParam.Trainable` flag added; head and top-N encoder layers are trainable;
     lower layers and (in partial mode) embeddings are frozen.
   - Keep gradient/momentum buffers allocated for parity; AdamW skips frozen.
2. `SilkDistilBertTrainer` gains `trainableLayerCount = -1` and `freezeEncoder` API.
   - `BackwardChain` runs classifier head always; loops encoder only for trainable
     top layers; skips embedding backprop in partial mode.
   - `AdamUpdateAll` / `ZeroMoments` filter to `Trainable` params.
   - Aliases: `NumTrainableLayers`, `TrainableLayers`.
3. Sample `NivaraFineTuning` gets `--trainable-layers` passthrough.
4. Docs: workflow note for CPU AutoDiff/Laya partial training via `requiresGrad: false`.
5. Tests: existing parity unchanged; targeted partial parity and phase reduction checks.

## Blast radius

- `SilkBertWeights.cs`, `SilkDistilBertTrainer.cs`: trainer/backward logic.
- `NivaraFineTuning/Program.cs` CLI parsing.
- Parity tests in `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerTests.cs`.

## GitHub issues log

- (none yet)

## Verification

- Build Nivara.Extensions and Nivara.Tests
- Targeted SilkDistilBertTrainer tests
- format whitespace gate
- hidden parity gates for trainable subset remain green

## Planned commits

1. docs: plan
2. feat(silk): Trainable flag + partial weight constructor
3. feat(silk): partial/frozen backward and Adam
4. feat(sample): --trainable-layers
5. docs: workflow/API note
6. tests: targeted parity for partial mode
7. docs: remove TODO.md