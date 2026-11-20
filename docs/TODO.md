# Plan: Issue #1363 — SilkTinyShakespeareTrainer trainableLayerCount (partial/frozen training)

## Problem
`SilkTinyShakespeareTrainer` always trains all layers. Need `trainableLayerCount` (default all) for partial/frozen training, mirroring `SilkDistilBertTrainer` / `SilkLayaHeadTrainer` via `SilkPartialTrainingPolicy`.

## Proposed changes
- Add `trainableLayerCount` constructor parameter (trailing, default -1 => all layers).
- Add mutable `trainableLayerCount` field and public `TrainableLayerCount` property with clamping.
- Convert `parameters` list to mutable `Param` class with `Trainable` flag; `AddParam` takes trainable bool.
- Mark tok_emb trainable only when all layers; final_norm always; per-layer per policy.
- Backward: gate LM-head tokEmb grad, run layer loop only for trainable top layers, embedding scatter only when all layers.
- AdamUpdateAll filters trainable params.
- ZeroGradients zeros all grads.

## Verification
- Build Extensions.
- Targeted tests.

## GitHub issues log
(none yet)
