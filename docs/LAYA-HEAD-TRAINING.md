# Laya head training — design notes (issue #1103, Phase 2)

## Status

Nivara does **not** currently ship a Silk trainer for the Laya decision head. The head trains
on CPU via AutoDiff (`LayaHeadModel` / `LayaHeadLayer`) on frozen ModernBERT features. This
document captures the design for a future Silk implementation so it is not lost.

## Motivation

The encoder is frozen during Laya head training, so only the small head trains. A Silk trainer
would mirror `SilkDistilBertTrainer` but for the head's parameters and kernels. It is gated
behind the passing DistilBERT/BERT BF16 parity gates (#1135).

## Scope (future)

- New `SilkLayaHeadTrainer` owning head weights + gradients + AdamW moments + activation
  workspace.
- Reuse `SilkLayaHeadGpuRunner` upload/layout and `SilkPartialTrainingPolicy` for the head's
  own layer count (independent of encoder `NumHiddenLayers`).
- Backward kernels needed: ReLU backward, LayerNorm backward (biased, eps 1e-5), attention
  backward (already present for encoder), GEMM NT/TN (present), ColSum (present). New
  kernels only if activation-level BF16 is pursued.
- Parity gate vs the CPU AutoDiff head on a tiny fixture (mirror
  `SilkDistilBertTrainerBf16ParityTests`).
- BF16 narrow path: matmul weights BF16, activations/biases/LayerNorm F32, moments F32
  (same as DistilBERT narrow lane).
- Export checkpoint back to F32 tensors compatible with `LayaDecisionHead.LoadWeights`.

## Non-goals

- No activation-level BF16 training (kernels remain F32) unless spun off separately.
- No encoder training in the Silk Laya trainer (encoder stays frozen).

## References

- `src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner.cs`
- `src/Nivara.Extensions/AI/LayaHeadModel.cs`, `LayaHeadLayer`
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`
- `src/Nivara.Extensions/Silk/SilkPartialTrainingPolicy.cs`
- `docs/SILK.md` (partial/frozen training, narrow-precision lane)
