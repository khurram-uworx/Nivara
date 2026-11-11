# Laya head training — design notes (issue #1148)

## Status

Nivara does **not** currently ship a Silk trainer for the Laya decision head. The head trains
on CPU via AutoDiff (`LayaHeadModel` / `LayaHeadLayer`) on frozen ModernBERT features. This
document captures the design for a future Silk implementation so it is not lost.

Issue #1148 is the execution ticket. It was split out of #1103, whose BERT-family half shipped
on `SilkDistilBertTrainer`'s existing naming-driven path; see also #982 (training a
`LayaDecisionHead` on frozen ModernBERT features) and #1013 (the opt-in BF16 lane).

## Motivation

The encoder is frozen during Laya head training, so only the small head trains. A Silk trainer
would mirror `SilkDistilBertTrainer` but for the head's parameters and kernels.

## Scope (future)

- New `SilkLayaHeadTrainer` owning head weights, gradients, AdamW moments and the activation
  workspace.
- Reuse `SilkLayaHeadGpuRunner` for upload/layout and `SilkPartialTrainingPolicy` for the head's
  own layer count (independent of the encoder's `NumHiddenLayers`).
- Wire the backward chain over kernels that already exist and are already gated:
  `relu_backward`, `layer_norm_backward` (it takes `eps`, so the head's 1e-5 needs no new
  kernel), the three `attention_backward_*` kernels, `gemm_row4_nt` / `gemm_row4_tn`, and
  `col_sum`. The work here is the trainer scaffolding, not new kernels.
- Parity gate vs the CPU AutoDiff head on a tiny, **seeded** fixture (mirror
  `SilkDistilBertTrainerBf16ParityTests`; the first version of that gate was not seeded and its
  baseline moved on every run).
- BF16 narrow path: matmul weights BF16, activations/biases/LayerNorm F32, moments F32
  (same split as the DistilBERT narrow lane, #1135).
- Export checkpoint back to F32 tensors compatible with
  `LayaDecisionHead.LoadWeights(tensors, headLayerCount, actClassCount)`.

## Non-goals

- No activation-level BF16 training (kernels remain F32) unless #1149 lands first.
- No encoder training in the Silk Laya trainer (encoder stays frozen).

## References

- `src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner.cs`
- `src/Nivara.Extensions/AI/LayaHeadModel.cs`, `LayaHeadLayer`
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`
- `src/Nivara.Extensions/Silk/SilkPartialTrainingPolicy.cs`
- `docs/SILK.md` (partial/frozen training, narrow-precision lane)
