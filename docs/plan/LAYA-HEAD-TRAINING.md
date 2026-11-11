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

## Proposed implementation plan (execution ticket #1148)

This section captures the concrete work breakdown so the doc is the single point of reference
when #1148 is picked up. Nothing here is implemented yet; #1148 stays in the backlog.

### New files

1. `src/Nivara.Extensions/Silk/SilkLayaHeadWeights.cs` (internal) — tracked, trainable weight
   set mirroring `SilkBertWeights` but for the head's key space:
   - Params: `TypeEmb`, `ScorerNormW/B`, `ScorerHiddenW/B` (transposed), `ScorerOutW/B`,
     `ActInW/B` (transposed), `ActOutW/B`, and `LayerWeights[]` (norm1, packed qkv, out-proj,
     norm2, linear1/linear2) with the same `SilkWeightSlot` layouts so `Export`/`ExportGradients`
     invert device layout back to checkpoint layout.
   - F32 constructor (from `Dictionary<string,(float[],int[])>`) and narrow constructor (from
     `ushort[]` + widen delegate), with `NarrowMatrix: true` on matmul weights only.
   - AdamW buffers + `Trainable` flags; `SilkPartialTrainingPolicy.HeadTrainable(true)` controls
     all of them.

2. `src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.cs` (sealed partial, `IDisposable`) — mirrors
   `SilkDistilBertTrainer`:
   - Constructor: `SilkRuntime`, tensors, `headLayerCount`, `actClassCount`, caps, and narrow/
     activation flags.
   - Saved per-layer activations: `PreNorm1`, `H1`, `Qkv`, `Attn`, `F1Pre`, `F1Post`,
     `PreNorm2`, plus `HiddenIn` (type-embedding-added input) and scorer/act intermediates
     (`MarkerRows`, `MarkerScored`, `MarkerScoredOut`, `ActRow`, `ActHidden`).
   - `Forward(hidden, type, markers, validLength)` runs the runner's sequence while saving
     intermediates.
   - `Backward(..., dScorerLogits, dActLogits)` runs the reverse chain: act head (into the
     pooled row), scorer (into marker rows), then layers (LN backward with eps 1e-5, attention
     backward via the three `attention_backward_*` kernels, ReLU backward, FFN backward), then
     type-embedding. Encoder gradient is discarded (frozen).
   - `AdamUpdateAll(lr, wd)` over `weights.Parameters.Where(Trainable)`, using `AdamUpdateBf16`
     for narrow params.
   - `ExportCheckpoint()` returns `weights.Export()` (F32, checkpoint keys/shapes).

3. `src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.Bf16.cs` (partial) — `CreateTrueBf16` factory
   and narrow constructor wiring, mirroring `SilkDistilBertTrainer.Bf16True.cs`.

4. Tests:
   - `tests/Nivara.Tests/Gpu/SilkLayaHeadTrainerTests.cs` — F32 trainer vs the CPU AutoDiff head:
     forward logits match; gradients match (seeded synthetic fixture, small hidden size).
   - `tests/Nivara.Tests/Gpu/SilkLayaHeadTrainerBf16ParityTests.cs` — BF16 vs F32 within the
     narrow bound, on a seeded fixture (mirror of `SilkDistilBertTrainerBf16ParityTests`, with the
     seeded-weight fix the issue calls out).
   - Seeded fixture helper (e.g. `SilkLayaHeadBf16Fixtures`) producing synthetic head tensors, a
     deterministic hidden state and a marker list.

5. Docs: update this file's Status when shipped; no ADR change required.

### Non-obvious points

- **Detached features.** In `LayaDecisionHead.Forward`, `probabilities` and `features` are built
  from a detached `float[]` extracted from `scored.Data`; `featureRow` enters the act head via
  `FromMatrix` with no gradient edge. Therefore the act loss does **not** flow back into the scorer
  logits — only the pooled row and the type embedding receive gradients from both paths. This is
  what makes the two backward paths independently summable.
- **Single question per step.** The head is per-question (variable marker count, `validLength`),
  so the trainer steps on one question at a time (`hidden` is `[seqLen, hidden]`), matching
  `SilkLayaHeadGpuRunner.Forward`. Batching would require ragged marker handling and is deferred.
- **Upstream gradients are caller-supplied**, exactly like
  `SilkDistilBertTrainer.Backward(batch, seqLen, dLogitsSpan)`: the caller builds the
  cross-entropy input gradients for both scorer and act logits, which keeps the parity gate
  comparable on both sides.
- **No activation-level BF16** (kernels stay F32) unless #1149 lands first, per non-goals.
- **No encoder training** — the encoder stays frozen; only head params are tracked.

### Open decisions to confirm before implementation

- Single-question vs. batched training steps.
- Whether the parity gate's reference is the CPU AutoDiff head (F32) or the F32 Silk trainer
  (narrow lane), or both (recommended: both — F32-vs-CPU for correctness, BF16-vs-F32 for the
  narrow bound).
- Whether to also ship a convenience `Train(...)` loop in v1 or just `Forward`/`Backward`/
  `AdamUpdateAll`/`ExportCheckpoint` (recommended: ship the primitives first, matching the
  DistilBERT trainer's surface, and add the loop only if a sample needs it).

### References

- `src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner.cs`
- `src/Nivara.Extensions/AI/LayaHeadModel.cs`, `LayaHeadLayer`
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`
- `src/Nivara.Extensions/Silk/SilkPartialTrainingPolicy.cs`
- `docs/SILK.md` (partial/frozen training, narrow-precision lane)
