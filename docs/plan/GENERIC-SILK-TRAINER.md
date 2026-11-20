# Generic Silk trainer for non-BERT models (issue #1283)

## Status

Design note only. No implementation yet. Issue #1283 is the umbrella; implementation
is tracked as follow-up work (see "Proposed implementation plan"). Related: #1284
(LLM fine-tuning on CPU and Silk), #1149 (activation-level BF16 training), #1273
(LoRA fine-tuning design note).

## Motivation

Silk training today covers only BERT-shaped encoders (`SilkDistilBertTrainer`), the
Laya decision head (`SilkLayaHeadTrainer`), the ModernBERT classification head
(`SilkModernBertHeadTrainer`, head-only) and the TinyShakespeare batched transformer
(`SilkTinyShakespeareTrainer`). There is no Silk trainer for causal LMs (SmolLM,
Qwen) or vision models, and no generic trainer abstraction. CPU side has generic
AutoDiff (`LlamaForCausalLM<TModel,TWeight>`, `Module<T>`, `AdamW`) but the samples
only distill a small student MLP for Qwen; no in-sample training exists for SmolLM/
Qwen or vision.

The first concrete step requested by the issue is a design note scoping which models,
kernels and precision lanes are in scope, with explicit non-goals. This document is
that note.

## What exists today (inventory)

### Silk trainers

| Trainer | Scope | Status |
|---|---|---|
| `SilkDistilBertTrainer` (+ BF16 true, FP16) | BERT-shaped encoder (DistilBERT/BERT/MiniLM) | shipped |
| `SilkLayaHeadTrainer` (+ BF16 narrow) | Laya decision head on frozen encoder features | shipped |
| `SilkModernBertHeadTrainer` | ModernBERT classification head (frozen encoder) | shipped |
| `SilkTinyShakespeareTrainer` | Batched causal transformer (TinyShakespeare) | shipped |

### Silk inference runners (forward-only, no backward/Adam)

| Runner | Model | Backward available? |
|---|---|---|
| `SilkSmolLMGpuRunner` (+ BF16 widen) | SmolLM-135M (Llama family) | no |
| `SilkQwenGpuRunner` (+ BF16 widen) | Qwen2.5-0.5B (Llama family, GQA + bias) | no |
| `SilkMobileNetV2GpuRunner` (+ BF16 shim) | MobileNetV2 | no |
| `SilkResNet18GpuRunner` (+ BF16 shim) | ResNet-18 | no |

### Kernel inventory relevant to training

Available: `gemm_row4_*`, `col_sum`, `relu`/`relu_backward`, `gelu`/`gelu_backward`,
`layer_norm`/`layer_norm_backward`, `rms_norm` (forward only), `silu`/`swiglu`,
`causal_gqa_attention`, `causal_attention_backward_prep`, `attention_backward_dq`/
`attention_backward_dkv`, `rotary`, `conv2d` + `conv2d_backward_{weight,input,bias}`,
`embedding_*`, `softmax_cross_entropy_backward`, `adam_update`/`adam_update_bf16`,
`batchnorm_eval` (forward only), `maxpool2d` (forward only), `adaptive_avgpool2d`
(forward only).

**Missing for this scope (must be added first):**
- `rms_norm_backward` (causal LM Tier 1/2).
- `batchnorm_backward` (vision Tier 3).
- `maxpool2d_backward` (vision Tier 3).
- `adaptive_avgpool2d_backward` (vision Tier 3).

### CPU reference models already in `Nivara.Samples`

- `LlamaForCausalLM<TModel,TWeight>` — token embedding, decoder blocks, final RMSNorm,
  tied LM head. Used by both SmolLM and Qwen. This is the parity baseline for Tier 1/2.
- Vision models are loaded via ML.NET/TorchVision weights today; no AutoDiff module
  exists for them yet (a `Conv2d`/`BatchNorm` AutoDiff module would be needed for a
  CPU parity reference — see open decisions).

### Precedent

`SilkTinyShakespeareTrainer` + `BatchedTransformer` (in `samples/NivaraChat`) is the
existing causal-transformer-on-Silk precedent: device forward/backward over a transformer
block, Adam step, export back to CPU weights. Its parity tests in
`tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs` define the gate shape.

## Scope

### Tier 1 — SmolLM-135M causal LM trainer (highest value)

New `SilkSmolLMTrainer` (sealed, `IDisposable`) mirroring `SilkTinyShakespeareTrainer`,
but over the Llama family: RoPE, GQA, SwiGLU, RMSNorm, tied embeddings. Kernels mostly
exist; the one gap is `rms_norm_backward`.

- New files:
  1. `src/Nivara.Extensions/Silk/SilkSmolLMWeights.cs` (internal) — tracked, trainable
     weight set mirroring `SilkBertWeights` but for the Llama key space: `tok_emb.weight`
     (transposed), per-layer `input_layernorm`, `q_proj/k_proj/v_proj/o_proj`,
     `gate_proj/up_proj/down_proj`, `post_attention_layernorm`, `final_norm`, with
     `SilkWeightSlot` layouts so `Export`/`ExportGradients` invert device layout back to
     checkpoint layout.
  2. `src/Nivara.Extensions/Silk/SilkSmolLMTrainer.cs` (sealed partial, `IDisposable`) —
     constructor (`SilkRuntime`, tensors, config, caps, trainable-layer count),
     `Forward(tokenIds)`, `Backward(targets, scale)`, `AdamUpdateAll(lr, wd)`,
     `ZeroGradients()`, `ExportCheckpoint()`, `ExportGradients()`.
  3. `src/Nivara.Extensions/Silk/SilkSmolLMTrainer.Bf16.cs` (partial) — only if #1149
     (activation-level BF16) lands; otherwise omit and stay f32.
  4. `rms_norm_backward` kernel + launcher entry in `SilkKernels.cs`/`SilkLauncher.cs`.
- CPU reference: `LlamaForCausalLM<float,float>` (existing), via `GradientUtils.Grad()`.
- Single-sequence steps first (matching `SilkSmolLMGpuRunner.Forward` prefill); batching
  deferred (ragged sequence handling).

### Tier 2 — Qwen2.5-0.5B

Reuses the Tier-1 trainer with config differences (GQA with qkv bias, larger vocab/hidden,
different RoPE theta). No new kernels beyond Tier 1. Likely a thin `SilkQwenTrainer`
subclass or a shared trainer parameterized by a config + bias flag. Decision deferred
to implementation (see open decisions).

### Tier 3 — Vision (MobileNetV2, ResNet-18)

Blocked on the 3 missing backward kernels. Recommend **head-only fine-tune first**
(mirroring `SilkModernBertHeadTrainer`): freeze the convolutional backbone, train only
the classifier head on device. This unblocks a Silk training path for vision without
requiring conv/backbone backward. Full-backbone training is a separate, later item.

- New files (head-only first):
  1. `SilkVisionHeadWeights.cs` / `SilkVisionHeadTrainer.cs` generic over the head
     (linear classifier on pooled features), reusable for both MobileNetV2 and ResNet-18.
  2. Backward kernels for full backbone (deferred): `batchnorm_backward`,
     `maxpool2d_backward`, `adaptive_avgpool2d_backward`, then a full
     `SilkMobileNetV2Trainer` / `SilkResNet18Trainer`.

## Precision lanes

- **Training: f32 only** (release constraint, consistent with `docs/SILK.md` §13 and the
  #1274 decision for text classifiers). No BF16/FP16 training on Silk.
- Inference keeps existing lanes: BF16 widen for SmolLM/Qwen (tokEmb/tied LM head bf16),
  vision BF16 shims remain as-is (tracked separately under #1282).
- Narrow training (BF16 weights, F32 activations) is out of scope unless #1149 lands;
  then add `SilkSmolLMTrainer.Bf16.cs` partial mirroring `SilkDistilBertTrainer.Bf16True.cs`.

## Parity gate

Mirror `SilkTinyShakespeareTrainerTests` / `SilkDistilBertTrainerBf16ParityTests`:
- Seeded tiny fixture (small hidden size, few layers, small vocab) so the gate runs on
  CPU-less? no — requires OpenCL; skip gracefully when no device.
- Forward logits match CPU within 1e-4.
- Gradients match CPU within 1e-3 (max-abs) for all parameter groups.
- One Adam step: weights match CPU within 1e-2.
- Checkpoint export → reload round trip.
- Seeded weights (issue calls out the unseeded-baseline problem in the first Laya gate).

## Non-goals (explicit)

- No activation-level BF16 training (kernels remain F32) unless #1149 lands first.
- No encoder/backbone training in the first vision pass (head-only).
- No full-parameter LLM fine-tuning beyond a minimal SFT loop (#1284).
- No QLoRA / quantization (#1273 LORA non-goals).
- No replacement of CPU reference path (stays the parity baseline).
- No promotion of `BatchedTransformer` from NivaraChat (deferred; not needed — SmolLM/
  Qwen use `LlamaForCausalLM` in Nivara.Samples).
- No bf16/fp16 training on Silk (release constraint).
- No generic trainer abstraction in v1 — per-model trainers sharing a common shape; a
  generic base class is a later refactor if duplication warrants it.

## Proposed implementation plan (execution, tracked separately)

### New files (Tier 1, SmolLM)

1. `SilkSmolLMWeights.cs` (internal) — weight set + Adam moments + `Trainable` flags +
   `Export`/`ExportGradients`, mirroring `SilkBertWeights`/`SilkLayaHeadWeights`.
2. `SilkSmolLMTrainer.cs` (sealed partial, `IDisposable`) — constructor, `Forward`,
   `Backward`, `AdamUpdateAll`, `ZeroGradients`, `ExportCheckpoint`,
   `ExportGradients`.
3. `rms_norm_backward` kernel + launcher (in `SilkKernels.cs`/`SilkLauncher.cs`).
4. Tests: `tests/Nivara.Tests/Gpu/SilkSmolLMTrainerTests.cs` (F32 vs CPU AutoDiff on a
   seeded tiny fixture), plus parity + round-trip tests mirroring TinyShakespeare.
5. Docs: update this file's Status when shipped.

### Tier 2 (Qwen)

- Either `SilkQwenTrainer.cs` reusing Tier-1 internals with a bias flag, or a shared
  `SilkCausalLmTrainer` parameterized by config. Decide before implementation.

### Tier 3 (vision, head-only first)

- `SilkVisionHeadWeights.cs` / `SilkVisionHeadTrainer.cs` (generic head trainer).
- New kernels for full-backbone (deferred): `batchnorm_backward`, `maxpool2d_backward`,
  `adaptive_avgpool2d_backward`.

## Non-obvious points

- **Tied embeddings.** `LlamaForCausalLM` ties the LM head to the token embedding. On
  device this means the tokEmb buffer is shared with the head projection; gradients must
  accumulate into the same buffer (mirror the CPU path's tied-grad handling).
- **RoPE is position-dependent.** The trainer must recompute cos/sin tables per sequence
  length (reuse `SilkSmolLMGpuRunner`'s rotary buffer logic) and save them for backward.
- **Single sequence per step.** Like `SilkTinyShakespeareTrainer`, step on one sequence
  at a time (`hidden` is `[seqLen, hidden]`). Batching deferred.
- **Upstream gradients are caller-supplied** for head-only vision; for causal LMs the
  trainer owns the cross-entropy loss internally (like TinyShakespeare's
  `Backward(targets, scale)`).
- **No activation-level BF16** (kernels stay F32) unless #1149 lands first.

## Open decisions to confirm before implementation

- Shared `SilkCausalLmTrainer` vs. separate `SilkSmolLMTrainer`/`SilkQwenTrainer`.
- Whether the parity reference is CPU AutoDiff `LlamaForCausalLM` (F32) only, or also
  the F32 Silk trainer for the narrow lane (recommended: CPU only for v1).
- Whether to ship a convenience `Train(...)` loop in v1 or just primitives (recommended:
  primitives first, matching DistilBERT's surface).
- Vision: head-only first (recommended) vs. full backbone requiring 3 new kernels.
- Whether a `Conv2d`/`BatchNorm` AutoDiff module is needed in `Nivara.Samples` to
  provide a CPU parity reference for vision (recommended: yes, but separate item).

## References

- `src/Nivara.Extensions/Silk/SilkTinyShakespeareTrainer.cs` (precedent causal-LM trainer)
- `src/Nivara.Extensions/Silk/SilkTinyShakespeareGpuRunner.cs`
- `src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs`, `SilkQwenGpuRunner.cs`
- `src/Nivara.Extensions/Silk/SilkModernBertHeadTrainer.cs` (head-only precedent)
- `src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.cs` (trainer scaffolding precedent)
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`
- `src/Nivara.Extensions/Silk/SilkPartialTrainingPolicy.cs`
- `samples/Nivara.Samples/LlamaForCausalLM.cs`
- `samples/NivaraChat/Transformer/BatchedTransformer.cs` (TinyShakespeare CPU model)
- `docs/SILK.md` §13 (training), `docs/plan/LAYA-HEAD-TRAINING.md` (template)
- `docs/plan/NIVARAINFERENCE-SILK-GAPS.md` (umbrella, gaps N5/N6)
- `tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs` (parity gate shape)