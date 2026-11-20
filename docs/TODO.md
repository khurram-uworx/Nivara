# Generic Silk trainer for non-BERT models (issue #1283)

## Problem

Silk training exists only for BERT-shaped encoders (`SilkDistilBertTrainer`), the Laya
decision head (`SilkLayaHeadTrainer`), the ModernBERT classification head
(`SilkModernBertHeadTrainer`) and the TinyShakespeare batched transformer
(`SilkTinyShakespeareTrainer`). There is no Silk trainer for causal LMs (SmolLM,
Qwen) or vision models, and no generic trainer abstraction. CPU side has generic
AutoDiff but no sample training for these (only Qwen distills a small student MLP).

Acceptance criteria: a design note under `docs/plan/` scoping the work, with explicit
non-goals. Implementation tracked separately.

## Proposed changes

Create `docs/plan/GENERIC-SILK-TRAINER.md` scoping:
- Tier 1: SmolLM-135M causal LM trainer (Llama family) on Silk, mirroring
  `SilkTinyShakespeareTrainer`, over `LlamaForCausalLM`. Requires a new
  `rms_norm_backward` kernel (currently only `rms_norm` forward exists) plus trainer
  scaffolding (weights, gradients, Adam moments, activations, forward/backward/AdamUpdate/
  ExportCheckpoint).
- Tier 2: Qwen2.5-0.5B reuses the Tier-1 trainer (GQA + qkv bias + tied embeddings).
- Tier 3: vision (MobileNetV2, ResNet-18). Requires 3 new backward kernels first
  (BatchNorm, MaxPool2d, AdaptiveAvgPool2d — only forward/eval exist). Recommend
  head-only fine-tune first, mirroring `SilkModernBertHeadTrainer`.
- Precision lanes: f32 only for training (release constraint), consistent with #1274.
- Parity gate vs CPU AutoDiff on a seeded tiny fixture (gradient max-abs bounds).
- Checkpoint export round-trip via weight-dictionary convention.
- Explicit non-goals (see below).

## Non-goals (explicit)

- No activation-level BF16 training (kernels remain F32) unless #1149 lands first.
- No encoder training in any Laya trainer (encoder stays frozen).
- No full-parameter LLM fine-tuning beyond a minimal SFT loop (#1284).
- No QLoRA / quantization (#1273 LORA non-goals).
- No replacement of CPU reference path (it stays the parity baseline).
- No vision full-backbone training until the 3 missing backward kernels land; head-only
  first.
- No promotion of `BatchedTransformer` from NivaraChat (deferred; not needed for this
  scope — SmolLM/Qwen use `LlamaForCausalLM` in Nivara.Samples).
- No bf16/fp16 training on Silk (release constraint).

## Verification steps

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` clean.
- Docs snippet gate compiles (no new gated csharp blocks in the note; plain fenced
  blocks only, matching LAYA-HEAD-TRAINING.md style).
- `DocsCoveragePins` unaffected (top-level docs/*.md count unchanged; design note is
  under docs/plan/).
- AdrIndexTests unaffected (no new ADR).

## Planned commits

1. `docs: plan generic Silk trainer (issue #1283) in TODO.md`
2. `docs: add generic Silk trainer design note (issue #1283)` (new docs/plan file)
3. `docs: remove TODO.md — plan executed`

## Blast radius

- New file only: `docs/plan/GENERIC-SILK-TRAINER.md`. No source/test changes.
- `DocsCoveragePins.ExpectedDocumentCount` counts top-level `docs/*.md` only; unaffected.
- No code paths touched; no tests to update.

## GitHub issues log

- #1283 — generic Silk trainer for non-BERT models (this issue; design note).
- #1284 — LLM fine-tuning (SmolLM/Qwen) on CPU and Silk (related).
- #1149 — activation-level BF16 training (prerequisite for narrow training lane).
- (new, at implementation time) missing kernels: `rms_norm_backward`; vision
  `batchnorm_backward`, `maxpool2d_backward`, `adaptive_avgpool2d_backward`.

## Grounding (Resistance G1)

- Codebase navigation via code-memory MCP: confirmed Silk trainer inventory, runner
  inventory, kernel inventory (rms_norm forward-only; batchnorm_eval forward-only;
  no pool backward kernels). Confirmed `LlamaForCausalLM` lives in
  `Nivara.Samples` (CPU reference available). Confirmed `BatchedTransformer` lives in
  `NivaraChat` (not needed for this scope).
- Official docs: Silk training is f32-only this release (docs/SILK.md §13, #1274
  decision). No Microsoft Learn grounding needed (this is an internal design note, not
  a BCL API change).
- Blast radius: docs-only change; no compiled code affected.