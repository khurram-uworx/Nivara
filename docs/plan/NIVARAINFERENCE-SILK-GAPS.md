# NivaraInference Silk gaps (umbrella)

**Status:** plan, not yet executed. Written 2026-11-19.
**Scope:** `samples/NivaraInference` (and, where noted, core `src/Nivara.Extensions/Silk`).
**Release constraint:** Silk training is allowed to be **f32 only** for this release. No bf16/fp16
training on Silk is required; inference may still use bf16/fp16 where a runner exists.

This document is the single umbrella for Silk gaps in the NivaraInference sample. It consolidates
the already-filed issues #1279–#1284 and #1343 and records the assessment of what is already on
Silk versus what is not. The NivaraChat counterpart was retired once all of its gaps (S1–S6) closed.

## Assessment summary

NivaraInference has the broadest Silk coverage of any sample: **all nine models** are reachable
on Silk (`SilkModes` in `Program.cs`: minilm, distilbert, distilbert_sst, modernbert, laya,
smollm, qwen, mobilenet_v2, resnet18), with F32 and (for most) BF16 inference paths.

- **On Silk (inference):** MiniLM, DistilBERT, DistilBERT SST-2, ModernBERT-large, Laya
  (encoder + decision head), SmolLM-135M, Qwen2.5-0.5B, MobileNetV2, ResNet-18 — F32
  (and BF16 where a true narrow runner exists).
- **On Silk (training):** Laya decision head only (`SilkLayaHeadTrainer`, via `laya train
  --silk`). Everything else is inference-only.
- **CPU-only training:** Laya head has no CPU path in this sample (Silk-only). SmolLM/Qwen
  have no training path at all (pretrained weights from safetensors). ModernBERT has no
  training path. Vision models have no training path.
- **Precision gaps:** Silk FP16 is rejected globally (`--silk --precision fp16`), yet runner-
  level FP16 constructors exist (dead code). Vision "BF16" runners only widen to F32. Text
  classifiers (in NivaraChat) are F32-only (tracked under NivaraChat, #1274).

The clearest gaps are the **FP16 story** (dead runner code vs. rejected CLI flag) and the
**vision BF16 shims** (advertise a lane that does not exist).

## Issues already filed (relevant)

| Issue | Title | Priority |
|---|---|---|
| #1343 | NivaraInference: Laya decision-head training is Silk-only (no CPU path) | Low |
| #1284 | NivaraInference: LLM fine-tuning (SmolLM / Qwen) on CPU and Silk | Low |
| #1283 | NivaraInference: generic Silk trainer for non-BERT models (SmolLM, Qwen LLM, vision) | Low |
| #1282 | NivaraInference: remove or clearly mark vision Silk Bf16 runner shims that only widen to float | Low |
| #1281 | NivaraInference: decide the FP16 story (remove unreachable Silk runners or expose the flag) | Medium |
| #1280 | NivaraInference: add a ModernBERT fine-tune path (head-only on frozen features) | Medium |
| #1279 | NivaraInference: expose BF16 training in NivaraFineTuning (CPU + Silk) | High |

## Identified gaps (not yet filed beyond the above)

### Gap N1 — Laya training is Silk-only (no CPU path) — Low (#1343)

`laya train` requires `--silk`; no CPU AutoDiff path exists in this sample (unlike
NivaraChat's `AdaptiveTraining`). Blocks parity-checking and use on OpenCL-less machines.

### Gap N2 — FP16 story undecided (#1281) — Medium

CPU narrow paths support FP16 (Half) for several models; Silk runners exist but the CLI
rejects `--silk --precision fp16` globally. Dead runner code remains. Decision needed:
remove unreachable runners + CPU FP16 paths (Option A) or expose the flag (Option B).

### Gap N3 — Vision BF16 shims (#1282) — Low (resolved)

`SilkMobileNetV2GpuRunner.Bf16.cs` and `SilkResNet18GpuRunner.Bf16.cs` widened to F32; no
true narrow path. Shims removed in #1282: vision Silk runners are now F32-only.

### Gap N4 — ModernBERT has no training path (#1280) — Medium

ModernBERT-large is inference-only. No AutoDiff wrapper or Silk trainer covers its hybrid
(full + sliding-window) attention. Head-only fine-tune on frozen `[CLS]` is the proposed
scope.

### Gap N5 - SmolLM/Qwen have no training path (#1283, #1284) - Low

Pretrained weights from safetensors; no in-sample training. Generic Silk trainer for causal
LMs does not exist (only BERT-shaped `SilkDistilBertTrainer` and `SilkLayaHeadTrainer`).

**Home decision.** LoRA/fine-tuning is homed in NivaraInference (see `docs/plan/LORA.md`,
which is the consolidated design note). NivaraChat keeps the RAG comparison and a thin
consumption path that loads an adapter produced here. Shared code (`LoRALinear`, data format,
templates) matures in `src/Nivara`/`Nivara.Samples` and is promoted once stable.

### Gap N6 — BF16 training unreachable in NivaraFineTuning (#1279) — High

`SilkDistilBertTrainer.CreateTrueBf16` exists but is not reachable from the sample; CPU has
no BF16 fine-tune path. Separate sample but part of the same Silk-training story.

## Non-goals (intentionally not generalized)

- LLM fine-tuning beyond a minimal SFT loop (large, open-ended; design note first, #1284).
- bf16/fp16 *training* on Silk beyond what already ships (release constraint: f32 only).
- Replacing the CPU paths as the reference (they stay the parity baseline).
- Qwen distill on Silk is f32 only (consistent with the constraint; not a gap).

## Key references for coding agents

- `samples/NivaraInference/Program.cs` (`SilkModes`, dispatch switch, precision guard).
- `src/Nivara.Extensions/Silk/` (runner inventory; `SilkDistilBertTrainer*`, `SilkLayaHeadTrainer*`).
- `samples/NivaraInference/Laya.Train.cs`, `Laya.Narrow.cs`, `ModernBert.Narrow.cs`.
- `docs/ACCELERATION.md`, `docs/SILK.md` (FP16 rejection rationale, precision lanes).
- `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` (CLI surface gate).

## Picking order

1. **N2 (#1281)** — decide the FP16 story; unblocks cleanup and consistency. Medium.
2. **N3 (#1282)** — remove vision BF16 shims (depends on N2 decision). Low.
3. **N4 (#1280)** — ModernBERT head-only fine-tune (CPU first, Silk optional). Medium.
4. **N6 (#1279)** — expose BF16 training in NivaraFineTuning. High, but separate sample.
5. **N1 (#1343)** — Laya CPU training path. Low.
6. **N5 (#1283/#1284)** — design notes first; large, open-ended.