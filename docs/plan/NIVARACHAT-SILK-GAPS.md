# NivaraChat Silk gaps (post-#1266/#1268/#1324)

**Status:** plan, not yet executed. Written 2026-11-19.
**Scope:** `samples/NivaraChat` (and, where noted, core `src/Nivara.Extensions/Silk`).
**Release constraint:** Silk training is allowed to be **f32 only** for this release. No bf16/fp16
training on Silk is required; inference may still use bf16/fp16 where a runner exists.

This document captures the remaining Silk gaps identified when comparing NivaraChat's
`--silk` coverage against the "first-class Silk" bar, now that issues #1266 (domain text
models), #1268 (Laya decision head) and #1324 (Silk `IChatClient` for TinyShakespeare) have
landed. The earlier `docs/plan/NIVARACHAT-GAPS.md` was retired once those closed; this
document supersedes it for the Silk-specific remainder.

## Assessment summary

NivaraChat's `--silk` coverage is broad but uneven:

- **Fully on Silk (inference + training):** MiniLM embeddings (`MiniLMEmbedder`), sentiment,
  validator and agents-validator document classifiers (`SilkTextClassifier` over
  `GpuTextLinearClassifier`), intent classifier (including incremental retrain via
  `IntentTrainer.TrainIncremental`), differentiable retrieval fusion (`SilkRetrievalTrainer`),
  Laya decision-head **inference** (`LayaDecisionEngine` with `silkSession`), and the
  TinyShakespeare / SmolLM / Qwen causal-LM generation paths (`SilkBatchedChatClient`,
  `SilkSmolLMChatClient`, `SilkQwenChatClient`).
- **Partially on Silk:** Laya decision-head **training** (adaptive head) still runs on CPU
  AutoDiff in NivaraChat even though core ships `SilkLayaHeadTrainer`; the NivaraInference
  sample already wires it, NivaraChat does not.
- **Not on Silk at all:** the entity (token-level) classifier — neither inference nor
  training has a Silk path; `EntityTrainer` and `EntityTextModel` are CPU-only and
  `GpuTextLinearClassifier` is document-level.
- **Stays on CPU AutoDiff (reference path):** SmolLM/Qwen (pretrained weights from
  safetensors, no training), classifier-free LLM fine-tuning. TinyShakespeare transformer
  training is now tracked as Gap S6 (#1342) rather than an implicit non-goal.

The clearest gap is the entity classifier: it is the only domain model with no Silk path, and
there is no `GpuTokenClassifier` in core to back it.

## Open issues already filed (relevant)

| Issue | Title | Priority |
|---|---|---|
| #1340 | NivaraChat: entity (token-level) classifier has no Silk path (inference + training) | High |
| #1341 | NivaraChat: Laya adaptive-head training is CPU-only (SilkLayaHeadTrainer not wired) | Medium |
| #1274 | NivaraChat: core Silk text classifiers are float-only (no bf16/fp16) | Low |
| #1276 | NivaraChat: mixed backends within one run are not reported accurately | Low |
| #1273 | NivaraChat: no training demo for Llama (SmolLM/Qwen pretrained-only) | Low |
| #1283 | NivaraInference: generic Silk trainer for non-BERT models (SmolLM, Qwen LLM, vision) | Low |
| #1284 | NivaraInference: LLM fine-tuning (SmolLM / Qwen) on CPU and Silk | Low |
| #1342 | NivaraChat: TinyShakespeare transformer training on Silk (AutoDiff → device) | Low |

## Identified gaps (not yet filed)

### Gap S1 — Entity classifier has no Silk path (inference + training) — High (#1340)

**Symptom:** `TrainingMode` prints "Silk (OpenCL) for sentiment/validator/agents-validator/intent;
CPU for entity". `EntityTrainer.Train` has no `useSilk` parameter; `EntityTextModel` has no silk
field; `NivaraToolFunctions`/`AgentsMode`/`QwenMode` (tools-nivara) never attempt a Silk
entity runner. There is no `GpuTokenClassifier` in `src/Nivara.Extensions/Silk/` — only
`GpuTextLinearClassifier` (document-level) and `GpuTextConvClassifier`.

**Goal:** add a Silk token-classifier (inference + f32 training) mirroring
`GpuTextLinearClassifier`, wire it into `EntityTrainer` and `EntityTextModel` with CPU fallback,
and update the mode reporting. Reuses the existing `TokenClassifierModel<float>` weight layout
(embedding + linear + per-token head).

**Depends on:** new core primitive `GpuTokenClassifier` (or a token-level variant of the linear
classifier). This is the largest single item here.

**Key files:** `samples/NivaraChat/Training/EntityTrainer.cs`, `samples/NivaraChat/Models/EntityTextModel.cs`,
`samples/NivaraChat/Modes/AgentsMode.cs`, `ToolsMode.cs`, `Qwen/QwenMode.cs`,
`src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs` (template).

### Gap S2 — Laya adaptive-head training in NivaraChat is CPU-only — Medium (#1341)

**Symptom:** `RagAdaptiveEvalMode` trains the adaptive Laya head via `AdaptiveTraining.Update`
(CPU AutoDiff) and `AdaptiveFeedbackCollector`. Core already ships `SilkLayaHeadTrainer` (f32 +
bf16 narrow lane) and the NivaraInference sample wires it, but NivaraChat does not call it.
Inference on Silk is already wired (`LayaDecisionEngine` with `silkSession`).

**Goal:** add a `--silk` head-training path in `RagAdaptiveEvalMode` using `SilkLayaHeadTrainer`,
with CPU fallback, and record the backend in the adaptive report. f32 only for this release.

**Depends on:** none (core primitive exists).

**Key files:** `samples/NivaraChat/Modes/RagAdaptiveEvalMode.cs`,
`samples/NivaraChat/Helpers/AdaptiveFeedbackCollector.cs`, `AdaptiveTraining.cs`,
`src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.cs`.

### Gap S3 — Mixed-backend reporting (#1276) — Low

Already filed. Per-component backend + precision matrix in run output and `AdaptiveReport`.

### Gap S4 - Silk text classifiers float-only (#1274) - Low

**Resolved (khurram/1274): decision documented.** Text classifiers remain f32-only: no bf16/fp16
partials for `GpuTextLinearClassifier`/`GpuTextConvClassifier` (or `GpuTokenClassifier`).
Rationale: text classifiers in NivaraChat/NivaraClassifier are trained and serialized as f32;
Silk training is f32-only this release; the CPU reference models are f32; narrow storage would
require new device kernels and weight loaders for no benefit at classifier scale and would break
the f32 parity gate. The f32 CPU↔Silk parity is pinned by
`tests/Nivara.Tests/Gpu/GpuTextConvClassifierTests.cs` and
`tests/Nivara.Tests/Gpu/GpuTextLinearClassifierTests.cs`. See `docs/SILK.md` §13c
("Precision: f32 only (decision, #1274)").

### Gap S5 — Llama/SmolLM/Qwen training demo (#1273/#1283/#1284) — Low (optional)

SmolLM/Qwen are inference-only (pretrained weights from safetensors). Core AutoDiff supports
training `LlamaForCausalLM<TModel,TWeight>` but it is not exercised in the sample. Generic Silk
training for causal LMs does not exist yet (only BERT-shaped trainers). Treat as design-first:
a design note scoping data format, loss, optimizer, memory budget, precision. Not a release
blocker.

### Gap S6 — TinyShakespeare transformer training on Silk — Low (#1342) — IMPLEMENTED

`--tinyshakespeare --silk` now trains on the device via `SilkTinyShakespeareTrainer`
(f32 only), exporting weights back into the CPU `BatchedTransformer` through
`LoadFromSilkWeights`. The CPU AutoDiff path remains the default and reference parity gate.
Parity tests in `tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs`.

**Goal:** a Silk training path for the batched transformer — device forward/backward for the
transformer block (attention + FFN + layernorm + embedding) and an Adam step, exporting
trained weights back to the CPU `BatchedTransformer` for save/load and inference. Preserve
the CPU AutoDiff path as the reference parity gate.

**Constraints:** f32 only this release; parity gate vs CPU AutoDiff; AutoDiff semantics
matched or documented as divergent.

**Depends on:** design note (done); large, open-ended. Not a release blocker.

**Key files:** `samples/NivaraChat/Transformer/BatchedTransformer.cs`,
`Transformer/TransformerMode.cs`, `src/Nivara.Extensions/Silk/SilkTinyShakespeareSession.cs`,
`src/Nivara/AutoDiff/ReverseGradOperations.cs`.

## Non-goals (intentionally not generalized)

- `ToSilkWeights` for SmolLM/Qwen: they do not train in Nivara; weights come from safetensors.
- Rank-3 (batched) MatMul in core (ADR-003).
- BPE migration for TinyShakespeare (D-014, issue #1328) — separate, model-breaking.
- bf16/fp16 *training* on Silk beyond what already ships (release constraint: f32 only).

## Key references for coding agents

- `samples/NivaraChat/SmolLM/SilkSmolLMChatClient.cs`, `Qwen/SilkQwenChatClient.cs`,
  `Transformer/SilkBatchedChatClient.cs` — chat-client template.
- `samples/NivaraChat/Helpers/SilkTextClassifier.cs` — document-classifier Silk adapter.
- `src/Nivara.Extensions/Silk/GpuTextLinearClassifier.cs` — f32 device classifier (template for S1).
- `src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.cs` — f32+bf16 head trainer (for S2).
- `samples/NivaraInference/Laya.Train.cs` — reference wiring of `SilkLayaHeadTrainer`.
- `samples/NivaraChat/Modes/TrainingMode.cs` — backend reporting line to update.
- `samples/NivaraChat/ARCHITECTURE.md` — decisions D-014..D-017 and "Not doing" section.

## Picking order

1. **S2** (Laya head training on Silk) — core primitive exists; medium effort, unblocks a
   real `--silk` training path in NivaraChat. Do first.
2. **S1** (entity Silk) — requires new core primitive `GpuTokenClassifier`; largest item.
   Design the weight mapping first.
3. **S3** (#1276) and **S4** (#1274) — low priority; S4 is a decision, not code.
4. **S5** — design note only; defer implementation.
5. **S6** (#1342) — design note first; large, open-ended; not a release blocker.