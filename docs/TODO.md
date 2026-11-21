# LoRA Fine-Tuning SLMs on iGPUs — Plan (Silk headline)

> **Plan-first workflow (iterative-work skill).** This file is the persisted plan. It
> will be removed only after G2 (branch review as a whole + against this plan) clears.
> As each task executes, if deferred work or a concern is found outside the current
> plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`)
> and record it in the GitHub issues log below. Do not rely on memory.

## Problem / goal

Nivara already showcases RAG strongly in NivaraChat (multi-view, eval, learn, adaptive)
and already ships Silk iGPU fine-tuning for DistilBERT (`NivaraFineTuning --mode train
--silk`). What is missing is the headline story the user asked for:

> **Silk is our main headline; FT SLMs on iGPUs is Nivara's headline.**
>
> **Model choice (user direction):** **Qwen2.5-0.5B** (already in repo) is the headline
> showcase model, not Qwen3. **Implementation must be generic enough** for Llama-family
> causal LMs (`LlamaForCausalLM<TModel,TWeight>`), and we will use the existing
> **SmolLM** model for verification that the work is generic. Plan references to the
> headline target mean Qwen2.5-0.5B; references to generic SLM/causal-LM support mean the
> Llama-family path verified on SmolLM too.

The existing `docs/plan/LORA.md` frames LoRA as a doc-QA RAG-vs-FT comparison. That is
the wrong showcase: small 135M/0.5B models do not memorize docs well, so FT looks weak
and the comparison frames RAG vs FT as competitors. The goal is **not** RAG vs FT; the
goal is to showcase that Nivara can do RAG *and* FT, each playing to its strengths.

This plan reshapes the LoRA work into a behavioral/domain-adaptation SLM fine-tuning
showcase, with Silk/iGPU as the headline path. RAG remains the docs/knowledge showcase
in NivaraChat; it is not reworked here except small touchpoints if needed.

## Headline / showcase framing

| Showcase | Strength | Where | Headline path |
|---|---|---|---|
| **RAG** | Fresh, grounded, inspectable factual recall over changing docs | NivaraChat (`--rag*` modes) | Already shipped; no regression |
| **FT SLMs on iGPUs** | Teach domain behavior/format/conventions to a small LM; run/train with Silk on iGPU | NivaraFineTuning (+ optional NivaraChat adapter consumption) | **Silk/iGPU is the headline**; CPU is the reference/fallback |

**FT scenario (proposed, G1 decision):** worklog / company-assistant behavioral
adaptation, implemented Nivara-natively as a synthetic SFT demo. The model learns
domain conventions and structured behavior from natural-language worklog entries.

**Captured scenario details (self-contained; no external dependency):**

- **Task:** convert a natural-language worklog entry into a structured `LogTime` record.
- **Fields:** `ticket` (e.g. `POS-100`, `AUTH-7`, `API-314`), `hours` (decimal), `date`
  (ISO `YYYY-MM-DD`; default/relative dates resolved deterministically in data gen).
- **Domain conventions:** standard office day = 8 hours; "all day"/"full day" = 8h
  (not 24h); "half day" = 4h; explicit durations used as-is (including fractions like
  2.5h, 20 minutes = 0.33h or rounded per rule — pick one deterministic rule).
- **Multi-ticket:** one message mentioning multiple tickets produces separate records/tool
  calls, one per ticket.
- **Ambiguity:** ambiguous/non-worklog input asks for clarification rather than guessing.
- **Data shape:** synthetic, deterministic, seeded JSONL with chat messages
  (`system`, `user`, `assistant`) in the target model's chat template; assistant output
  is a structured tool-call format (decide exact format during G1: XML-style tool call,
  JSON function call, or small JSON record — must be parseable for eval).
- **Eval metrics:** valid-structure %, ticket accuracy, hours accuracy, date accuracy,
  multi-ticket decomposition accuracy, clarification/abstention accuracy, convention
  accuracy (all-day/half-day). Report before/after (base vs LoRA-adapted).
- **Why this showcases FT:** base small LMs often get "all day" = 24h, miss tool schema,
  or format incorrectly; LoRA can teach these conventions/behavior with a small adapter.

This plays to small-LoRA strengths and gives a vivid before/after.

**Why not doc-QA memorization:** LORA.md §7 already predicts small SLMs + rank-8 LoRA
will not memorize a large doc set; that makes FT look bad. Behavioral adaptation is the
right SLM/LoRA showcase.

## What exists today (facts)

- `LoRALinear<T>` and `LoRALinear<TModel,TWeight>` in core (`src/Nivara/AutoDiff/Nn/LoRALinear.cs`), with `Merge()`, frozen base, Kaiming A / zero B.
- `LoraAdapterSerializer` (`src/Nivara/AutoDiff/Serialization/LoraAdapterSerializer.cs`) saves/loads adapter-only `A/B` + metadata; has round-trip tests.
- `Module<T>.Freeze()/Unfreeze()/SetRequiresGrad()` exists; prereq #1374 closed.
- `LlamaForCausalLM<TModel,TWeight>` has `AttachLora(...)`, `MergeAllAdapters()`, `Forward(int[] inputIds)`; core AutoDiff supports training it generically.
- Chat templates promoted to `samples/Nivara.Samples/SmollmChatTemplate.cs`, `QwenChatTemplate.cs` (prereq #1372 closed).
- Causal-LM loading helpers extracted to `Nivara.Samples` (prereq #1373 closed).
- Silk inference runners exist: `SilkSmolLMGpuRunner`, `SilkQwenGpuRunner`.
- Silk trainers exist: `SilkDistilBertTrainer` (+ BF16/FP16/TrueBf16), `SilkLayaHeadTrainer`, `SilkModernBertHeadTrainer`, `SilkTinyShakespeareTrainer`.
- **No** `SilkSmolLMTrainer` / `SilkQwenTrainer`; **no** `rms_norm_backward` kernel (LORA.md §14). Causal LM training on Silk is not implemented. Qwen2.5-0.5B already has Silk inference (`SilkQwenGpuRunner`) and chat template/loading helpers in repo.
- NivaraFineTuning currently fine-tunes DistilBERT SST-2 classification, CPU and Silk.
- Untracked files on branch start: `samples/NivaraChat/Helpers/FtDataBuilder.cs`, `FtDataModels.cs` (look like prior doc-QA ft-data work; must be reconciled, not silently dropped).

## Grounding / constraints (G1 inputs)

- **.NET tensors/spans are new/unstable in places** — ground any new Tensor/Span/BitConverter/BFloat16 usage via microsoft-learn MCP and a compile probe before committing design.
- **Silk training is f32 only** (release constraint; see docs/SILK.md §13 and LORA.md §14). BF16 narrow training is out of scope unless #1149 lands.
- **No `rms_norm_backward` kernel today.** A Silk causal-LM trainer for Llama-family needs it (LORA.md §14 Tier 1). This is a hard dependency for true "FT on iGPU" for Qwen2.5-0.5B.
- **Memory:** Qwen2.5-0.5B BF16 base weights are ~1GB; F32 activations/gradients for seq len 256-512 can be large. LoRA keeps adapter params small, but backward through frozen base still computes gradients flowing to adapters unless carefully designed. The plan must include a memory budget/guard and small default batch/seq settings.
- **Behavioral FT must be measurable.** Define metrics before implementation: field accuracy (ticket/hours/date), format validity, clarification accuracy, all-day/half-day convention accuracy.
- **Do not break existing RAG showcase.** Any NivaraChat changes must be additive and not regress existing RAG modes.
- **Benchmarks/tests:** targeted tests autonomously; full suite only with human confirmation; never run benchmarks as agent unless explicitly approved.

## Decision gates

- **G1 - Plan clearance:** CLEARED by user ("go ahead and execute the plan"). 
- **D1 - FT scenario:** CONFIRMED. Worklog/behavioral SLM fine-tuning showcase (Nivara-native, deterministic synthetic data), as captured above. Doc-QA memorization is not the showcase.
- **D2 - "FT SLMs on iGPUs" scope:** **D2a selected, no compromise on end state.** User direction: "we should have Silk and CPU parity; no compromise for the release and going forward". Execution may be CPU-first or Silk-first, but when the whole LoRA plan is executed, CPU/Silk parity is required for release readiness. Implement real Silk causal-LM LoRA training for Qwen2.5-0.5B on iGPU with CPU/Silk parity. Requires `rms_norm_backward` kernel + backward chain + Qwen/Llama-family Silk trainer path + parity tests. No false iGPU claim; no release without parity.
- **D3 - Where LoRA module lives:** CONFIRMED. Keep `LoRALinear`/serializer in core; SFT dataset/training CLI in `samples/NivaraFineTuning`; shared helpers only when clearly reusable.
- **D4 - Data generation:** CONFIRMED. Deterministic synthetic generator, no LLM required, seeded, provenance header, refuse overwrite without `--force`.
- **D5 - Genericity (user direction):** CONFIRMED. Implementation must be generic over Llama-family causal LMs; Qwen2.5-0.5B is headline, existing SmolLM verifies genericity.

## Implementation Plan (GitHub issues for order + tracking)

Issues to create after G1 clearance (use TASKS-TEMPLATE style in issue bodies where useful). The issue numbers below are placeholders until created; replace with real `#NNN` after creation.

| Order | Issue | Depends on | Parallel? | Notes |
|---:|---|---|---|---|
| 1 | **Plan/grounding: reshape LORA.md + docs/TODO tracking** | G1 | no | Update `docs/plan/LORA.md` to reflect headline pivot; keep reusable technical sections; record decisions. |
| 2 | **Reconcile untracked ft-data artifacts** | 1 | no | Inspect `FtDataBuilder.cs`/`FtDataModels.cs`; either integrate into new plan or remove/move with explanation. Do not silently drop. |
| 3 | **Behavioral SFT dataset format + deterministic generator** | D1,D4 | after 1 | Worklog scenario: synthetic JSONL/TSV, system+user+assistant, prompt-masked labels, deterministic seed, train/val/test split, provenance header. |
| 4 | **Causal LM LoRA SFT trainer (CPU reference)** | 3, D1 | after 3 | Generic Llama-family LoRA SFT (AutoDiff/AdamW, prompt masking, checkpoint best-by-val, memory guard, deterministic). Headline run: Qwen2.5-0.5B; generic verification: SmolLM. |
| 5 | **Silk/iGPU headline path (D2a, no compromise)** | 3, D1 | can run CPU-first or Silk-first; parity required before G2/release | Real Silk causal-LM LoRA training: `rms_norm_backward` + backward chain + Llama-family/Qwen Silk trainer path + CPU/Silk parity tests. Headline Qwen2.5-0.5B, verify genericity on SmolLM where feasible. No false iGPU claim. |
| 6 | **NivaraFineTuning CLI + report** | 4,5 | after 5 | `--mode ft-lora` / `--mode ft-eval` (or equivalent) for generic Llama-family SLMs; headline Qwen2.5-0.5B. Prints param count, memory budget, loss, accuracy, before/after; `--silk` runs real iGPU training path with CPU/Silk parity reporting. |
| 7 | **NivaraChat adapter consumption touchpoint (optional thin)** | 6 | parallel with 8 | Load adapted model/adapter for interactive demo; do not disturb RAG modes. |
| 8 | **Tests + docs + showcase README** | 3-7 | after impl | Unit tests (parity, frozen base, grad check, overfit sanity, determinism, round-trip, prompt masking). Update docs/LORA.md, NivaraFineTuning README, sample README. DocsCoveragePins/ADR index only if docs count/citations change. |
| 9 | **G2 review + PR** | all | no | Branch review whole + against TODO; fix gaps; push/create PR only with confirmation. |

## Proposed file areas (subject to G1/G2)

- `docs/plan/LORA.md` — reshape to headline/showcase plan; keep technical appendices.
- `docs/TODO.md` — this plan (removed after G2).
- `samples/NivaraFineTuning/` — SFT data/trainer/CLI modes for behavioral LoRA demo.
- `samples/Nivara.Samples/` — only if a truly shared helper is needed (chat template/loading already there).
- `src/Nivara/AutoDiff/Nn/` — only if core LoRA/module gaps found (LoRALinear already exists).
- `src/Nivara.Extensions/Silk/` — only if D2a/D2c selected (rms_norm_backward kernel, trainer).
- `tests/Nivara.Tests/AutoDiff/` — LoRA/SFT unit tests.
- `tests/Nivara.Tests/Gpu/` — Silk parity tests (skip when no OpenCL), only if D2a/D2c.
- `docs/` — LORA.md / README updates; watch DocsCoveragePins.

## Verification plan

- **Before each commit:** build touched projects; targeted tests for touched area.
- **Genericity check:** headline showcase runs on Qwen2.5-0.5B; run at least a targeted smoke/verification path on existing SmolLM to prove generic Llama-family support (no Qwen-only hardcoding in core trainer/adapter path).
- **Ask before full suite / long-running commands.** Full suite filter: `Category!=Performance&Category!=ReleaseReadiness&Category!=Gpu`.
- **Silk tests:** skip gracefully when no OpenCL device (`SilkFact`/existing pattern).
- **Docs changes:** run docs gate subset as AGENTS.md requires; check DocsCoveragePins if doc/citation count changes.
- **No benchmarks run by agent** unless human explicitly approves; provide commands for human.

## Risks / red flags

- **D2 scope risk (high):** True Silk causal-LM LoRA training needs rms_norm_backward + backward + trainer parity. This is non-trivial. I will not fake "on iGPU" by only running inference on iGPU.
- **Small-model quality risk (medium):** Behavioral SFT should be picked so small LoRA can win; metrics must be chosen accordingly.
- **Memory/runtime risk (medium):** CPU SFT over Qwen2.5-0.5B can be slow/heavy; need max-memory guard, small defaults, optional `--max-examples`.
- **Untracked ft-data files (low):** Must reconcile before building on top.
- **Plan drift risk:** LORA.md currently describes doc-QA comparison; reshaping it is a doc change with possible DocsCoveragePins/ADR implications.

## GitHub issues log

- [ ] (to be created after G1 clearance) Plan/grounding: reshape LORA.md
- [ ] (to be created) Reconcile untracked ft-data artifacts
- [ ] (to be created) Behavioral SFT dataset format + deterministic generator
- [ ] (to be created) Causal LM LoRA SFT trainer (CPU reference)
- [ ] (to be created) Silk/iGPU headline path (scope per D2)
- [ ] (to be created) NivaraFineTuning CLI + report
- [ ] (to be created) NivaraChat adapter consumption touchpoint (optional)
- [ ] (to be created) Tests + docs + showcase README
- [ ] (to be created) G2 review + PR tracking

## G1 grounding checklist (to complete before asking for clearance)

- [ ] microsoft-learn grounding for any new .NET tensor/span/BFloat16/Tokenizer APIs used by plan.
- [ ] code-memory navigation for LoRALinear, LlamaForCausalLM, LlamaLoader, Silk trainers/runners, NivaraFineTuning, chat templates.
- [ ] Confirm rms_norm_backward absence and identify kernel/trainer files to touch if D2a/D2c.
- [ ] Confirm LlamaForCausalLM forward/backward works under `GradientUtils.Grad()` for training (existing tests/patterns).
- [ ] State blast radius by component.
- [ ] Present D1-D4 decisions to human; do not assume.
