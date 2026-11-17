# Model Loading Honesty — Expansion Plan

**Status:** Plan (draft, 2026-11-16)
**Owner:** (unassigned)
**Related:** PR #1215 (`khurram/smollm`) "Load SmolLM at its snapshot dtype and make native the default"; issue #1219 (Silk SmolLM BF16 decode crash); `docs/SMOLLM.md`, `docs/BFLOAT16.md`, `docs/SILK.md`.

## 1. What "Model Loading Honesty" means

Shipped in PR #1215 for SmolLM-135M:

1. **Load exactly one dtype.** A run never holds both a F32 map and a BF16 map of the same checkpoint. `Program.cs` now uses a lazy `F32()` accessor so the F32 map is materialized only on first use.
2. **Keep weights in their checkpoint (storage) dtype** when compute runs in a wider dtype, widening in-register instead of holding a widened copy. SmolLM's default is now `native` = BF16 storage / F32 compute, halving resident weight memory (~256.6 MB vs ~513.1 MB) at ~1.26× the latency of pure F32.
3. **Default to the snapshot dtype** for models whose checkpoint is narrow, instead of silently widening to F32 at load.

The mechanism is already generic: `LlamaForCausalLM<TModel, TWeight>`, `Linear<TModel, TWeight>`, `Bert*<TModel,TWeight>`, `ModernBert*<TModel,TWeight>`, `Laya*<TModel,TWeight>` all accept independent compute (`TModel`) and storage (`TWeight`) types, and `GradKernels.MixedMatMul*` / `NarrowFloatKernels` widen in-register. `SafeTensorsLoader.Read<T>` translates on-disk dtypes to `T` at load.

This document plans the expansion of (2) and (3) to the other supported models.

## 2. Checkpoint dtypes

| Model | On-disk dtype | Native candidate? |
|---|---|---|
| SmolLM-135M | BF16 (all 272 tensors) | Done (PR #1215) |
| Qwen2.5-0.5B | BF16 (all 290 tensors) | Yes — strong candidate |
| Laya (ModernBERT-large + decision head) | F16 (205 tensors) + 1 F32 (`temperature`) | Yes — essentially pure F16 |
| ModernBERT-large | F32 | No — already honest |
| MiniLM / DistilBERT / DistilBERT-SST | F32 | No — already honest |
| MobileNetV2 / ResNet-18 | F32 (+ int64 indices) | No — no narrow path; vision models are F32-only |

## 3. Phased plan

### Phase 1 — Qwen (highest value, lowest risk) — implemented on branch `khurram/1217` (#1217)

Qwen is BF16 on disk and Llama-family. Work:

- Add `LlamaForCausalLM<float, BFloat16>` (F32 compute, BF16 storage) variants of `Run`, `RunPlain`, `RunTools`, `RunBenchmark` in `samples/NivaraInference/Qwen.cs`, mirroring `RunSmolLMCore<TModel,TWeight>` / `RunSmolLMNative`. `LoadModel` already calls `LlamaLoader.Load<T,T>`; add a native variant `LoadModelNative` calling `LlamaLoader.Load<float, BFloat16>`.
- `RunDistill` stays F32 (teacher generation only; the student is a tiny BOW classifier, so the memory win is small). Document this exception.
- CLI: allow `--precision native` for `qwen`, default to `native` (like `smollm`), reject `native` where the run is F32-only (e.g. distill, vision). Update `InferenceCliArgumentTests` accordingly.
- Silk: Qwen already has a BF16 Silk path (`SilkQwenSession` with BFloat16 tensors). #1219's crash was SmolLM-only; Qwen's BF16 cached decode was already correct. Resolved by #1222: `qwen --silk` now runs the BF16 device path on native/default (BF16 storage / F32 compute); an explicit `--precision f32` keeps the F32 device path. `distill` stays F32.
- NivaraChat: `QwenChatClient` loads via the same dispatch; ensure the native default propagates. Silk chat client is session-based and dtype-agnostic, so it inherits once dispatch changes.

### Phase 2 — Laya

Laya's checkpoint is effectively pure F16, so native = `Laya.Run<float, Half>` (F32 compute, Half storage).

- Add a native entry point in `samples/NivaraInference/Laya.cs` (F32 compute / Half storage), analogous to SmolLM's `RunSmolLMNative`. `Laya.Run<TModel,TWeight>` already supports mixed dtype.
- The single F32 `temperature` tensor must be handled (narrowed to Half at load, or read separately as float). Precision impact is negligible but document it.
- Silk: Laya has no native Half path on Silk (fp16 is rejected on Silk; only bf16 is supported, and Laya's checkpoint is F16, not BF16). So Laya native is CPU-only; Laya Silk stays F32/bf16 (widened). Document this asymmetry, mirroring SmolLM's CPU-only `ab` mode.
- NivaraChat: check whether any Laya chat path loads Half; expand if present.

### Phase 3 — Hygiene (all models, low risk)

Cherry-pick the branch's "load exactly one dtype" pattern (`F32()` lazy accessor, no simultaneous F32+BF16 maps) into MiniLM / DistilBERT / ModernBERT / vision dispatch in `Program.cs`. These are F32 checkpoints, so there is no memory win, but it removes widened-duplicate holding and keeps dispatch honest. Apply the same to NivaraChat.

### Phase 4 — NivaraChat

Ensure the native default propagates to `--smollm` and (after Phase 1) `--qwen` chat clients, and that the Silk chat clients handle the native default (see #1219).

## 4. What NOT to expand

- ModernBERT / MiniLM / DistilBERT: F32 on disk, so "native" == F32. No default change; already honest.
- Vision models: F32 + int64; no narrow path exists, and `--silk --precision fp16` is already rejected.
- Pure-Half compute (`Laya.Run<Half,Half>`): keep as opt-in `--precision fp16`; do not make it the default. The memory win comes from storage, not compute, and F32 compute is far better.
- Do not refactor Laya/ModernBERT to per-tensor mixed storage dtypes; the single `TWeight` is sufficient because Laya's checkpoint is effectively uniform F16.

## 5. Prerequisites / blockers

1. **Issue #1219** — Silk SmolLM BF16 decode crash (`CL_OUT_OF_RESOURCES` in `ArgMaxDevice` during `ForwardCachedArgmax`) must be fixed before Qwen native defaults to BF16 on Silk; otherwise `qwen --silk` (default) breaks. Until then, keep Silk defaulting to F32 for Qwen (override the native default when `--silk` is set).
2. Add BF16 Silk decode regression tests (none exist today) so the path is covered.
3. Decide whether Laya's `temperature` scalar needs special-casing when loading as Half.

## 6. Suggested first commit

Qwen only, CPU, behind `--precision native` (opt-in, not default yet), with `LoadModel<float, BFloat16>` and F32-compute variants of `Run` / `RunPlain` / `RunTools` / `RunBenchmark`. Default flip and Silk come after #1219 is resolved and BF16 Silk decode is tested. This keeps the change reviewable and avoids breaking the Silk default.

## 7. References

- PR #1215 (`khurram/smollm`) — "Load SmolLM at its snapshot dtype and make native the default".
- Issue #1219 — "Silk SmolLM BF16 decode path crashes (CL_OUT_OF_RESOURCES) in ArgMaxDevice".
- `docs/SMOLLM.md` § "Native loading" (to be added with the merge).
- `docs/BFLOAT16.md` — widen-in-register rationale and the SmolLM BF16 numbers.
- `docs/SILK.md` §13b — causal decoders and the BF16 packed-transport quartet.
- `src/Nivara/AutoDiff/Nn/Linear.cs`, `LlamaDecoderBlock.cs`, `LlamaFusedKernels.cs` — mixed-dtype matmul surface.
- `src/Nivara.Extensions/AI/SafeTensorsLoader.cs` — `Read<T>` dtype translation.
- `samples/NivaraInference/Program.cs` — dispatch tables (`SupportedModes`, `SilkModes`, `ConsumesNarrow`), `F32()` lazy accessor.

## 8. Synthetic weights (checkpoint-free runs)

Synthetic weights (fabricated, deterministic tensors shaped like a real checkpoint) let every supported scenario — CPU/Silk, native and explicit precisions, single-shot and `benchmark` — run without a multi-GB checkpoint. This is the same honesty property as (2)/(3): a run should be reproducible on a clean checkout, and "no checkpoint present" must not be a special case.

### 8.1 Qwen — done (#1217/#1222 + consistency commit)

- `Qwen.SynthesizeTensors(LlamaConfig)` fabricates the exact tensor keys `LlamaLoader.Load<TModel,TWeight>` expects, gating q/k/v biases on `AttentionBias` (Qwen-style vs SmolLM/Llama-style). Generic variant `SynthesizeTensors<T>` (PR consistency work, commit `db9d5e0d`) generates directly in the storage dtype, so no widened F32 pool is held alongside the storage map.
- CLI: `--synthetic-weights` accepted for `qwen` in the default single-shot run and `benchmark`; routes to `RunSyntheticBenchmark<float,float>`, `<BFloat16,BFloat16>`, `<float,BFloat16>` (native), and `RunSyntheticBenchmarkSilk(plain, bf16||native)`.
- Honors `--precision` (f32 / bf16 / native) and `--silk`; Silk uses BF16 device path when `bf16 || native`, F32 otherwise.
- Tests: `InferenceCliArgumentTests` pins scope; `SynthesizeTensorsBiasGatingTests` pins bias gating; Silk BF16 cached-decode parity covered.

### 8.2 SmolLM — gap (issue #1254)

- Synthetic benchmark helpers already exist (`RunSmolLMSyntheticBenchmark`, `RunSmolLMSyntheticAb`, `RunSmolLMSyntheticBenchmarkSilk`) and call `Qwen.SynthesizeTensors(SmolLMSyntheticConfig)`, but are **dead code**: no CLI mode reaches them, and `--synthetic-weights` is rejected for `smollm`.
- Work: wire the flag (default + `benchmark`), route to the runners for CPU (f32/bf16/native) and Silk, honor `ConsumesNarrow`/precision, update the CLI gate. Uses `SynthesizeTensors<T>` directly in storage dtype (BF16 native).
- Silk: SmolLM BF16 path is CPU-only `ab` historically; confirm Silk synthetic uses BF16 device path where supported.

### 8.3 Laya — gap (issue #1255)

- No synthetic generator exists for Laya/ModernBERT. Needs fabrication of both ModernBERT encoder tensors (prefix `encoder`) and decision-head tensors, directly in storage dtype (Half for native).
- Needs a self-contained minimal encoder + agent config (or embedded defaults) so the run needs no model directory/tokenizer.
- Silk: Laya has no native Half Silk path (fp16 rejected on Silk; only bf16), so synthetic on Silk stays F32/bf16 (widened); document the asymmetry.
- The single F32 `temperature` scalar needs special-casing (keep as float or narrow to Half and document).

### 8.4 Follow-ups

- Issue #1254 — SmolLM synthetic weights wiring.
- Issue #1255 — Laya synthetic weights support.
- After both land, update `InferenceCliArgumentTests` coverage comment and `docs/QWEN.md`/`docs/SMOLLM.md`/`docs/LAYA.md` synthetic-weight sections.

