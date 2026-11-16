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
- Silk: Qwen already has a BF16 Silk path (`SilkQwenSession` with BFloat16 tensors). Verify it end-to-end before making native the default there. **Blocked by issue #1219** (Silk SmolLM BF16 decode crash) — the same latent bug may affect Qwen's BF16 Silk decode. Until #1219 is resolved, keep Qwen Silk defaulting to F32 (override the native default when `--silk` is set).
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
