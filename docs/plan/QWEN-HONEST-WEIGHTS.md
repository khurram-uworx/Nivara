# Qwen honest weights — native loading (BF16 storage, F32 compute)

**Status:** Plan (execution-ready draft, 2026-11-16)  
**Owner:** (unassigned)  
**Issue:** [#1217](https://github.com/khurram-uworx/Nivara/issues/1217) “Qwen2.5-0.5B should default to native loading (BF16 at rest, F32 compute)”  
**Follows:** PR #1215 / commit `ffe79b06` (SmolLM native default) and `docs/plan/MODEL-LOADING-HONESTY.md` (Phase 1 = Qwen).  
**Related:** [#1219](https://github.com/khurram-uworx/Nivara/issues/1219) (Silk SmolLM BF16 decode crash, `CL_OUT_OF_RESOURCES` in `ArgMaxDevice` during `ForwardCachedArgmax` — OPEN).

## 1. Objective

Make Qwen2.5-0.5B load at its snapshot dtype by default: keep the BF16 checkpoint tensors as BF16 at rest and run F32 compute by widening in-register, with no second full-checkpoint copy. `--precision f32` remains available as the explicit widened-on-load path. This completes the Qwen half of “Model Loading Honesty” (Phase 1 in `docs/plan/MODEL-LOADING-HONESTY.md`).

## 2. Verified snapshot facts (local checkpoint, 2026-11-16)

Source: `samples/data/qwen2.5-0.5b-instruct/model.safetensors` + `config.json` (downloaded via `hf download Qwen/Qwen2.5-0.5B-Instruct ...`, HF token auth; `samples/data/` is gitignored working data).

- **Config:** `model_type=qwen2`, `architectures=["Qwen2ForCausalLM"]`, `torch_dtype=bfloat16`, `hidden_size=896`, `intermediate_size=4864`, `num_hidden_layers=24`, `num_attention_heads=14`, `num_key_value_heads=2`, `rms_norm_eps=1e-6`, `rope_theta=1_000_000.0`, `max_position_embeddings=32768`, `vocab_size=151936`, `tie_word_embeddings=true`, `use_cache=true`, `bos_token_id=151643`, `eos_token_id=151645`. No `attention_bias` key in config, but Q/K/V bias tensors are present in the checkpoint (see below) and `LlamaLoader`/`LLamaConfigLike` detect them via key presence.
- **Safetensors:** 290 tensors total, **all `BF16`**, no F32 tensors. Total BF16 payload = **988,065,536 bytes** (~942.3 MiB) over the BF16 tensor data; file size 988,097,824 bytes.
- **Widened-in-native set (compute-type parameters):** 121 tensor entries = 49 RMSNorm gammas (`model.layers.{0..23}.input_layernorm.weight`, `model.layers.{0..23}.post_attention_layernorm.weight`, `model.norm.weight`) + 72 QKV biases (`model.layers.{0..23}.self_attn.{q,k,v}_proj.bias`, shapes `[896]`, `[128]`, `[128]` per layer). Loaded via `StateDictLoader.LoadRMSNorm` / mixed `LoadLinear` bias path these become F32/`TModel` parameters: **286,208 bytes** when widened to F32 (49×896 + 24×(896+128+128) = 43,904 + 27,648 = 71,552 elements × 4). Embedding and all projection weights remain BF16 `Parameter<TWeight>`/storage arrays.
- **Residency target:** one BF16 copy of the checkpoint weights (+ 286,208 B widened norm/bias copies), i.e. ~942 MB at rest vs the prior widened-to-F32 load of ~1.88 GB. The implementation/test must assert “no second full-checkpoint copy” by reference/byte accounting against the real file, not by trusting a constant.

> **Parity fixtures note:** the repo references `qwen_tool_ids_py.bin` / `qwen_tool_logits_py.bin` as F32 tool-loop fixtures for Qwen; they were **not** found under `samples/data/` at capture time and were not part of the downloaded HF snapshot. The implementation must locate the canonical fixtures (or record/generate them in the same disciplined way as existing SmolLM fixtures, if that is the established practice) and pin the parity criterion to them; if only F32 references exist, native parity is asserted against the widened F32 path and/or those fixtures with an explicitly stated bound derived from measurement.

## 3. Existing infrastructure (reuse, no new kernels)

The mixed-dtype stack required for Qwen CPU already exists and is exercised by SmolLM (#1215):
- `samples/Nivara.Samples/LlamaForCausalLM.cs`: `LlamaForCausalLM<TModel,TWeight>` (:19), mixed `TiedHead` (:222), `ThrowIfMixedDtypeTraining` (:244), `LlamaLoader.Load<TModel,TWeight>` (:526) with bias detection via `config.AttentionBias || tensors.ContainsKey("model.layers.0.self_attn.q_proj.bias")`.
- `src/Nivara/AutoDiff/Nn/Linear.cs`: `Parameter<TWeight> weight`, `Parameter<TModel>? bias`, mixed forward via `MixedMatMulTransposedB`.
- `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs` (`<TModel,TWeight>`), `LlamaFusedKernels.DecoderBlockFused` (default `true`) with `MixedMatMulTransposedB` and `ParamSpan` bias handling.
- `src/Nivara/AutoDiff/Nn/Embedding.cs`: mixed-target `Embedding<TModel,TWeight>`.
- `src/Nivara.Extensions/AI/StateDictLoader.cs`: mixed-target overloads binding checkpoint arrays by reference (`LoadEmbed` :200, `LoadLinear` :214) and widening bias to `TModel` (:230-232); `LoadRMSNorm` (:184) converts to `TModel`.
- `samples/NivaraInference/Program.cs`: lazy `F32()` (:696) and single-dtype loads — `tensorsBf16` when `bf16 || native` (:709), `tensorsHalf` when `fp16` (:720).

Qwen differs from SmolLM in ways the plan must respect: QKV biases are present (widened to `TModel`), fused decode/prefill is on by default, `qwen distill` enters `GradientUtils.Grad()` (mixed dtypes throw — inference-only), and tool-loop parity is compared against F32 references.

### .NET 11 BFloat16 (microsoft-learn, Nivara targets .NET 11)

`System.Numerics.BFloat16` implements `IBinaryFloatingPointIeee754<BFloat16>` (narrowing, widening to float/double). .NET 11 adds `BitConverter.GetBytes(BFloat16)`, `BitConverter.ToBFloat16(byte[],int)` / `ToBFloat16(ReadOnlySpan<byte>)`, `BFloat16ToInt16Bits`, `BFloat16ToUInt16Bits`, `Int16BitsToBFloat16`, `UInt16BitsToBFloat16`; generic `Complex<T>` supports BFloat16; .NET 11 JIT uses hardware FP16 (F16C/AVX10.1) for `Half` (distinct from BF16). **Implication:** rely on the existing `IBinaryFloatingPointIeee754`-generic / `SafeTensorsLoader.Read<BFloat16>` paths used by SmolLM; introduce new BitConverter BFloat16 APIs only if a concrete local need is proven.

## 4. Implementation plan

### 4.1 `samples/NivaraInference/Qwen.cs` — dual-type surface

**Goal:** mirror `RunSmolLMCore<TModel,TWeight>` / `RunSmolLMNative` so Qwen runs as `<float,float>` (widened F32) or `<float,BFloat16>` (native).

- **`LoadModel`** (:218): retype to `<TModel,TWeight>` returning `(LlamaForCausalLM<TModel,TWeight>, LlamaConfig, Gpt2BpeTokenizer)` and call `LlamaLoader.Load<TModel,TWeight>(config, tensors)`. Keep `LoadConfigAndTokenizer` (:230) unchanged.
- **Generation:** make `Generate` (:337), `GenerateCore` (:349), `RunDecodeBenchmark` (:858), `TimeGeneration` (:898) generic over `<TModel,TWeight>` with `LlamaForCausalLM<TModel,TWeight>` / `LlamaKVCache<TModel>`. Cold reloads inside timed paths must reload the same `<TModel,TWeight>` from the same storage-typed tensor dictionary.
- **Logits / fixture diff:** generalize `ForwardLastLogits` (:630) from `float,float` to `<TModel,TWeight>` returning `float[]` (convert `TModel` logits to float for comparison) so numeric diff works for native and widened runs without materializing widened weights.
- **Run entry points** (`RunTools` :644, `RunBenchmark` :774, `RunPlainBenchmark` :793, `RunPlain` :807, `RunSyntheticBenchmark` :834): accept tensor dictionaries in the run’s storage dtype and instantiate the correct model:
  - native/default: `<float,BFloat16>` over BF16 tensors (storage preserved, F32 activations/compute).
  - `--precision f32`: `<float,float>` over `F32()` tensors.
  - define `NativeQwenPrecision` display label (e.g. `BFloat16 storage / F32 compute`, paralleling `NativeSmolLMPrecision` in `Program.cs`) and thread it through `PrecisionLabel`/output.
- **Tools parity:** keep the authoritative id/logit diff against the existing F32 `qwen_tool_*_py.bin` fixtures runnable in F32 mode. For native mode, assert greedy decode/token ids (and logits/argmax where applicable) match the widened F32 path and the fixtures over the same fixed trajectory — **prefer exact token-id equality on 32 greedy tokens**; if BF16-stored weights introduce a measured logit delta, codify an explicit numeric bound derived from observation (do not guess). Do not weaken the existing F32 fixture check.
- **Distill (inference-only mixed dtypes):** `qwen distill` builds gradients and cannot run mixed (`ThrowIfMixedDtypeTraining` inside `GradientUtils.Grad()`). Enforce a **named, specific** guard in the Qwen distill entry and/or CLI: reject native/mixed with a clear message directing `--precision f32` (e.g. “qwen distill requires widened F32 weights; mixed BF16-storage/F32-compute is inference-only — rerun with --precision f32”). Never let it fall through to the generic training exception.
- **Synthetic weights:** keep `--synthetic-weights` coherent with the selected precision (synthesize in the storage dtype for native or document/keep uniform as today) and ensure it does not silently create a widened duplicate.

### 4.2 `samples/NivaraInference/Program.cs` — defaults, guards, dispatch, help

- **Default precision** (:333-337): when `!precisionSet`, default `qwen` to `native` (like SmolLM). Resolve `qwen distill` explicitly: either reject bare `qwen distill` asking for `--precision f32`, or accept and force F32 with a loud named notice — **prefer explicit guard/rejection over a silent override** (record the chosen behavior in help text and tests).
- **Allow native for qwen** (:626-630): replace `if (native && !isSmolLM)` with a check that permits models with a native path (SmolLM, Qwen) and rejects others with an updated, accurate message. Update help/usage strings (:406-414, :440, :459-465) to describe `native` for qwen and that distill is F32-only.
- **Narrow-precision policy** (`ConsumesNarrow` :85, guards :657-663): make qwen honor the existing precision arg as SmolLM does. Support `native` (default) and `f32` per acceptance; keep or refine `bf16`/`fp16` handling with precise, reason-accurate rejections for modes/dtypes that remain unsupported (e.g. Half/fp16 range/precision concerns for qwen, if unchanged). Ensure each rejection fires for its own stated reason (not an earlier guard), to satisfy `InferenceCliArgumentTests`.
- **Dispatch** (:854-868, and `plain` paths): route `qwen` tools/benchmark/plain to `<float,BFloat16>` over `tensorsBf16` when native, `<float,float>` over `F32()` when `--precision f32`; route `qwen distill` to the F32 path with the named guard.
- **Silk + qwen (#1219, OPEN — blocking for BF16 device decode):** current logic (:757-764) picks BF16 tensors/runner when `bf16` else F32. Because native loads `tensorsBf16`, a naive flip would run bare `qwen --silk` on the BF16 cached-decode path, which exercises the same shared BF16 quartet implicated in #1219 (`GemmGemvAddBf16`, `RmsNormGemvBf16`, `MlpBlockBf16`, `GatherBf16` / `DecodeGqaAttentionOproj` dispatch) and has no BF16 Qwen decode regression coverage. **Required policy (record in PR):** keep `qwen --silk` on the known-good F32 device path when precision resolves to native/default — either coerce to F32 with an explicit diagnostic naming #1219, or reject native-on-Silk until #1219 is fixed. **Do not silently exercise the broken BF16 decode path.** Only enable native/BF16 Qwen Silk once #1219 is resolved and a BF16 cached-decode regression passes on real hardware.
  - Note: `SilkQwenGpuRunner.Bf16.cs` uploads/applies Q/K/V biases unconditionally and uses `hasBias: true` in fused slots (Qwen always has attention bias); this is consistent and not the SmolLM conditional-bias defect. No change there without evidence.

### 4.3 `samples/NivaraChat/Qwen/*` (alignment)

- `QwenChatClient.cs` currently holds `LlamaForCausalLM<T,T>`; `QwenMode.cs` exposes `--precision f32|bf16` and loads F32 for the f32 path. Extend to accept `LlamaForCausalLM<TModel,TWeight>` / `QwenChatClient<TModel,TWeight>` and load BF16 tensors for native/default, mirroring CLI. Update precision parsing/help (`f32|bf16|native` as appropriate) and keep Silk chat on the F32 device path while #1219 blocks BF16 decode.
- **Scope:** deliver the minimum needed for the chat sample to build and remain correct with the new default; defer broader chat refactors as follow-ups if they exceed #1217.

## 5. Tests

### 5.1 `tests/Nivara.Tests/Inference/QwenNativeCheckpointTests.cs` (new; template `SmolLMNativeCheckpointTests.cs`)

Real-checkpoint, skip-when-absent (CI has no weights/fixtures). Resolve `samples/data/qwen2.5-0.5b-instruct` from `RepoRoot`; `Assert.Ignore` with a clear reason if checkpoint/tokenizer/required fixtures are missing.

1. **Byte-identical load:** tensors returned by the native/BF16 read are identical to the safetensors payload (no load-time dtype translation). Assert against the file via the same approach as SmolLM (dtype map/count = 290 BF16; payload/array identity as appropriate).
2. **Reference identity and widened set:** `LlamaLoader.Load<float,BFloat16>` keeps BF16 checkpoint arrays by reference in stored-weight parameters (`Unsafe.AreSame`) for embed + all projection weights (169 stored weights, mirroring SmolLM’s accounting but recomputed for Qwen’s 290 tensors / 121 widened). Only RMSNorm gammas (49) and Q/K/V biases (72) become widened `Parameter<float>` copies — assert the exact widened key set/count/byte delta (286,208 B F32) derived from the snapshot. Assert `LlamaFusedKernels.DecoderBlockFused` is true for the fused parity path. Prove no second full-checkpoint copy (resident = BF16 payload held once + widened copies), not by magic numbers.
3. **Greedy parity (32 tokens):** native `<float,BFloat16>` matches widened `<float,float>` and the existing tool ids/logits F32 fixtures on the fixed prompt/trajectory with KV cache behavior consistent with references. Token-id equality required on the fixed greedy sequence; logit comparison uses the F32 fixtures with an explicit measured bound if needed. Run on a machine with the checkpoint present before merge.

Match SmolLM test structure/categorization (`[NonParallelizable]`, cached holders, disposal). If runtime exceeds the 2s filtered-suite budget when data is present, use the established local-only/release-readiness categorization consistent with existing checkpoint tests (do not silently slow CI).

### 5.2 `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`

- Accept `qwen --precision native` for supported qwen modes (reach model-file lookup / appropriate next guard), removing qwen from the smollm-only native-reject cases; keep non-native models rejected with the updated message.
- Update `QwenPrecision_Rejected` / narrow-precision tables so each case asserts its own guard (native/f32 accepted where specified; unsupported dtype/mode combos rejected for the correct reason).
- Add cases for `qwen distill` + native hitting the **named** guard (not generic mixed-training throw, not model lookup).
- Add case(s) for `qwen --silk` under the chosen #1219 policy (explicit coerce/reject, not a crash path).
- Refresh the file’s coverage comment to include the expanded qwen precision/mode surface.

### 5.3 Silk decode (deferred by #1219)

Do not claim or add a passing BF16 Qwen Silk decode test while #1219 is open. If #1219 is fixed in-scope, add a `[Category("Gpu")]` BF16 cached-decode regression for Qwen (real hardware/checkpoint, skip when unavailable). Otherwise track as a follow-up gated on #1219.

## 6. Documentation and pins

- **`docs/QWEN.md`**: update load-path/precision (~:48-59) to native default (BF16 storage, F32 compute, ~942 MB at rest) vs prior ~1.88 GB widened load; document `--precision f32` override, distill F32-only guard, and `--silk` behavior under #1219. Keep synthetic-weight and benchmark-command guidance accurate.
- **`samples/NivaraInference/README.md`**: update Qwen dtype/precision rows and examples (native default, `f32`, distill guard, Silk note).
- **`samples/NivaraChat/README.md`** / `QwenMode` help: align if the chat surface changes.
- **`docs/plan/MODEL-LOADING-HONESTY.md`**: mark Phase 1 progress/status consistent with repo convention.
- **Pin/coupling discipline (same commit as the triggering change):** `docs/*.md` add/remove → `DocsCoveragePins.ExpectedDocumentCount`; citation moves/adds → citation pin counts; ADR add → ADR index entry (`AdrIndexTests`); new/changed gated ```csharp fences → snippet gate/ADR-005. For any `docs/` change run `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` and `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore` before pushing.

## 7. Verification (local, pre-PR)

- **Build:** `dotnet build Nivara.slnx -c Release`.
- **Format:** `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- **Targeted tests (autonomous):**
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`
  - `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~QwenNativeCheckpointTests"` (requires local checkpoint/fixtures for a real pass; skip is expected otherwise)
  - Regress shared path if touched: `FullyQualifiedName~SmolLMNativeCheckpointTests|SmolLMF32CheckpointTests`
  - Docs gate if docs changed: `FullyQualifiedName~Nivara.Tests.Docs`
- **Manual/acceptance (machine with checkpoint):** bare `qwen`/`qwen tools`/`qwen` benchmark run native with ~942 MB-at-rest residency (no widened duplicate) and match 32-token greedy parity; `--precision f32` reproduces prior results; `qwen distill` hits named guard on native and succeeds on `--precision f32`.
- **Benchmarks:** human-run only per `docs/QWEN.md` § “Benchmark commands for agents (human-run, too slow for agents)”; provide commands, have human run to `data\qwen-bench\`, then read outputs. Do not run or poll benchmarks in-session.

## 8. Commit slicing (one reason per commit)

1. Runtime: `Qwen.cs` + `Program.cs` (+ minimal `samples/NivaraChat/Qwen/*` compile/runtime alignment) — `<TModel,TWeight>` retype, native default, `f32` override, distill named guard, Silk/#1219 policy.
2. Tests: `QwenNativeCheckpointTests` + `InferenceCliArgumentTests`.
3. Docs/pins: `docs/QWEN.md`, `samples/NivaraInference/README.md`, plan-status/pin arithmetic as required.

## 9. Open decisions / follow-ups

1. **`qwen --silk` + native policy (#1219):** coerce-to-F32-with-diagnostic vs reject-until-fixed. Recommend coercion to F32 with explicit #1219 diagnostic so `qwen --silk` remains usable.
2. **`qwen distill` default under native:** reject-and-ask-for-`--precision f32` vs force-F32-with-loud-notice. Recommend explicit rejection/guard.
3. **Parity criterion details:** confirm token-id equality vs measured logit bound once fixtures/checkpoint run is executed; locate or (re)generate canonical `qwen_tool_{ids,logits}_py.bin` references per established practice.
4. **Phases 2–4 (Laya, hygiene, Chat propagation):** remain follow-ups per `docs/plan/MODEL-LOADING-HONESTY.md`, except minimal Chat compatibility needed here.
5. **Silk BF16 decode:** unblock and regression-test Qwen after #1219 is fixed.
