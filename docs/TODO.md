# TODO — #1217 Qwen2.5-0.5B native loading (BF16 storage, F32 compute)

**Branch:** `khurram/1217` (off `main`)  
**Issue:** [#1217](https://github.com/khurram-uworx/Nivara/issues/1217) — Qwen2.5-0.5B should default to native loading (BF16 at rest, F32 compute)  
**Detailed plan:** `docs/plan/QWEN-HONEST-WEIGHTS.md` (captured) and `docs/plan/MODEL-LOADING-HONESTY.md` Phase 1  
**Follows:** PR #1215 (SmolLM native default, merged `ffe79b06`)

As each task executes: if deferred work/concern outside this plan is found, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record it in the GitHub issues log — do not rely on memory.

## Problem

Qwen2.5-0.5B (`samples/data/qwen2.5-0.5b-instruct/model.safetensors`, BF16 on disk, 290 tensors / 988,065,536 B payload) is currently widened to F32 at load in the CLI/inference paths (`Qwen.LoadModel<T>` returns `LlamaForCausalLM<T,T>`, dispatch uses `F32()`/uniform dtypes), holding ~1.88 GB instead of ~942 MB at rest. The mixed-dtype stack (`LlamaForCausalLM<TModel,TWeight>`, `Linear/Embedding/LlamaDecoderBlock<TModel,TWeight>`, `LlamaLoader.Load<TModel,TWeight>`, `StateDictLoader` mixed overloads, `GradKernels.MixedMatMul*`, fused decode) already exists and is exercised by SmolLM; Qwen needs to be retyped and defaulted to native (BF16 storage, F32 compute) like SmolLM.

Verified snapshot facts (local, 2026-11-16): 290 BF16 tensors, all BF16; widened-in-native set 121 tensors = 49 RMSNorm gammas + 72 QKV biases (`layers.*.self_attn.{q,k,v}_proj.bias`, attention bias present by key) = 286,208 B widened to F32; config qwen2, 24 layers, hidden 896, heads 14/2, intermediate 4864, vocab 151936, tied embeddings.

## Proposed changes

### 1) `samples/NivaraInference/Qwen.cs` — dual-type surface
- Retype `LoadModel` to `<TModel,TWeight>` returning `LlamaForCausalLM<TModel,TWeight>` via `LlamaLoader.Load<TModel,TWeight>` (keep `LoadConfigAndTokenizer`).
- Retype `Generate`, `GenerateCore`, `RunDecodeBenchmark`, `TimeGeneration` to `<TModel,TWeight>` (`LlamaKVCache<TModel>`, cold reload same dtypes/dict). Generalize `ForwardLastLogits` to produce `float[]` from `<TModel,TWeight>` logits.
- Retype run entry points (`RunTools`, `RunBenchmark`, `RunPlainBenchmark`, `RunPlain`, `RunSyntheticBenchmark`) to accept storage-typed tensor dicts and instantiate native `<float,BFloat16>` or widened `<float,float>` per selected precision; add `NativeQwenPrecision` display label.
- Keep F32 fixture-diff path authoritative against existing F32 refs; for native assert greedy parity vs widened F32 path / fixtures per measured criterion (exact 32-token ids preferred; explicit measured logit bound only if needed). Do not weaken existing F32 checks.
- Add **named guard** on distill/mixed-grad paths: reject native/mixed with clear “inference-only; rerun with --precision f32” message (defense-in-depth; never fall through to generic `ThrowIfMixedDtypeTraining`).

### 2) `samples/NivaraInference/Program.cs` — defaults, guards, dispatch, help
- Default `qwen` to `native` when `!precisionSet` (parallel to SmolLM). Distill under native must not run mixed: explicit named guard/rejection (see G1 decision).
- Replace `native && !isSmolLM` rejection with allow-list for models with native path (SmolLM, Qwen); update help/usage (`--precision native` text, qwen distill/tools wording).
- Adjust `ConsumesNarrow`/narrow-precision guards so qwen honors `--precision f32` (and rejects unsupported dtype/mode combos for their own stated reasons). Update dispatch at qwen/plain switch (`:854-868`) and Silk qwen block (`:757-764`): native → `<float,BFloat16>` over `tensorsBf16`; f32 → `<float,float>` over `F32()`; distill → F32-only with named guard.
- **Silk + qwen (#1219 OPEN):** do not route qwen native/default onto BF16 cached-decode kernels. Apply explicit policy (see G1): keep qwen Silk on F32 device path when native selected with diagnostic naming #1219 (or reject) — never silent crash path.

### 3) `samples/NivaraChat/Qwen/*` — minimal alignment
- Extend `QwenChatClient`/`QwenMode` to support `LlamaForCausalLM<TModel,TWeight>` and native/default BF16 storage loading consistent with CLI; keep Silk chat off BF16 decode while #1219 open. Minimum scope to build/remain correct.

### 4) Tests
- **New** `tests/Nivara.Tests/Inference/QwenNativeCheckpointTests.cs` (template `SmolLMNativeCheckpointTests.cs`): skip when checkpoint/tokenizer/fixtures absent. Gates: (a) BF16 tensors byte-identical to safetensors payload; (b) model keeps BF16 arrays by reference for stored weights, widens only 49 norm gammas + 72 QKV biases (286,208 B F32, 121 tensors) — residency = BF16 payload once + widened copies, no second full-checkpoint copy, fused path asserted; (c) 32-token greedy parity native vs widened F32 (and fixtures when present) with stated criterion/bound.
- **Update** `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`: accept qwen native for supported modes; revise/remove qwen native-smollm-only and narrow-precision rejections to match new contract (each rejection for its own reason); add qwen-distill+native named-guard case; add qwen --silk + native policy case; refresh coverage comment.

### 5) Docs & pins
- `docs/QWEN.md`: native default load path (~942 MB at rest) vs prior ~1.88 GB, `--precision f32` override, distill guard, Silk/#1219 note, benchmark guidance.
- `samples/NivaraInference/README.md`: Qwen dtype/precision rows/examples.
- `docs/plan/MODEL-LOADING-HONESTY.md`: mark Phase 1/Qwen progress per convention (if appropriate).
- Recompute `DocsCoveragePins` / snippet/ADR gates only if doc/citation/fence counts change (same commit as trigger). Run docs namespace gate for any docs change.

## G1 — Grounding, blast radius, decisions (PENDING HUMAN CLEARANCE)

**Grounding (microsoft-learn + codebase):** .NET 11 `System.Numerics.BFloat16` implements `IBinaryFloatingPointIeee754<BFloat16>`; BitConverter BFloat16 methods added in .NET 11 (`GetBytes/ToBFloat16/BFloat16ToInt16Bits/Int16BitsToBFloat16` etc.). Existing generic `SafeTensorsLoader.Read<BFloat16>` and mixed `LlamaLoader/Linear/StateDictLoader` paths (used by SmolLM, #1215) are the correct reuse surface; no new kernels required for Qwen CPU. QKV biases present in snapshot and already handled by loader (`AttentionBias` key-detected) — widened to `TModel`.

**Blast radius:**
- Primary: `samples/NivaraInference/Qwen.cs`, `samples/NivaraInference/Program.cs`.
- Secondary/alignment: `samples/NivaraChat/Qwen/QwenChatClient.cs`, `samples/NivaraChat/Qwen/QwenMode.cs` (build/runtime consistency with new default).
- Tests: `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`, new `tests/Nivara.Tests/Inference/QwenNativeCheckpointTests.cs`; existing `tests/Nivara.Tests/AutoDiff/QwenInstructParityTests.cs` and `tests/Nivara.Tests/Qwen/SilkQwenToolsWeatherLoopTests.cs` must remain valid (skip behavior intact).
- Docs: `docs/QWEN.md`, `samples/NivaraInference/README.md`, possibly `samples/NivaraChat/README.md`, `docs/plan/MODEL-LOADING-HONESTY.md`, and pin counts if citations/fences/docs-in-scope counts move.
- Unchanged core libraries: `LlamaForCausalLM`, `Linear`, `StateDictLoader`, fused kernels already support mixed dtypes; Silk BF16 kernels unchanged (blocked by #1219).

**Decisions / red flags requiring human input before implementation:**
1. **`qwen --silk` when native/default (BF16 decode blocked by #1219):** recommend coerce effective precision to F32 for qwen+Silk+native with an explicit diagnostic naming #1219 so `qwen --silk` stays usable and never hits the broken BF16 cached-decode path. Alternatives: reject native-on-Silk until #1219 fixed. **Confirm preferred policy.**
2. **`qwen distill` and native default:** distill requires gradients (mixed dtypes throw). Recommend explicit named rejection when distill requested with native (and reject bare `qwen distill` if it would default native, instructing `--precision f32`) rather than silent F32 override. **Confirm preferred behavior.**
3. **Parity fixtures availability:** repo/tests reference `qwen_tool_ids_py.bin`, `qwen_tool_logits_py.bin` and other `qwen_*` prompt/id fixtures in the model dir, but they are **not present** locally (not in `samples/data/qwen2.5-0.5b-instruct` and not found elsewhere in repo/data). `tests/Nivara.Tests/AutoDiff/QwenInstructParityTests.cs` skips when absent. Acceptance for #1217 states “32 tokens of greedy decode on the real checkpoint match the widened F32 path and the existing tool ids/logits fixtures”. On this machine I can verify native-vs-F32 parity on the real checkpoint; I **cannot** validate against the PyTorch fixture bins without them. **How to proceed:** provide/locate fixtures, generate them (needs reference env/script), or accept native-vs-widened-F32 parity (+ fixture-diff when files exist) as the implemented/verified criterion?
4. **Narrow modes:** keep `bf16`/`fp16` semantics aligned with “honor existing precision arg as smollm does” (support `native`+`f32` per acceptance; preserve precise rejections for unsupported forms) — confirm if uniform `--precision bf16` for qwen benchmark/plain should also be supported now (like smollm) or limited to `native`/`f32`.

**Implementation must not proceed past G1 until human clears/amends the above.**

## Verification steps

- Targeted (ask before long runs): `dotnet build Nivara.slnx -c Release`; `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgumentTests"`.
- With local checkpoint present: `--filter "FullyQualifiedName~QwenNativeCheckpointTests"` and `--filter "FullyQualifiedName~QwenInstructParityTests"` (fixture-dependent tests may skip if bins absent).
- Regression if shared paths touched: `--filter "FullyQualifiedName~SmolLMNativeCheckpointTests|SmolLMF32CheckpointTests"`.
- Docs changed: `--filter "FullyQualifiedName~Nivara.Tests.Docs"` and snippet expectations as applicable.
- Manual acceptance on real checkpoint: bare `qwen`/`qwen tools`/`qwen benchmark` native path, residency drop, `--precision f32` parity, `qwen distill` named guard behavior, `qwen --silk` policy behavior.
- Benchmarks human-run only (`docs/QWEN.md` § benchmark commands).

## Planned commit list

1. `docs: plan #1217 Qwen native loading in TODO and plan doc` (this file + `docs/plan/QWEN-HONEST-WEIGHTS.md`) — **G1 clearance required before step 2**
2. `feat(qwen): retype inference to mixed TModel/TWeight and default native precision`
3. `feat(cli): honor qwen precision/native defaults, guards, and silk policy`
4. `feat(chat): align Qwen chat client/mode with native loading` (if needed for build/consistency)
5. `test(inference): add Qwen native checkpoint and CLI argument coverage`
6. `docs: update Qwen/CLI docs and pins for native loading` (+ pin arithmetic if required)
7. `docs: remove TODO.md — plan executed` (only after G2 clears)

(Commits may be split/merged to keep one logical change per commit; fixes from review become additional commits.)

## GitHub issues log

- [ ] (none yet) — create immediately if deferred/follow-up work is discovered during execution

## Blast radius summary

See G1 above. Core AutoDiff/Nn/loader/Silk kernels not modified; changes concentrated in inference sample CLI/Qwen implementation, chat Qwen integration, inference tests, and docs. Risk areas: precision/dispatch guard correctness, distill-grad safety, Silk policy under #1219, and parity criteria without full fixture set.
