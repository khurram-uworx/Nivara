# Plan: load SmolLM at its snapshot dtype and run it correctly by default

Branch: `khurram/smollm`

## Problem

`NivaraInference` widens every checkpoint to F32 at load. `Program.cs:658` calls
`SafeTensorsLoader.Read(path)` unconditionally, and a `--precision bf16|fp16` run then loads a
*second* copy at native width (`Program.cs:667`, `:677`). For SmolLM-135M (all 272 on-disk tensors
are BF16) that means three resident copies of the same weights: the discarded F32 array (513 MB),
the BF16 array (269 MB) and — because `LlamaLoader` only calls `StateDictLoader.LoadLinearWidened`,
whose `TypeConverter.Convert` always allocates a fresh array (`TypeConverter.cs:40`) — a widened
copy of the BF16 array inside the model (269 MB).

The default is therefore not what the snapshot says, and the "native" run is not native either.

The compute problem is already solved, and this plan does not add a kernel. `GradKernels` exposes
mixed-precision span kernels (`MatMulTransposedB(ReadOnlySpan<float>, ReadOnlySpan<BFloat16>, …)` at
`GradKernels.cs:827`, the `Half` sibling at `:847`, plain `MatMul` at `:833`) and
`LlamaFusedKernels` forwards to them (`LlamaFusedKernels.cs:82`, `:86`). Silk has always worked this
way: every `*_bf16` kernel is `_bf16_act` — narrow weights resident, F32 activations
(`SilkKernels.cs:102-151`).

What blocks the CPU path is one helper: `LlamaDecoderBlock<TModel,TWeight>.WeightSpan`
(`LlamaDecoderBlock.cs:446-461`) allocates and widens a full F32 weight copy on **every** call when
`TWeight != TModel`. In fused decode that is seven projections across thirty layers — 210 widening
allocations per generated token. That is the memory-bandwidth tax this plan removes.

## Scope

One model: **SmolLM-135M**. It is the only model with both an F32 and a BF16 PyTorch reference
fixture on disk (`samples/data/compare_smollm_py_f32.bin` / `_bf16.bin` and the `_logits_` pair), so
"correct" is measurable in seconds per run. Qwen2.5-0.5B (BF16) and Laya (F16) keep today's default
until each has its own parity gate; Laya's only gate today is against the `laya==0.3.20` wheel on
F32-widened weights (`docs/LAYA.md:121`), so a native default there would be a silent quality
regression, not a policy change.

Native means **narrow storage, F32 activations**: weights stay bit-identical to the snapshot in
memory, dot kernels widen in-register. Not narrow arithmetic — `Vector<BFloat16>.IsSupported` is
false on .NET 11 (`docs/BFLOAT16.md:355`), which is why `docs/QWEN.md:57` widened at load in the
first place. This plan changes *when* we widen (per dot, in registers) and *what we keep* (the
snapshot bytes), not the maths.

## Proposed changes

### 1. Loader honesty (`src/Nivara.Extensions/AI/SafeTensorsLoader.cs`, `samples/NivaraInference/Program.cs`)

- Expose the dtype map the loader actually read: `ReadDtypes(string path)` returning
  `IReadOnlyDictionary<string, string>` keyed by tensor name, built from the same parsed header.
- Add the zero-hop `F16 → Half` cast in `ConvertF16<T>` (mirroring the existing
  `ConvertBF16ToBFloat16`): when `typeof(T) == typeof(Half)`, `MemoryMarshal.Cast<byte, Half>` and
  copy. Today a Laya-sized F16 native read walks 421M elements through `CreateChecked`.
- `Program.cs`: load exactly one dtype. The F32 read happens only when F32 is what the run will
  consume; `tensorsBf16`/`tensorsHalf` are unchanged in meaning.

### 2. Module seam (`samples/Nivara.Samples/LlamaForCausalLM.cs`)

- `LlamaForCausalLM<TModel, TWeight>` (base unsealed), plus
  `LlamaForCausalLM<T> : LlamaForCausalLM<T, T>` so `NivaraChat/Qwen/QwenChatClient`,
  `NivaraInference/Qwen.cs` and `LlamaForCausalLMPrefillTests` do not move. Exact precedent:
  `LlamaAliases.cs:5-15`.
- `LlamaLoader.Load<TModel, TWeight>` switches from `LoadEmbedWidened` / `LoadLinearWidened` to the
  zero-copy `StateDictLoader.LoadEmbed` / `LoadLinear<TModel,TWeight>` overloads
  (`StateDictLoader.cs:200-235`), which reference `w.Data` in place.
- `Embedding<TModel, TWeight>`, `LlamaKVCache<TModel>`, `RMSNorm<TModel>`, `RotaryEmbedding<TModel>`
  stay F32-activation side. Norm gammas and qkv biases are `Parameter<TModel>` by design and widen
  once at load (~140 KB for SmolLM) — that is intended, not a leak.
- `Forward(int[])` keeps the autograd path when `TModel == TWeight` (the existing grad-scope test
  must keep passing) and uses the mixed inference kernel when the types differ, throwing inside
  `GradientUtils.Grad()` with a message that says why.

### 3. The allocation that has to die (`src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`)

- Dispatch the seven matmul sites in `ForwardCachedFused` / `ForwardPrefillFused` through one
  helper: same types → `GradKernels.MatMulTransposedB<TModel>` (bit-identical to today),
  `float × BFloat16` / `float × Half` → the mixed overload, anything else → the widen fallback with
  a comment saying it is cold.
- `WeightSpan` collapses to a plain `Parameter<TModel>` accessor; its generic widening branch is
  deleted, so a mixed model can no longer widen silently.
- Tied-head GEMV in the sample routes through `LlamaFusedKernels.MatMulTransposedB(float, BFloat16)`.

### 4. CLI policy (`samples/NivaraInference/Program.cs`)

- No `--precision` → native for `smollm`. `--precision native` accepted as the explicit spelling.
  `f32`, `bf16`, `fp16` keep today's meaning exactly.
- Help text + the `--precision` reject message updated;
  `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` pins that surface and moves in the
  same commit (AGENTS.md coupling rule).
- Other models keep their current default, each with a one-line reason printed in help.

### 5. Gates and docs

- Native SmolLM run must: (a) assert per-tensor bit-identity against the snapshot bytes,
  (b) hold token-stream parity against the BF16 PyTorch fixture at ≥ 22/32 with cosine ≥ 0.999,
  (c) print and assert resident weight bytes ≤ 1.05× snapshot, (d) prove `--precision f32` still
  reaches 32/32 against the F32 fixture.
- Docs: `docs/SMOLLM.md` (default + honest "footprint, not CPU speed" line),
  `samples/NivaraInference/README.md`, `docs/BFLOAT16.md` where it describes widen-at-load. Then the
  Docs namespace tests and, if citations move, `DocsCoveragePins`.

## Verification

1. `dotnet build Nivara.slnx -c Release` after each commit.
2. Targeted tests per commit (ask before running): `LlamaForCausalLMPrefillTests`,
   `SafeTensorsLoader*Tests`, `InferenceCliArgumentTests`, `MixedPrecisionMatmulTests`,
   `LlamaForCausalLMTests`.
3. End-to-end, Release:
   - `dotnet run -c Release --project samples/NivaraInference -- smollm`
   - `dotnet run -c Release --project samples/NivaraInference -- smollm --precision f32`
   - `dotnet run -c Release --project samples/NivaraInference -- smollm --precision bf16`
4. If `docs/` changes: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`.
5. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan native SmolLM loading in TODO.md`
2. `Report safetensors dtypes and add the F16 zero-hop native read`
3. `Load each model at exactly one dtype in NivaraInference`
4. `Carry storage and compute dtype separately in LlamaForCausalLM`
5. `Dispatch fused Llama decode to mixed-precision kernels without widening`
6. `Default SmolLM to its snapshot dtype`
7. `Gate native SmolLM parity, bit-identity and resident weight bytes`
8. `Document native-by-default SmolLM loading` (+ `DocsCoveragePins` if required)
9. `docs: remove TODO.md — plan executed`

## Blast radius

- Core: `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs` (fused decode/prefill — used by SmolLM, Qwen
  and the Llama tests), `src/Nivara/Extensions/AI/SafeTensorsLoader.cs` (all checkpoint loading).
- Extensions: `StateDictLoader` (callers unchanged — the overloads already exist).
- Samples: `samples/Nivara.Samples/LlamaForCausalLM.cs` (consumed by `NivaraInference` and
  `NivaraChat`), `samples/NivaraInference/Program.cs` (CLI).
- Tests: `LlamaForCausalLMPrefillTests` pins fused-vs-per-op bit-exactness for the same-type path —
  that is the guard for the `WeightSpan` change. `InferenceCliArgumentTests` pins the CLI surface.
- Risk: a dispatch mistake in `WeightSpan` would change the F32 path's arithmetic. Mitigation: the
  same-type branch keeps the existing call verbatim and is covered by the bit-exact tests.

## GitHub issues log

(none yet — record deferred work here as it is discovered, per AGENTS.md)

## Probe lifecycle

No new probe. The measurement this plan needs is the existing SmolLM `benchmark` / `ab` modes plus
the fixture diff already in `ConsumeGenerated`; `tests/Nivara.SimdProbe` was checked first and has
no mode-shaped gap here (the question is end-to-end weight residency, not a kernel A/B).
