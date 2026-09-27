# SmolLM-135M-Instruct (HuggingFaceTB/SmolLM-135M-Instruct)

## Model overview

The **first causal LM / generative model** in the sample and the primary driver for the BF16 widening work ([docs/BFLOAT16.md](BFLOAT16.md)). It is a **BF16-native** Llama-family causal LM — all 272 on-disk tensors are `BF16` (269 MB), exercising the native `SafeTensorsLoader.Read<BFloat16>` zero-hop path (unlike the other 4 models, which are F32 on disk). The Nivara side runs the full stack in Nivara's AutoDiff engine over the model ops below and greedily decodes a response.

- **Config**: `hidden_size=576`, `intermediate_size=1536`, 30 layers, `num_attention_heads=9`, **`num_key_value_heads=3` (GQA)**, `hidden_act=silu` (gated FFN), RMSNorm (`eps=1e-5`), RoPE (`theta=10000`), `max_position_embeddings=2048`, `vocab_size=49152`, `tie_word_embeddings=true`
- **Tokenizer**: **GPT-2 byte-level BPE** (not SentencePiece — see the note below), chat variant (`<|im_start|>`/`<|im_end|>` template; bos `<|im_start|>`, eos/pad `<|im_end|>`), 49152-token vocab built from `vocab.json` + `merges.txt`
- **Tied LM head**: the input embedding weight is reused as the output projection — the checkpoint has no separate LM-head tensors
- **Reference fixture**: `Python/smollm_generate_reference.py` saves the token-id stream and final-position logits for diffing:

  ```bash
  python samples/NivaraInference/Python/smollm_generate_reference.py
  # -> samples/data/compare_smollm_py.bin, samples/data/compare_smollm_logits_py.bin
  ```

  The C# `smollm generate` mode diffs against these when present (see "Causal-LM generation" below).

> **Tokenizer correction (historical)**: this README previously listed SmolLM's tokenizer as SentencePiece. It is actually a **GPT-2 byte-level BPE** tokenizer (`tokenizer_class: GPT2Tokenizer`, `add_prefix_space: false`). The `Microsoft.ML.Tokenizers` BPE path cannot reproduce SmolLM's byte-level token IDs (every pre-tokenizer variant diverges at space-prefixed tokens), so a sample-local `Gpt2BpeTokenizer` (HF `bytes_to_unicode` map + GPT-2 regex + ranked greedy merges) implements the reader.

## How this model improved the library

Adding a causal-LM path surfaced capabilities the Nivara core library did not yet have. SmolLM was the driver, but each gap was filled as a **reusable, unit-tested addition to `src/Nivara`** (same forward + VJP + JVP + SIMD kernel shape as the ops the first four models exercise), then verified end-to-end against the PyTorch reference:

- **New core module — `RMSNorm<T>`** (`src/Nivara/AutoDiff/Nn/RMSNorm.cs`): Llama RMS normalization with a per-channel affine `gamma` (Llama normalizes by row root-mean-square, unlike the existing mean/var LayerNorm). Reuses the existing `RMSNormKernel<T>` and the `RMSNormKernel.PerRowRMSNormForward/Backward` span kernels; forward + input-grad + gamma-grad all wired.
- **New core activation — SiLU (Swish, `x·sigmoid(x)`)**: `Activation.Silu` + `ReverseGradOperations.Silu` (forward + VJP), `ForwardGradOperations.Silu` (JVP, `t_out = silu'(a)·t_a`), and `GradKernels.Silu/SiluGradient` (SIMD `TensorPrimitives` chain). Llama's gated FFN gates on SiLU rather than GELU.
- **New core op — RoPE (`RotaryEmbedding<T>`)** (`src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs`): precomputed cos/sin from `inv_freq = theta^{-2i/dim}`, Q/K rotary position embedding, with `GradKernels.RotaryForward/RotaryBackward` and the module's own graph op. **Layout bug found & fixed during verification**: the first implementation used the GPT-NeoX **interleaved-pairwise** rotation, but the Llama family uses HF **`rotate_half` (half-split)** — the wrong layout rotated Q/K so the logits were near-anti-correlated (cosine −0.92) and corrected to +0.24; the end-to-end F32 greedy match went 4/32 → 25/32 with byte-identical text through the matched prefix (the current F32 counts are in the Results tables below).
- **New core module — `LlamaCausalAttention<T>`** (`src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`): **GQA** (9 Q / 3 KV heads) via KV-repeat — `ReverseGradOperations.GqaRepeatKV` (VJP with `GradKernels.HeadRepeat`/`HeadRepeatBackward`) and `ForwardGradOperations.GqaRepeatKV` (JVP) — feeding a fused, causal-masked per-head attention loop.
- **New core module — `LlamaDecoderBlock<T>`** (`src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`): pre-norm self-attention + residual, then pre-norm **gated SiLU FFN** (`down(silu(gate)⊙up)`) + residual.

**Sample-scoped additions** (not core library — `samples/Nivara.Samples` / `Program.cs`): `LlamaForCausalLM<T>` (embed → 30 blocks → final RMSNorm → tied-embedding LM head), `LlamaConfig : LLamaConfigLike`, `LlamaLoader.Load<TModel,TWeight>`, the `StateDictLoader.LoadRMSNorm/LoadLinear` binding helpers, the `Gpt2BpeTokenizer` byte-level BPE reader (see the tokenizer-correction note above), and enabling `UseWidenSimd` for the BF16/Half narrow runs so generation is practical (see the BF16 section above).

## Library features used

| Capability | Where exercised |
|---|---|
| `RMSNorm<T>` affine gamma | Pre-norm in every decoder block + final norm |
| `Activation.Silu` (forward/VJP/JVP) | Gated SiLU FFN gate path |
| `RotaryEmbedding<T>` (RoPE, `rotate_half`) | Q/K rotary position embeddings |
| `LlamaCausalAttention<T>` + `GqaRepeatKV` | GQA self-attention (9 Q / 3 KV) |
| `LlamaDecoderBlock<T>` | Pre-norm attention + gated SiLU FFN + residuals |
| `LlamaForCausalLM<T>` + tied LM head | Embed → blocks → final norm → `hidden @ embed^T` |
| `Gpt2BpeTokenizer` | Sample-local GPT-2 byte-level BPE tokenization |
| `NivaraPrimitives.UseWidenSimd` | SIMD widen-compute-narrow BF16 matmul (native path) |
| Greedy generation (inference-default) | 32-token decode, no `GradientUtils.Grad()` scope |
| `smollm ab` A/B + `smollm benchmark` | Scalar-vs-widen comparison; median-of-3 generation timing |

## Verification

Greedy argmax agreement with the PyTorch reference (32-token generation):

| Metric | F32 vs Ref | BFloat16 vs Ref | Half (fp16) vs Ref |
|---|---|---|---|
| generated-token argmax match | 25/32 | **22/32** | 0/32 |
| final-position logits cosine | 0.24* | **0.94** | NaN |

BF16 is the strongest numeric match (0.94 cosine, mean abs diff 3.63) and is the natural native-on-disk choice. F32 comes close (25/32 tokens) but its *final*-position logits are compared against a divergent suffix once the greedy streams part ways (*), so the cosine is less meaningful. **Half is unusable for SmolLM**: a merely 10-bit mantissa cannot hold the accumulating attention geometry across 30 layers, so the final logits go NaN and the decode collapses to all-pad tokens (0/32). In short: pick BF16 (native) or F32 — never Half — for SmolLM generation.

See [docs/BFLOAT16.md](BFLOAT16.md) for engine-level BF16 details.

## Performance

| Precision | Weights | Nivara (AC) | vs F32 |
|---|---|---|---|
| F32 (widened) | 513.1 MB | 7404 ms | — |
| BF16 (native on disk) | 256.6 MB | 11499 ms | **~1.55× slower** |

BF16 halves the weight memory (256.6 MB vs 513.1 MB), but on CPU it is **not** faster — the F32 path runs fully-optimized native `float` SIMD kernels while BF16 still pays the widen/widen-back overhead, so F32 is ~1.55× *faster* for generation here. The ratio is strikingly stable across power states (1.53× on battery, 1.55× on AC), because it is a same-session Nivara-internal comparison and therefore never crosses frameworks. The takeaway: on CPU, use BF16 only when you need the halved memory footprint; if you have the ~513 MB headroom, F32 gives both faster generation and better numerical fidelity. (The BF16 native load also skips the F32→BF16 truncation the other models' narrow modes do — on disk SmolLM is already BF16.) Tracked as **#363** / **#391**.

**BF16 scalar-fallback vs widen SIMD A/B** (`smollm --precision bf16 ab`, re-measured 2026-09-27 on AC, same machine):

| Mode | ms/token prev | ms/token cur | Full gen prev | Full gen cur | vs (cur) |
|---|---|---|---|---|---|
| BF16 scalar fallback (`UseWidenSimd = off`) | 7,032 | 2,667 | 225,037 ms | 85,368 ms | — |
| BF16 widen (`UseWidenSimd = on`, default for narrow) | 705 | 374 | 22,591 ms | 11,972 ms | **7.13× faster** |
| F32 native (control) | 333 | 231 | 10,660–10,926 ms | 7,404 ms | widen transparent |

The `--simd-widen` flag toggles the widen path from the CLI; for narrow models it is enabled by default (without it, BF16 matmul falls back to the scalar dot). The F32 control confirms the toggle is a no-op for `float` (identical token streams, 32/32). Reading the table together with the memory-vs-performance table above: BF16 **widen** is 7.1× faster than BF16 **scalar** and ~1.6× slower than F32 native.

Two things changed on AC and are worth recording rather than smoothing over:

- **The scalar path gained the most (2.6×) and the widen path the least (1.9×)**, so the widen advantage narrowed from ~10× to **7.13×**. The absolute win is unchanged in character — the scalar fallback is still unusable — but "~10×" is no longer the right headline number. The earlier figure was inflated by a power state that penalised the scalar path's long serial dependency chain harder than the widen path's vectorizable work.
- **The F32-vs-BF16 ratio is ~1.6× here and ~1.55× in the memory table above**, so that conclusion held across power states. It is a Nivara-internal comparison and never crosses frameworks, which is why it is stable where the cross-framework rows were not.

See the [NivaraInference README](../samples/NivaraInference/README.md) for full benchmark tables.

## GPU path

Status: **Completed — probe answers recorded. No production code.**

### Problem

SmolLM-135M-instruct inference in `samples/NivaraInference` is CPU-only and its `smollm` mode still decodes **cache-free** (`model.Forward(sequence)` per token). We want GPU branches in the fused inference kernels to accelerate decode and prefill. Before choosing a backend (Level Zero SPIR-V vs DX12/ComputeSharp) and a precision strategy (F32 / native BF16 / widened BF16), we must **verify on the actual Arc 140T iGPU and current driver** what the Level Zero stack can really do.

`tests/Nivara.GpuProbe` already proves real iGPU compute and documents a wall: the current Intel driver's IGC (OpenCL/Level Zero frontend) crashes on `OpAccessChain` (IGCIT #1144 bug class) and mis-links private variables, which today rules out indexed tensor kernels (GEMV/GEMM/attention). Whether that still holds, whether BF16 is natively usable, and what the fallback options are — those are the questions this branch answers with the probe itself.

### Investigation questions (the probe-first deliverables)

#### Q1 — Current driver state: is indexed access still broken?
- Re-run the existing `run` bisection + `l0` enumeration as committed.
- Record: driverVersion, L0 API version, extension list, SPIR-V max version, fp16/fp32/fp64/DP4A module caps.
- Attempt a minimal GEMV-shaped kernel (weight row × input row → output) to (re)confirm the access-chain crash boundary — reported as expected driver diagnostic if it fires.
- **Expected outcome**: on this driver build, indexed access remains broken via Level Zero (OpenCL frontend); only the safe subset works.

**Answer (probe findings):** driver `0x010393E2`, L0 API 1.15, SPIR-V max 1.0 — indexed access remains **broken**. All 8 access-chain variants (OpAccessChain / InBounds / PtrAccessChain, u32/i32/u64 index, param/global base, Restrict/NoAlias) trigger `IGC: Internal Compiler Error: Access violation` at compile time. Same bug class as IGCIT #1144 (Blender AV on Arc B580). A minimal GEMV-shaped kernel (`c[i] = a[i] + b[i]`) crashes identically. **The safe working subset** (direct loads/stores, OpPhi loops, 1-lane atomics, BF16 conversion ops) compiles and runs. IGC bug #4 confirmed: LocalSize ≥ 8 atomics drop the upper SIMD half; localSize=1 (simd1) passes.

#### Q2 — Does BF16 work natively on this GPU (Level Zero)?
- Report the `ZE_extension_bfloat16_conversions` extension version + presence (via the existing extension enumeration) and any BF16-related module caps.
- Probe a **native-conversion module**: SPIR-V declaring the `Bfloat16ConversionINTEL` capability (and 16-bit float storage where legal) that converts a scalar BF16 → F32 — does IGC build it?
- Probe an **emulated-conversion kernel** (safe subset): BF16 held as `ushort`, converted with shift/&/exponent math (same bit-arithmetic as the CPU `WidenBf16ToF32` kernel), accumulated in F32 through an OpPhi loop, atomic aggregation via 1-lane workgroups, host-verified exact.
- **Expected outcome**: native conversion may or may not build (capability gate); emulated conversion proves BF16→F32 math is GPU-correct but needs indexed access (Q1 result) to become a real dot-product path.

**Answer (probe findings):** `ZE_extension_bfloat16_conversions` is present. `BFloat16` round-trip is **fully proven on this iGPU**:
- **Native path**: `OpConvertBF16ToFINTEL` (capability 6115) — IGC honors the contract; BF16 values pass through losslessly to `f32` registers; the widened `f32`'s upper 16 bits map back to the original `BFloat16` exactly.
- **Safe-subset emulation**: `OpUConvert` + `<<16` shift — same result, no extension required (future-proof if the extension disappears).
- **1M-iteration accumulation**: exact (1,000,000.0) — the BF16→F32→f32-accumulate pipeline works end-to-end on device.
- **Host edge is pure `BFloat16`**: no `ushort`, no `ToSingle`, no host bit arithmetic; the raw 16-bit `BFloat16` layout IS the on-device storage pattern. Zero widening on the input path; the `f32` accumulator is the hardware's native BF16 compute model (Xe2 DPAS also accumulates BF16 products in `f32` — `f32` accumulation is required for precision, not an added widening).

The BF16 compute path is ready. The blocker is the access-chain ICE (Q1).

#### Q3 — If BF16 is not natively usable, what are the options?
Produce a documented recommendation (in this file / `docs/BFLOAT16-GPU.md`):
- **(a) Driver update + re-probe** — IGC fix is upstream (IGCIT #1144 class); `run` bisects the whole surface in seconds, so re-validation is cheap.
- **(b) BF16-emulated on GPU** — store BF16 weights on device, widen in-register per dot (the CPU `UseWidenSimd` pattern); viable only once indexed access works (Q1) since it needs real GEMV kernels.
- **(c) F32 GPU path only** — weights widened to F32 at upload (2× memory, 513 MB for SmolLM); BF16 stays CPU-only. Simplest, works where kernels can run at all.
- **(d) DX12 / ComputeSharp** — Intel's **DX12 driver is a separate path** from the IGC OpenCL frontend; the probe's driver bugs don't apply there, and a managed backend would run real GEMM/attention kernels today. Cost: a NuGet dependency in a new project (core stays dependency-free). This is the likely landing spot if Q1 comes back "still broken" and a driver bump is not available/practical.

**Answer (probe findings):**
- **(a) Driver update + re-probe** — still recommended. The IGC access-chain ICE is the same class as IGCIT #1144 (Arc B580), fixed upstream. Re-run `run` after any driver bump (seconds of work).
- **(b) BF16 on GPU** — already proven (see Q2); the blocker is indexed access, not BF16 itself. Once Q1 clears, the native `OpConvertBF16ToFINTEL` path is ready as-is.
- **(c) F32-only GPU path** — still viable as fallback (2× weight memory, 513 MB for SmolLM); BF16 stays CPU-only until the access-chain fix lands.
- **(d) DX12 / ComputeSharp** — **the Arc 140T reaches D3D12 FL 12_2 + shader model 6.8** (`D3d12Check.cs`). A managed DX12 compute backend targets a separate shader compilation path (HLSL → DXIL / usc) that bypasses the IGC OpenCL frontend entirely. This is the **immediate viable fallback** for real GEMM/attention kernels on the iGPU while the Level Zero access-chain fix is pending. Cost: a NuGet dependency in a new project; core stays dependency-free. **Recommendation: pursue (d) DX12/ComputeSharp in parallel with (a) driver re-probe on next driver update.**

### Scope / non-goals for this branch

- **No production GPU code**: this branch extends `tests/Nivara.GpuProbe` and updates docs only. The production plan (`-gpu` flag, `src/Nivara.Gpu`, `LlamaFusedKernels` GPU branches, SmolLM KV cache in NivaraInference) is recorded below for the follow-up branch, gated on the probe answers.
- No changes to `src/Nivara`, `samples/NivaraInference`, or `NivaraChat`.

### Probe extensions (planned commits)

1. ✓ `8c0c46d` — `docs: plan SmolLM GPU probe-first investigation` (this file)
2. ✓ `a70fc79` — baseline re-run recorded (driver `0x010393E2`, L0 API 1.15, SPIR-V max 1.0); `ZE_extension_bfloat16_conversions` + BF16-relevant module caps summarized in `L0Probe` output; the minimal GEMV-shaped access check is covered by the bisection's `straight` variant (`c[0] = a[0] + b[0]`) and still triggers the access-chain ICE
3. ✓ `a70fc79` + `027c80b` — `bf16_native` / `bf16_emul` / `bf16_native_acc` kernels built, launched, host-verified exact; host edge refined to a pure `BFloat16` round trip (no `ushort`/`ToSingle`)
4. ✓ `0393130` + this file — `dx12` availability probe committed; Q1/Q2/Q3 answer blocks + BF16/DX12 recommendation recorded in `docs/SMOLLM.md` (`docs/BFLOAT16-GPU.md` not needed separately — the answers live here)

### Verification steps

- `dotnet run -c Release --project tests/Nivara.GpuProbe -- run` passes its real gates (add_parallel, add_loop, bf16_native, bf16_emul, bf16_native_acc) — exit 0; the expected IGC-bug diagnostics (access-chain ICE, 15) are reported separately and do not fail the run.
- New BF16 kernels print host-verified results (exact f32 where the subset allows) and are clearly reported as PASS.
- `dx12` mode passes on this machine: Arc 140T FL 12_2 / shader model 6.8.
- All probe modes (`list`/`run`/`spv`/`ocl`/`dx12`) keep working.

### Blast radius

- `tests/Nivara.GpuProbe/LevelZero/SpvKernels.cs`, `L0Run.cs`, `L0Probe.cs`, `README.md`, `tests/Nivara.GpuProbe/Program.cs`, `tests/Nivara.GpuProbe/D3d12Check.cs` — probe-only additions.
- `docs/SMOLLM.md`, `docs/BFLOAT16-GPU.md` — documentation.
- No core library, no samples, no Nivara.Tests, no package changes.

### Follow-up (next branch, gated on probe answers)

- SmolLM KV-cached decode in `NivaraInference` (mirror Qwen), then `-gpu`: new `src/Nivara.Gpu` (DX12/ComputeSharp or updated-L0 backend per Q1–Q3), GPU branches at `LlamaFusedKernels` / `LlamaDecoderBlock` hooks, `GpuMatMul`, `GpuDecodeAttention`, `GpuRmsNorm`, `GpuSoftmax`, `GpuRoPE`, `GpuSilu`, `smollm -gpu [benchmark]`, parity + PyTorch-fixture validation.

### GitHub issues log

- (empty — no issues created yet)
