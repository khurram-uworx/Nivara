# Acceleration — Nivara GPU (Silk backend, draft)

Draft for review (2026-10-18). This does not replace any existing reference. The acceleration story is
Silk-only: ILGPU is completed, gated work that is no longer being extended, and its details remain in
[ILGPU.md](ILGPU.md) as history. Code-level detail for the route we ship lives in [SILK.md](SILK.md). This page
is the story and the index.

For model-specific architecture and usage, see:

- [DISTELBERT.md](DISTELBERT.md) — DistilBERT, MiniLM, SST-2
- [MODERNBERT.md](MODERNBERT.md) — ModernBERT-large
- [SMOLLM.md](SMOLLM.md) — SmolLM-135M
- [QWEN.md](QWEN.md) — Qwen2.5-0.5B-Instruct
- [LAYA.md](LAYA.md) — Laya decision head
- [VISION.md](VISION.md) — MobileNetV2, ResNet-18

Where the detail lives: the story and measurements are here. Kernels, geometry, design decisions, and durable
lessons are in [SILK.md Part 2](SILK.md). Probe findings are [SILK.md Part 1](SILK.md). Earlier backend
verdicts stay in [ILGPU.md](ILGPU.md) as history. The shipped streaming contract is
[STREAMING.md](STREAMING.md); the backend case studies are in
[`tests/Nivara.GpuProbe/`](../tests/Nivara.GpuProbe/README.md).

Scope today: the measured paths in this index, MiniLM, DistilBERT, DistilBERT+SST-2, ModernBERT, and Laya, are F32
only, with `default`, `benchmark`, and `compare` modes. The same Silk backend also runs SmolLM, Qwen,
MobileNetV2, and ResNet-18, with the Qwen and vision modes described in SILK.md. Silk is distributed as
Silk.NET.OpenCL 2.23.0 and compiles OpenCL C in-process on the Intel OpenCL ICD.

## Why this route

We settled on one backend: raw OpenCL C, written by hand and compiled by the driver through Silk.NET. It is the fastest
hand-authored leg on the kernels that matter, it gives a vendor-neutral path that does not depend on a high-level
runtime abstraction, and it shares one host model with the work that preceded it (`Transposed`, `QkvPacked`, `Req`), so
the maths never forked. It is also deliberately F32-only: neither lower precision pays off at the shapes we measured.
The honesty of the bar is that "fast" is shape-specific, and the numbers below are read accordingly.

## 1. Acceleration results

All results are F32 on an Intel Arc 140T iGPU unless noted. The BERT table uses the Nivara GPU runs at 128 tokens
(3 warmup + 10 timed, median of 3 runs); the CPU columns are the baselines already documented for these models and are
used as reference. The one question these numbers answer is whether the GPU path clears the PyTorch CPU baseline at the
shape we intend to ship; the answer depends on the shape, and the rest of this section is the evidence for it.

### 1.1 BERT family, 128 tokens

| scenario | PyTorch CPU | Nivara CPU | Nivara CPU vs PyTorch | Nivara GPU | Nivara GPU vs PyTorch |
|---|---|---|---|---|---|
| distilbert | 35 ms | 194.7 ms | ~5.6× slower | **52.3 ms** | ~1.5× slower |
| distilbert_sst | 35 ms | 166.4 ms | ~4.8× slower | **52.1 ms** | ~1.5× slower |
| minilm | 11 ms | 76.3 ms | ~6.9× slower | **20.9 ms** | ~1.9× slower |

A 128-token forward on a 128-EU iGPU is latency- and bandwidth-bound, so the GPU remains behind PyTorch's MKL-backed
CPU path here. That deficit is shape-specific rather than a property of the device: at Laya's larger shape the same
accelerator clears PyTorch. Measure at the shape you intend to ship (details in [LAYA.md](LAYA.md)).

### 1.2 ModernBERT-large, seq 128

| path | time |
|---|---|
| PyTorch CPU | 287.3 ms |
| Nivara CPU | 1249.8 ms |
| Nivara GPU | 343.8 ms |

Nivara GPU is about 3.6× faster than Nivara CPU and about 1.2× slower than PyTorch CPU at seq 128. Longer sequences
favour Nivara GPU over Nivara CPU by a wider margin, and throughput is about 125 tok/s at seq 128.

### 1.3 Laya decision head, `max_len` 512

| path | time |
|---|---|
| PyTorch CPU | ~2.97 s |
| Nivara CPU | ~8.48 s |
| Nivara GPU | 1.64–1.84 s per question |

Nivara GPU runs all four questions (choice2, choice13, score4, noul) in 1.64–1.84 s: about 4.6–5.2× faster than Nivara
CPU and about 1.6–1.8× faster than PyTorch CPU at this shape. This is the shape where the accelerator stops tying
PyTorch and starts beating it.

### 1.4 Microkernels

The three probe kernels, measured with the same discipline as the other legs (1 warmup + best-of-25, synchronized
dispatch):

| kernel | CPU (production Nivara) | Nivara GPU | gate |
|---|---|---|---|
| dot16 | 1.5–1.7 µs | **6.4–6.7 µs** | PASS — 0.0 ULP |
| silu | 31.2–41.6 µs | **6.1–7.1 µs** | PASS — 576/576, worst 4.0 ULP |
| gemv | 2785–4158 µs | 90.1–100.1 µs | PASS — 1536/1536 |

Cross-host medians on the Arc 140T are dot16 7.0 µs, silu 6.7 µs, gemv 100.6 µs; on Iris Xe they are 71.4 / 69.2 /
486.2 µs. The Nivara GPU backend is the fastest hand-authored leg: the compiled binary dispatches directly. The
4.4–5.2 ms in-process build time is reported separately and is not comparable across runs.

### 1.5 Precision on this iGPU

| | BF16 | FP16 (`half`) |
|---|---|---|
| Extension | `cl_intel_bfloat16_conversions` | `cl_khr_fp16` |
| Convert | yes | yes |
| Arithmetic | no | yes |
| Faster at these shapes | not reachable | no — tie, 0.97–1.07× |

BF16 has conversion only; FP16 has native arithmetic but no speed win at the measured shapes. That is why the production
backend stays F32. Detail: [SILK.md §5](SILK.md).

## 2. Correctness

Speed only counts if it agrees. Every supported model is checked against the same-process CPU encoder/head at
`|gpu−cpu| ≤ 1e-3·(1+|cpu|)`:

- `minilm`, `distilbert`, `distilbert_sst`: GPU versus same-process CPU, with pooled-embedding cosine for MiniLM and
  the PyTorch 3-way (argmax 8/8) for SST-2.
- `modernbert`: maxAbs 2.861E-005, maxRel 5.577E-004, cosine 1.0000001, 0 non-finite on either side.
- `laya`: worst marker-logit residual 1.110E-005, 1.1% of the bound; act matching to 0.0; mask clamp PASS on both.

Kernel-level checks and CLI rejects are covered in §§5–6.

## 3. Honest caveats

- Do not benchmark on battery. Earlier MiniLM readings drifted from 25.6 to 33 ms as charge drained.
- ModernBERT's 4096-row banded number is the weakest single row; its min and median differ by 11.8% and the
  re-measurement has not been run.
- Timing gates are not correctness gates. A ratio outside x0.8..x1.2 is reported, not a failure.
- Rejected benchmark combinations fail before any model-file lookup.
- The PyTorch ratios in §1 are shape-specific: a 128-EU iGPU is not expected to beat PyTorch's MKL-backed path at 128
  tokens, although it does at Laya's shape.

## 4. Where the Nivara GPU implementation lives

| file | role |
|---|---|
| `src/Nivara.Extensions/Silk/SilkKernels.cs` | OpenCL C translation unit, geometry, shared accumulation macro; `internal` |
| `src/Nivara.Extensions/Silk/SilkRuntime.cs` | API acquisition, Intel/AMD-preferred GPU selection, in-process build with decoded build log, buffers and sub-buffers, per-argument status checks, blocking data movement, reverse-order disposal |
| `src/Nivara.Extensions/Silk/SilkLauncher.cs` | typed dispatch shared by all runners |
| `src/Nivara.Extensions/Silk/WeightLayout.cs` | `Transposed`, `QkvPacked`, `Req`; the shared host math |
| `src/Nivara.Extensions/Silk/SilkBertEncoderRunner.cs` | shared MiniLM/DistilBERT runner |
| `src/Nivara.Extensions/Silk/SilkModernBertGpuRunner.cs` | bias-free GEMM, row-interleaved QKV split, CPU-uploaded RoPE, per-layer band |
| `src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner.cs` | biased norms, fused biased QKV, host-side act-head features |
| `src/Nivara.Extensions/Silk/SilkModernBertSession.cs` / `SilkLayaSession.cs` | session implementations |
| `src/Nivara.Extensions/AI/GpuSeam.cs` | `IBertEncoderRunner`, session interfaces, open delegates |

`Nivara.Extensions.csproj` pins Silk.NET.OpenCL 2.23.0. `SilkKernels` and `SilkLauncher` are `internal`, exposed to
`Nivara.Tests` and `Nivara.PerformanceTests`.

## 5. Kernels, geometry, and contract

`KernelNames` (`SilkKernels.cs:126`) is defined in build/creation order:

```
EmbedSum, GatherRows, LayerNormResidual, LayerNorm, Add, AddBias, SplitColumns, Rotary, GeGlu,
Gemm, GemmBias, GemmRelu, GemmGelu, GemmQkv, Attention,
RmsNorm, RmsNormLmHead, RmsNormGemv, Silu, SwiGlu, CausalGqaAttention, DecodeGqaAttention,
ArgMax, GemmAdd, GemmGemvAdd, MlpBlock, DecodeGqaAttentionOprojFused,
Conv2d, BatchNormEval, Relu, Relu6, MaxPool2d, AdaptiveAvgPool2d
```

Four families: the original BERT encoders, the ModernBERT/Laya ports, the Qwen/SmolLM decode kernels, and the vision
kernels. Full signatures and semantics: SILK.md §11.1.

Geometry: `GemmTileSize` 16, `GemmBlockCols` 4, `GemmTileCols` 64, `GemmRowStride` 80, `LinearGroupSize` 256,
`AttentionGroupSize` 64, `MaxHeadDim` 64, plus the flat decode group sizes. `headDim` above 64 requires raising
`MaxHeadDim` (compile-time); construction rejects it otherwise.

One design decision carries the whole result: `#pragma OPENCL FP_CONTRACT OFF`. The kernel emits `acc += a * b`; the
OpenCL driver may otherwise fuse the multiply-add and change rounding. Transcendentals (`exp`, softmax, GELU's `erf`)
are checked on the numeric bound rather than bit-for-bit. `GEMM_ROW4_ACCUMULATE` is shared across the whole Row4 family so
its variants cannot drift, and the Row4 spans the fused epilogues (bias, ReLU, GELU, QKV) without a second accumulation
path.

## 6. Gate and CLI

Run the kernels gate:

```
dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels
```

The reference implementation's kernels and the Nivara GPU kernels run in one process. Byte-identical is required for:

```
gemm_row4, gemm_row4_bias, gemm_row4_qkv, gemm_row4_relu, gather_rows, add, add_bias, split_columns, rotary
```

Numeric-bound cells (`gemm_row4_gelu`, `geglu`, `layer_norm`, `batched_attention`, elementwise) use
`|nivara-gpu − reference| ≤ 1e-3·(1 + |reference|)`. One attention cell uses `band = 64`; the BERT cells use the global
band. Shapes: MiniLM (seq 128, hidden 384, 12 heads, headDim 32) and DistilBERT (seq 128, hidden 768, 12 heads, headDim
64; headDim drives the compiled local-memory bound). Timing is best-of-N after warmups and is reported, not a
correctness gate. An unavailable device or program exits `UNBUILT` (10).

CLI allow-list (`SilkModes`): `minilm`, `distilbert`, `distilbert_sst`, `modernbert`, `laya` with empty, `benchmark`,
or `compare`; Qwen with `""`, `benchmark`, or `tools`; `compare_diag` remains rejected. Checks run in this order:
mutual exclusion, model scope before model-file lookup, F32-only, and no `--gpu` alias.

## 7. Open work

- Publish benchmark tables and gate snapshots for the SmolLM, Qwen, MobileNetV2, and ResNet-18 Silk runners in this index (#628-#630).
- Cross-backend GEMM regression harness: #435.
- ModernBERT's 4096-row re-measurement: #447.
- LayerNorm one-work-item-per-row: #467.
- Vision convolution kernels and the Nivara-CPU deficit: #457, #458.
- Driver-bump hygiene: rerun the probe gate and all model compares after an Intel driver update.
- bf16/fp16 on the GPU: a promotion-phase item; raw OpenCL has BF16 conversion but no BF16 multiply.
- Streaming budget enforcement and Phase 2 spill: #325.
- Promotion of the Nivara GPU path into `src/Nivara.Gpu`: pending, and nothing in core changes until that decision.

Earlier CPU/GPU divergence notes (cost model, dispatch counts, throughput) are retained in
[ILGPU.md](ILGPU.md) as history.

## 8. Where to read next

| topic | document |
|---|---|
| Silk.NET/OpenCL probe, setup, precision deep-dive, eight-way comparison | [SILK.md Part 1](SILK.md) |
| production `--silk` runtime, kernels, runners, adding a model | [SILK.md Part 2](SILK.md) |
| ILGPU implementation, retained as historical reference | [ILGPU.md](ILGPU.md) |
| model pages for architecture, usage, and gates | [DISTELBERT.md](DISTELBERT.md), [MODERNBERT.md](MODERNBERT.md), [LAYA.md](LAYA.md), [SMOLLM.md](SMOLLM.md), [QWEN.md](QWEN.md), [VISION.md](VISION.md) |
| shipped streaming contract | [STREAMING.md](STREAMING.md) |
| backend-probe readers guide | [tests/Nivara.GpuProbe/README.md](../tests/Nivara.GpuProbe/README.md) |

## References

- [SILK.md](SILK.md) — probe verdicts and production `--silk` backend.
- [ILGPU.md](ILGPU.md) — historical implementation and GEMM reference.
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) — usage, quick starts, benchmark
  tables, CPU baselines.
- Issues: **#435** (GEMM regression harness) · **#447** (banded attention) · **#457** (GPU vision) ·
  **#458** (CPU deficit) · **#467** (LayerNorm) · **#606** (OpenVINO bf16 IR silu) · **#628** (SmolLM `--silk`) ·
  **#629** (Qwen `--silk`) · **#630** (vision `--silk`) · **#325** (streaming budget).