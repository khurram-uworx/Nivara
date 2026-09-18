# DistilBERT on GPU — assessment before the first end-to-end sample scenario

Status: **Assessment only — no code.** Decision record for the first GPU
end-to-end scenario in `samples/NivaraInference`: `distilbert --gpu` (F32 via
ILGPU). Follows the probe verdicts in [docs/ILGPU.md](ILGPU.md) and the
SmolLM GPU investigation ([docs/SMOLLM-GPU.md](SMOLLM-GPU.md)); supersedes the
SmolLM-first plan for the *first* scenario (see "Why DistilBERT first").

## TL;DR

- **ILGPU is the right long-term backend bet** for Nivara: pure-managed C#→OpenCL
  JIT, `dotnet restore` is the whole install, kernels are the same language as
  the host, 3/3 correctness gates PASS on this exact machine (Intel Core Ultra 7
  255H / Arc 140T), and it is the fastest GPU leg measured on elementwise
  (`silu` 6.9 µs, ~5–14× CPU) with a gemv within 1.5× of OpenVINO's *tuned*
  gemm using a deliberately naive shape.
- **DistilBERT is the right first scenario**: single encoder forward pass, no KV
  cache, no greedy decode loop, F32 weights already on disk (255.5 MB), a
  PyTorch CPU baseline (35 ms) and Nivara CPU baseline (185 ms) already
  recorded, and an existing PyTorch compare fixture (`last_hidden_state_py.bin`)
  to gate against. SmolLM is a farther target: it needs cache-free→KV-cached
  decode work first and its generation loop compounds per-token error.
- **The probe's gemv win does not transfer to DistilBERT's GEMM shapes as-is.**
  DistilBERT is dominated by batch-128 GEMMs (`[128,768]·[768,3072]`-class). A
  naive one-thread-per-row GEMM at the probe's measured 6.6 GMAC/s would run
  **slower than CPU (~825 ms)**. A tiled/shared-memory GEMM is the open lever and
  the first kernel to build + measure; the estimate below (5–15 ms matmul
  portion) is a target, not yet a measurement.
- **Scope of the sample phase**: add `ILGPU 1.5.3` reference in
  `samples/Nivara.Samples` *only*. No `src/Nivara`, no `Nivara.Extensions`, no
  `src/Nivara.Gpu` project. F32-only (`--precision` ignored/rejected with a
  clear message until narrow GPU is a later decision). Final library design
  postponed to after the first measured E2E numbers.

---

## 1. Why DistilBERT first (not SmolLM)

| dimension | DistilBERT (encoder) | SmolLM (causal LM) |
|---|---|---|
| forward shape | one `[128,768]` pass through 6 layers + head | greedy loop; currently **cache-free** re-forward of the whole growing sequence per token (`model.Forward(sequence)` per token) |
| KV cache | none (bidirectional) | needs cache-free → KV-cached decode first (the Qwen work exists but is not in the SmolLM path) |
| precision story | F32 on disk; bf16/fp16 are load-time truncations, all 8/8 argmax | BF16-native on disk; native BF16 GPU needs ILGPU's **unmerged** packed-widen path (no native BF16 kernels in 1.5.3) |
| E2E numbers available | **yes** — PyTorch 35 ms vs Nivara 185 ms (128 tok), SST-2 184 ms, compare fixture + logit diffs | yes, but decode-loop semantics muddy a kernel comparison |
| correctness gate | hidden states vs `last_hidden_state_py.bin` (max abs diff 5e-6, cosine 0.99999988 today); SST-2 argmax 8/8 | 32-token greedy argmax agreement (25/32 F32, "numeric precision diff" class) |

The user-facing goal of the first scenario is a *clean* GPU-vs-CPU number and a
verifiable, impactful demo (SST-2 sentiment REPL at interactive latency). That is
DistilBERT. SmolLM stays the follow-up generative target once a KV cache exists.

## 2. What we already know (the numbers we have)

### 2.1 CPU baselines — same machine class as the probe

From `samples/NivaraInference/README.md` (Intel Core Ultra 7 255H, .NET 11,
Release, recorded 2026-09-01; probe host is the same Arrow Lake-H iGPU class):

| model / input | PyTorch (CPU) | Nivara (CPU) | slowdown |
|---|---|---|---|
| MiniLM-L6, 128 tok | 11 ms | 64 ms | ~6× |
| **DistilBERT, 128 tok** | **35 ms** | **185 ms** | **~5×** |
| **DistilBERT SST-2, 128 tok** | 35 ms (same arch) | **184 ms** | ~5× |

Precision ground truth (vs F32 HF reference, CPU): SST-2 argmax **8/8** for F32 /
BFloat16 / Half; max abs logit diff ~1e-6 / ~0.33 / ~0.22. Encoder hidden states
match HF to max abs diff 5e-6. These are the fidelities the GPU path must preserve
(with f32-first, the expectation is CPU-parity numbers, i.e. ~1e-6-class diffs).

### 2.2 ILGPU probe micro-kernels — same machine (Arc 140T, ILGPU 1.5.3)

From [docs/ILGPU.md](ILGPU.md) / `tests/Nivara.GpuProbe` (steady state = 1
warmup + best-of-25, persistent buffers; production Nivara kernels as gold):

| kernel | shape | ILGPU | CPU Nivara | margin |
|---|---|---|---|---|
| `dot16` | K=16 | 11.4 µs | 1.2–4.8 µs | launch-bound — **CPU wins** (floor for any GPU dispatch) |
| `silu` | 576 | **6.9 µs** | 29–99 µs | ~5–14× (fastest GPU leg) |
| `gemv` | 1536×576 | **133.4 µs** | 2291–4745 µs | ~17–35× (one thread per row, naive) |

Setup, one-time: accelerator creation + first sync ~9 ms; per-kernel JIT
~1.3–3.8 ms. Correctness: all three kernels gate PASS against production Nivara
within `|gpu − cpu| ≤ 1e-6 + 1e-5·|cpu|` (dot16 0.0 ULP, worst 4.0 ULP).

The gemv figure is the only ILGPU matmul measurement that exists. Rate implied:
1536×576 dots = 884 736 MAC in 133.4 µs ≈ **6.6 GMAC/s** (naive shape; compare
OpenVINO's *tuned* gemm bf16 at 86.8 µs on the same shape ≈ 5.9 T MAC/s — two to
three orders of magnitude apart for the same hardware).

## 3. DistilBERT F32 work inventory (what the GPU must run)

Config: `dim=768`, `n_layers=6`, `n_heads=12` (head 64), FFN 3072, seqLen 128,
batch 1. Per layer, the matmul rows (base `distilbert` = `BertEncoder`; SST-2
adds a `[1,768]`-class head):

| kernel | shape | MACs | share of layer matmuls |
|---|---|---|---|
| `q_lin` / `k_lin` / `v_lin` / `out_lin` | `[128,768]·[768,768]` ×4 | 4 × 75.5M = 302M | 33% |
| attention core (QKᵀ, softmax, ·V) | 12 heads × `[128,64]·[64,128]` + `[128,128]·[128,64]` | ~25M | ~3% |
| `ffn.lin1` | `[128,768]·[768,3072]` | 302M | 33% |
| `ffn.lin2` | `[128,3072]·[3072,768]` | 302M | 33% |
| LayerNorm ×2, GELU(erf), bias adds | elementwise/row-reduce over 768/3072 | — | — |

6 layers ≈ **5.44 GMAC matmul + ~0.15 GMAC attention** ≈ **~5.6 GMAC**, plus
embeddings (two gathers, no matmul — `[30522,768]`/`[512,768]` tables) and the
small head. CPU Nivara runs this in ~185 ms → effective ~30 GMAC/s (≈60 GFLOPS),
so **the CPU is already running the GEMMs near the naive-GPU rate** — that is the
whole point of tiling.

Weight footprint: 66.9M params F32 = **255.5 MB**. iGPU shares system DRAM with
the CPU, so device upload is a memcpy-class one-time cost (~30–50 ms at
5–10 GB/s), not a PCIe bottleneck. The iGPU also shares bandwidth/l3 with the
CPU — the E2E win comes from *compute* (128 EU), not bandwidth.

GPU kernel set needed (all expressible in ILGPU C#):
1. **Tiled GEMM** (the keystone; see §4) — one kernel parameterized by
   `aRows/aCols/bCols` covers q/k/v/o, lin1, lin2, and the head.
2. **Attention core** — per-head QKᵀ (or the 12-head batched form), fused
   scale+mask+row-softmax (the CPU `AttentionKernels<T>` pattern), ·V, O-scatter.
   Needs `XMath.Exp` (ILGPU.Algorithms, already proven).
3. **Row reductions / LayerNorm** (mean+var over 768 per row) and **GELU(erf)**
   elementwise (erf needs a poly/rational approximation or `XMath` — ILGPU core
   has no erf; `ILGPU.Algorithms` provides distributions/`XMath`; erf can be
   built from `XMath.Tanh`-class primitives or a table poly — verify availability
   in 1.5.3 during implementation; fallback: load `System.Math`-free managed erf
   poly, same precision class as CPU `GeluExact`).
4. **Bias adds + residual adds** — elementwise, trivial.
5. **Embedding gather** — index `[30522,768]`/`[512,768]` tables; either a gather
   kernel or host-side copy (tables are 118 MB; upload once, gather on device).

Layout: Nivara weights are stored row-major `[out, in]` and the CPU path is
transpose-free (`MatMulTransposedB` reads `B[c, k]`). On the GPU the simplest
correct first cut is a **one-time host-side transpose at upload** so the GEMM
consumes plain row-major operands; a transpose-free in-kernel variant is a
follow-up optimization. First-cut correctness >> first-cut layout cleverness.

## 4. Expected impact — honest estimate, not measurement

### 4.1 The naive shape loses

At the probe's measured naive rate (6.6 GMAC/s), DistilBERT's 5.44 GMAC would
take **~825 ms** of pure matmul — ~4.5× *slower* than CPU. The probe gemv shape
(one thread per 576-length row dot) is memory-latency-bound, not compute-bound,
and does not generalize to batch-128 GEMMs. **A tiled GEMM is a requirement,
not an optimization.**

### 4.2 A realistic tiled-GEMM range (target, to be measured)

Arc 140T = 128 EU. F32 FMA-class theoretical peak on this iGPU is ~4 T MAC/s
(order-of-magnitude; Xe2 DPAS gives BF16 ~2× that for the OpenVINO bf16 gemm
row). A competent ILGPU tiled GEMM (work-group tiling, local-memory staging,
vectorized FMA loop) typically lands at 15–40% of peak → **0.6–1.6 T MAC/s** →

| scenario | matmul time (5.44 GMAC) | E2E (with attention/LN/GELU/readback) |
|---|---|---|
| CPU Nivara today | ~170–180 ms | **185 ms** |
| naive GPU (measured rate, 6.6 GMAC/s) | ~825 ms | ~900 ms — **loses** |
| tiled GPU, conservative 0.6 T MAC/s | ~9 ms | ~15–25 ms |
| tiled GPU, good 1.6 T MAC/s | ~3.4 ms | ~8–15 ms |

vs **PyTorch CPU 35 ms**: even the conservative tiled case is a clear win; the
SST-2 REPL (8 sentences, 184 ms each → 1.5 s today) would drop to ~100–200 ms
for all eight — interactive. **Decision gate**: build the tiled GEMM first,
measure it on the Arc 140T before writing the rest of the kernel set. If it
lands below ~0.3 T MAC/s, step back and re-evaluate against the proven
OpenVINO path (closed tuned runtime) before committing the full model kernel
set.

Fixed-cost budget (fine at any plausible tiled-GEMM number): ~40–50 dispatches
per forward × ~11 µs launch floor ≈ 0.5–1 ms; upload 255.5 MB ≈ 30–50 ms once;
per-kernel JIT 1.3–3.8 ms once (~10–12 kernels ≈ 20–40 ms once). Steady-state
per-forward GPU cost is dominated by the GEMMs, exactly as on CPU.

## 5. Why ILGPU long term (the strategic read)

From the probe's side-by-side, ILGPU is the strongest *managed* backend:

- **Delivery**: NuGet `ILGPU 1.5.3` + `ILGPU.Algorithms 1.5.3` (NCSA, pure C#);
  in-box Windows `OpenCL.dll` ICD + Intel driver — no SDK, no toolchain, no
  native install. This is the only backend where adding a package reference
  *is* the whole story.
- **Correctness**: 3/3 gates PASS on the real iGPU with no CPU fallback — the
  IGC question is answered (compiler-produced OpenCL C runs correctly; only
  hand-authored SPIR-V breaks).
- **Kernel ergonomics**: kernels are the same language as the rest of Nivara
  (C# static methods, `Index1D`, high-level launchers, no boxing) — the natural
  long-term substrate for a managed `src/Nivara.Gpu` without introducing a shader
  language into the codebase.
- **Measured strengths**: fastest GPU leg on silu (elementwise/activation class);
  gemv within 1.5× of OpenVINO's *tuned* gemm with a deliberately naive shape —
  i.e. the tiling lever is still mostly unused.
- **Honest gaps / watch items**: no native BF16 in 1.5.3 (packed-widen path is
  proven but non-trivial); OpenCL ICD must exist at runtime; two moving
  compilers (ILGPU JIT + IGC frontend) — re-run `kernels` after every driver
  bump; gemv *tuned* still trails OpenVINO today.

ComputeSharp (DXIL/D3D12) is the complementary managed option — same ergonomics
story, different driver stack — and remains the fallback/alternative if the
OpenCL ICD story ever regresses. The sample phase can be ILGPU-only; it costs
nothing to keep that door open.

## 6. First end-to-end scenario spec (the sample-phase deliverable)

Goal: a real, measurable, gated GPU forward of DistilBERT on the Arc 140T with
the least possible new machinery, using only the numbers already recorded.

1. **`samples/Nivara.Samples` gains the ILGPU references** (the only package
   change this phase): `ILGPU 1.5.3`, `ILGPU.Algorithms 1.5.3`. No `src/Nivara`
   or `Nivara.Extensions` edits; no new project.
2. **`distilbert --gpu`** (and **`distilbert_sst --gpu`**) in `NivaraInference`:
   - **F32-only**; `--precision bf16|fp16` with `--gpu` is rejected with a clear
     "GPU is f32-only in this phase" message (no precision confusion; bf16 GPU is
     a later decision because ILGPU 1.5.3 has no native BF16).
   - Load the existing `float[]` tensors (no load-path change), transpose weights
     once at upload into persistent ILGPU buffers (iGPU shared memory ⇒ upload is
     memcpy-class).
   - Run the §3 kernel set: tiled GEMM, attention core, LayerNorm, GELU(erf),
     bias/residual adds, embedding gather.
   - **Readback + gate**: hidden states vs the CPU path and the PyTorch fixture
     (`last_hidden_state_py.bin`, today's max abs diff 5e-6) with the probe's
     gate contract `|gpu − cpu| ≤ 1e-6 + 1e-5·|cpu|`; SST-2 prints the 8-sentence
     argmax table (CPU parity 8/8 expected).
   - `benchmark` mode: `--gpu` reports per-forward GPU median (existing
     `benchmark` plumbing) → the headline **GPU vs CPU (185 ms) vs PyTorch
     (35 ms)** number.
3. **Keep the sample-scoped convention**: all GPU orchestration lives in
   `samples/Nivara.Samples` (like `Gpt2BpeTokenizer`, `LlamaLoader`,
   `DistilBertLoader`) gated by a `UseGpu` flag / `GpuRunner<T>`-style sample
   helper — nothing public in core, no `src/Nivara.Gpu` yet. If the numbers
   justify it, the *next* phase decides the promotion (which backend, which
   project, bf16, which models).

## 7. Scope / non-goals for the sample phase

- **No** `src/Nivara` changes (no GPU branches in `GradKernels` /
  `LlamaFusedKernels` / `AttentionKernels`), **no** `Nivara.Extensions` changes,
  **no** `src/Nivara.Gpu` project, **no** `-gpu` in core APIs.
- No bf16/fp16 GPU; no SmolLM/Qwen GPU; no KV-cache/decoding GPU; no training on
  GPU.
- The probe (`tests/Nivara.GpuProbe`) is untouched except optionally a
  tiled-GEMM reference kernel if a probe-first micro-benchmark is preferred to
  going straight to the sample (see §4.2 decision gate — either way the tiled
  GEMM is measured before the model kernel set is written).

## 8. Risks & decision gates

| # | risk | mitigation / gate |
|---|---|---|
| 1 | **Tiled GEMM underperforms** (below ~0.3 T MAC/s on Arc 140T f32) | measure first; fallback = OpenVINO proven path or revisit f32 tiling strategy before writing the full kernel set |
| 2 | `XMath` gap: no erf in ILGPU core/Algorithms (GELU) | implement managed erf poly (CPU `GeluExact`-class) in the sample kernel — verify availability during kernel authoring |
| 3 | Transpose-free layout hurts tiled GEMM coalescing | host-side one-time transpose at upload (255 MB class, ~30–50 ms) |
| 4 | OpenCL ICD / driver variance | device-select assert (`CL_DEVICE_TYPE_GPU`, no CPU fallback); re-run probe `kernels` gate after driver bumps (seconds) |
| 5 | Precision drift vs CPU f32 path | per-tensor gate against CPU hidden states at ~1e-6 class; SST-2 argmax 8/8 |
| 6 | bf16 temptation mid-phase | deferred — f32-only, explicit rejection message (keeps the first E2E diff clean) |

## 9. References

- [ILGPU case study](ILGPU.md) (§5 setup/steady split, §6 PRO/CON) — the probe
  verdict and all ILGPU gotchas (`Index1D`, padded-grid bounds checks,
  `CL_DEVICE_TYPE_GPU`, `AsContiguous().GetAsArray()`, `MathMode.Fast` off)
- [ComputeSharp case study](COMPUTESHARP.md) — the alternative managed-D3D12
  backend, kept in reserve
- [SmolLM GPU investigation](SMOLLM-GPU.md) — why SmolLM is the follow-up, not
  the first scenario
- [`tests/Nivara.GpuProbe/README.md`](../tests/Nivara.GpuProbe/README.md) — the
  seven-way backend comparison table (gemv 133.4 µs ILGPU vs 137.9 µs OpenVINO
  tuned) and the `kernels` correctness gate
- `samples/NivaraInference/README.md` — CPU baselines, precision results, compare
  fixtures
- `samples/Nivara.Samples/BertModel.cs`, `DistilBertModel.cs` — the exact
  DistilBERT kernel shapes in §3