# BERT-family encoders on GPU — first ILGPU model: implementation reflection

Status: **Implemented, gated, and measured** (2026-09-19; DistilBERT via PR
#436, MiniLM via `khurram/minilm-gpu`). This is a
**reflection/documentation** of what we built and what we learned while adding
the first end-to-end GPU model support through ILGPU — it is *not* a usage guide
(that lives in [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md))
and *not* the forward-looking roadmap (that lives in
[ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md)). Historical assessment content
before implementation was rewritten away; only the durable facts and lessons
remain.

Related: probe verdicts in [docs/ILGPU.md](ILGPU.md), the SmolLM GPU
investigation ([docs/SMOLLM-GPU.md](SMOLLM-GPU.md)).

## 1. Result — gated and measured

First end-to-end GPU sample scenarios: `distilbert --gpu` / `distilbert_sst --gpu`,
then `minilm --gpu` on the same config-driven runner. All F32-only, OpenCL iGPU
via ILGPU 1.5.3. Sample-scoped (no `src/Nivara`,
`Nivara.Extensions`, or `src/Nivara.Gpu` changes). 128 tokens, 3-pass warmup +
10 timed, AC power (GPU↔CPU-Nivara same-session; PyTorch = recorded CPU baseline):

| scenario | Nivara GPU (iGPU) | Nivara CPU | PyTorch CPU | vs Nivara CPU | vs PyTorch |
|---|---|---|---|---|---|
| distilbert | **65.3 ms** (62–73) | 194.7 ms (152–232) | 35 ms | **~3.0× faster** | ~1.9× slower |
| distilbert_sst | **64.0 ms** (61–71) | 166.4 ms (134–208) | 35 ms | **~2.6× faster** | ~1.8× slower |
| minilm | **26.8 ms** (25–29) | 76.3 ms (49–102) | 11 ms | **~2.9× faster** | ~2.4× slower |

**M2 — kernel fusion (issue #437) landed 2026-09-19** on `khurram/lazystream-gpu`
(PR targets `khurram/minilm-gpu`). Same session, AC power:

| scenario | pre-M2 (M1) | post-M2 | Δ |
|---|---|---|---|
| minilm | 26.8 ms | **24.3 ms** (23–25) | **1.10×** |
| distilbert | 65.3 ms | **63.1 ms** (61–67) | **1.04×** |
| distilbert_sst | 64.0 ms | **63.7 ms** (62–67) | ~1.00× |

What fused (all sample-scoped; parity gates byte-identical at every step —
MiniLM maxAbs 1.87e-5 / DistilBERT 1.53e-5 / SST 3.8e-6 + argmax 8/8): GEMM
epilogue bias (+ exact-GELU for fc1, ReLU for the head pre-classifier) via lean
per-activation sibling kernels; `q/k/v` merged into **one** GEMM writing a
block-separable destination (`[q-block | k-block | v-block]`, dense SubViews for
the unchanged attention kernel); residual-add folded into LayerNorm;
word/position/token-type embedding sums fused into one gather; seqLen-
deterministic posIds cached. Dispatches went **~113–119 → 44–48** per forward.

**Honest outcome:** the pre-M2 launch model was wrong. Removing ~70 dispatches
moved MiniLM 26.8 → 24.3 ms, i.e. **~30 µs per dependent kernel** (not the
~0.18–0.2 ms first estimated — the model was off ~6× because GEMM + non-GEMM
compute dominate at these shapes). The #437 ">2× for MiniLM" acceptance
(~13.4 ms) is **not reachable via fusion on this iGPU** — it would need ~420
more dispatches removed, and only ~50 exist. The real lever is GEMM *throughput*
(§5 item 2); even an optimistic 2–3× GEMM speedup lands MiniLM near ~22 ms at
batch 1. The remaining micro-fusions were completed because they were already
designed; the numbers above are the honest result.

**Battery caveat confirmed again:** an intermediate session on battery produced
contaminated 25.6–33 ms MiniLM numbers drifting downward as the charge drained —
only AC numbers are valid. Do not benchmark this path on battery.

Correctness gates — **all PASS**:
- `distilbert --gpu compare`: final hidden `[128,768]` vs same-process CPU —
  maxAbs 1.53e-5, maxRel 3.24e-6, **0/98304 violations**
  (bound `|gpu−cpu| ≤ 1e-3·(1+|cpu|)`).
- `distilbert_sst --gpu compare`: logits vs CPU maxRel 6.7e-7, vs PyTorch
  fixture maxRel 5.8e-7, **argmax 8/8** both (CPU's own doc'd bound vs HF is
  9.5e-7 — same class).
- `minilm --gpu compare`: hidden `maxRel 1.04e-5`, pooled-embedding `maxRel 1.5e-7`,
  **0/245760 violations**, minimum pooled-embedding cosine **1.000000**.
- `--precision bf16|fp16` + `--gpu` → clear F32-only rejection (exit 1).

## 2. What shipped — architecture and decisions

All GPU code lives in `samples/Nivara.Samples/Gpu/`:

| file | contents |
|---|---|
| `IlgpuRuntime.cs` | context/accelerator/stream lifecycle; device select (`CL_DEVICE_TYPE_GPU` + Intel vendor, **asserted — no CPU fallback**); persistent buffer upload/download helpers |
| `GemmKernels.cs` | **Row4** register-blocked 1×4 tiled GEMM (local-memory staging, `TiledGemmKernelRow4`) — the keystone; used for every matmul (q/k/v/o, lin1, lin2, head). M2 added the fused-epilogue siblings (`Row4Bias`/`Row4Gelu`/`Row4Relu` — bias added in the register accumulators, GELU/ReLU folded into fc1 / head pre-classifier) and `Row4Qkv` — one packed `q/k/v` GEMM with a block-separable destination |
| `AttentionKernels.cs` | fused 12-head score+scale+mask+row-softmax+weighted-V (`XMath.Exp`) — unchanged by M2 (consumes the q/k/v SubViews as dense row-major views) |
| `ElementwiseKernels.cs` | LayerNorm row-reduce, GELU (direct A–S 7.1.26 erf poly port of `GradKernels.Erf`, `XMath.Exp`), bias/residual adds, embedding gather. M2 added `LayerNormResidual1D` (residual folded into the LN row-reduction) and `EmbeddingSum` (word/position/token-type gathers + sum in one launch) |
| `BertEncoderGpuRunner.cs` (was `DistilBertGpuRunner.cs`) | uploads weights (transposed once at ctor) by the loader key set — naming/role-resolved via `BertGpuNaming` (`DistilBert` | `Bert`), config-driven from `BertConfig`; runs the full BERT-family encoder forward + optional SST-2 head; returns per-stage readbacks for gating. M2: every projection through a fused epilogue launch, `q/k/v` one packed GEMM (`Wqkv`/`Bqkv` uploads), posIds cached per seqLen, `Add`/`AddBias` launches removed |

Design decisions (deliberate, and worth keeping for the next model):
- **Correctness-first runner**: one kernel launch per operation, everything on
  one stream, one device sync before readbacks. Fusion/stream tricks were
  deliberately deferred — the first milestone had to be trivially verifiable.
- **Weights transposed once at upload** so the GEMM consumes plain row-major
  operands (first-cut correctness over layout cleverness, as the assessment said).
- **Persistent cap-sized workspace** with `Ensure`-only growth; payload copies
  (mask/ids/posIds/clsIds) go through explicit-length `View.SubView(0, n)` —
  see the `CopyFromCPU` lesson below.
- **Attention mask semantics copied exactly from CPU**: per-batch column-j
  `−inf` when `mask[b][j] < 0.5`, identical across query rows.
- **Gate discipline**: hidden states + logits vs same-process CPU forward
  (identical tokenization) with the per-element bound above; SST-2 argmax 8/8;
  explicit GATE PASS/FAIL verdicts. Weights were **not** pre-provisioned —
  provisioned via `hf download` (`samples/data/distilbert{,_sst}`, ~511 MB,
  now part of the README quick start).

## 3. What we learned using ILGPU (for the next model)

These are the durable, non-obvious lessons from implementing the first model:

1. **`ArrayView.CopyFromCPU<T>(T[])` fills the *whole* view** — its guard is
   `data.Length ≥ view.Length` and it throws `ArgumentOutOfRangeException('data')`
   if your payload is shorter than the buffer. Copying a 128-element payload into
   a 1024-cap persistent buffer crashed the first gate run. Fix: copy into an
   explicit-length `View.SubView(0, n)`. There is no "copy prefix" overload.
2. **Stream ordering is your responsibility.** Buffer-level `CopyFromCPU` uses
   the default stream; kernels run on your own stream. Route *all* payload copies
   through the kernel stream, and add a device sync after bulk ctor uploads, or
   the uploads may not be visible to kernel launches. (This bit us; it also
   silently would have produced flaky results.)
3. **Launch/dispatch overhead dominates at inference shapes — but the per-kernel
   cost is ~30 µs, not the ~0.4–0.7 ms first estimated.** ~100 dispatches per
   forward at 128-row shapes. M2 (#437) removed ~70 of them and moved MiniLM
   26.8 → 24.3 ms / DistilBERT 65.3 → 63.1 ms — consistent with a
   **~30 µs dependent-kernel latency**. The corrected model: launch overhead
   was ~3 ms of the baseline, not ~21 ms. Fusion is nearly exhausted; the next
   lever is kernel *throughput* (GEMM headroom, §5 item 2), not launch count.
4. **A tiled GEMM is a requirement, not an optimization**: the probe's naive
   gemv rate (6.6 GMAC/s) would take ~825 ms for DistilBERT's matmuls. Row4
   (1 thread owns 4 accumulators, one shared A-tile load) hit **303–379 GMAC/s**
   on all DistilBERT shapes, keeping the per-output-column accumulation order
   unchanged → bit-identical results to the 1×1 kernel, which kept parity simple.
5. **Double-precision-truth GEMM gates catch real bugs.** A Row4 `colBase` bug
   (global thread-Y instead of group index) produced maxAbs ~40–77 and was
   caught by the bounds check. This is why issue #435 (a lasting GEMM
   correctness+perf harness) is worth doing.
6. **f32 parity between different summation orders has a floor.** Two valid
   reduction orders disagree at ~4.4e-5…1.6e-4 (K=768…3072), so a `1e-6`-class
   gate is impossible — the `1e-3·(1+|x|)` bound is right, and real bugs still
   produce ~40-class outliers that trip it. Measured hidden maxAbs 1.5e-5 is the
   summation-order floor, not a kernel defect.
7. **`XMath` has no `erf`** — GELU needs an in-kernel polynomial. The direct
   A–S 7.1.26 port with `XMath.Exp` matched CPU `GeluExact` inside hidden-state
   parity (maxRel 3.2e-6); no `XMath.Tanh`-based approximation needed.
8. **Battery throttles the iGPU flat** — every shape drops to ~150–160 GMAC/s
   regardless of size (this finally explained the probe's flat gemv class). **All
   GPU perf claims require AC power.**
9. **ILGPU deployment is the whole story**: NuGet package = entire install,
   in-box Windows `OpenCL.dll` + Intel ICD; kernels are C# static methods.
   Assert `CL_DEVICE_TYPE_GPU` (no silent CPU fallback) and re-run the gates
   after every driver bump (two compilers in the path: ILGPU JIT + IGC).
10. **iGPU shares DRAM** — the 255.5 MB weight upload is memcpy-class (not a
    PCIe bottleneck); transposing once at upload costs nothing and buys GEMM
    coalescing.
11. **Workspace hygiene**: allocate cap buffers once and reuse (`Ensure` only
    grows); integer payloads need an explicit integer-buffer alloc. Readbacks
    via `GetAsArray()` return the full cap — always slice by payload length.
12. **Same-process CPU reference beats fixture-only gating**: the compare mode
    (identical tokenization, `BertEncoder.Forward` vs the GPU runner in one
    process) caught everything the fixture couldn't (fixtures may be absent).

## 4. Where reality diverged from the pre-implementation estimate

Honest delta between the assessment's expectations and the measured outcome:

- **E2E landed at ~65 ms vs the estimated ~15–25 ms.** The GEMM extrapolation
  was right (Row4 rates × 5.44 GMAC ≈ 20–30 ms); the miss was the *unmeasured
  portion* — attention + elementwise legs **plus per-dispatch overhead × ~100
  dispatches**, first over-estimated at ~0.4–0.7 ms/launch. M2 measured the
  truth: **~30 µs/dependent kernel**, so the ~70 dispatches fusion removed
  (MiniLM 26.8 → 24.3 ms) matched the corrected model, and the remaining gap is
  kernel throughput — not a launch-count problem fusion can fix.
- **M2's >2× acceptance was not reachable.** The #437 ">2× MiniLM" target
  (~13.4 ms) assumed ~0.19 ms/launch; at the measured ~30 µs it would need
  ~420 fewer dispatches than exist. Re-scoped to honest ~1.1× with the GEMM-
  throughput item promoted to the leading GPU follow-up.
- **The "clear win vs PyTorch CPU 35 ms" did not materialize.** The iGPU is
  ~1.9× *behind* PyTorch's MKL-backed CPU path here (PyTorch starts ~5.6× ahead
  of Nivara-CPU). We still won ~3.0× over our own CPU, which was the scenario's
  actual goal — but the assessment's PyTorch-vs-GPU framing was optimistic for
  a 128-EU iGPU.
- **The keystone GEMM decision gate held exactly as designed**: Row4 measured
  303–379 GMAC/s (inside the 0.6–1.6 T MAC/s *good* window) and the OpenVINO
  fallback was never needed.

## 5. Suggestions & follow-ups

Prioritized for the next iterations of the GPU journey (see also
[ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) for the CPU-side picture):

1. ~~**Kernel fusion + lazy stream (M2)**~~ — **DONE 2026-09-19**
   (`khurram/lazystream-gpu` → PR targets `khurram/minilm-gpu`): every planned
   fusion landed and the parity gates stayed byte-identical. The lazy-stream
   half was already true (one stream, one sync before readback). Honest result:
   MiniLM 26.8 → **24.3 ms**, DistilBERT 65.3 → **63.1 ms** — the per-launch
   model was corrected by measurement to ~30 µs, so launch-count fusion is
   nearly exhausted; the remaining gap is kernel throughput.
2. **GEMM headroom is now the leading GPU item** — a 2×2 or tile-32
   register-block pass (Row4 measured 303→379 GMAC/s across shapes;
   occupancy/register headroom exists) is the only realistic path toward the
   2×-class numbers fusion-targeted. Tracked as the follow-up issue created from
   M2 (see #437 + ROADMAP-SUGGESTION.md). Keep the double-truth bounds check —
   and promote it into issue **#435** (lasting GEMM regression harness).
3. **Transpose-free in-kernel GEMM** — dropped for first-cut correctness; worth
   revisiting if upload time ever matters (it doesn't here — shared DRAM).
4. **bf16/fp16 GPU** — later decision; neither precision gets **native** support
   in ILGPU 1.5.3, so F32 stays the GPU path until a promotion-phase decision
   (possibly re-evaluating the backend for that model class):
   - **BF16: no native type or kernels at all.** The only proven route is the
     "unmerged packed-widen" trick — pack two BF16 values per 32-bit slot, widen
     to FP32 on load, compute in FP32, pack back. That pays FP32-class compute
     plus pack/unpack overhead, negating most of BF16's memory/bandwidth win.
   - **FP16 (`Half`): software-emulated only.** ILGPU ships a `Half` kernel type
     but its arithmetic is emulated scalar math — there are **no native
     vectorized FP16 kernels**. Real hardware FP16 in OpenCL requires the
     `cl_khr_fp16` extension (device/context-gated), which ILGPU's OpenCL path
     does not drive for you, so no hardware-FP16 performance is reachable either.
   - For contrast: OpenVINO's bf16 gemm probe row (86.8 µs, ~5.9 T MAC/s) is
     genuinely native — it sits on Xe2 **DPAS BF16** hardware instructions that
     ILGPU 1.5.3 cannot reach.
   The F32-only reject (`--gpu` + `--precision bf16|fp16` → clear error) keeps
   the door clean until that decision.
5. ~~**Second model**: MiniLM~~ — **DONE on `khurram/minilm-gpu`** (PR follows; on
   the same "one config-driven runner, N encoder models" shape at ~20% of the
   original effort): the runner is now `BertEncoderGpuRunner`, naming- and
   config-driven (`BertGpuNaming.DistilBert | Bert`; `BertConfig` ctor), so
   MiniLM reused the kernel set unchanged. One genuine generalization lesson:
   MiniLM (BERT-style keys) **does** feed token-type embeddings — the CPU
   `BertEncoder` defaults `includeTokenTypeEmbedding: true` and PyTorch adds
   `token_type_embeddings[0]` (all-zero segment ids) — where DistilBERT does
   not; the runner broadcasts that row 0 when the key is present (key-presence
   driven, so the DistilBERT path is untouched). Gates: hidden `maxRel 1.0e-5`,
   0/245760 violations, pooled-embedding cosine 1.000000; benchmark **26.8 ms**
   iGPU vs 76.3 ms Nivara CPU (~2.9×), launch-overhead-bound at ≈1.36 GMAC —
   tracked as **#437**; post-M2 (fused epilogues + q/k/v merge, same runner):
    **24.3 ms** (see the M2 block in §1) — still latency-bound at batch 1, which
    is exactly why the GEMM-throughput item (§5.2) is now the leading follow-up.
    SmolLM once KV-cached decode exists (see SMOLLM-GPU.md)
   remains the next, much larger candidate.
6. **Promotion decision**: with real measured numbers in hand, decide whether
   GPU support moves into `src/Nivara.Gpu` (which backend, which project, bf16,
   which models). Nothing in core changes until that decision.
7. **Regenerate the `last_hidden_state_py.bin` fixture** (absent in this
   checkout) to restore the full 3-way GPU-vs-CPU-vs-PyTorch gate on hidden
   states, matching the SST-2 fixture coverage.
8. **Driver-bump hygiene**: re-run the probe `kernels` gate (seconds) and the
   two model compares after any Intel driver update (two compilers in the path).
9. **CPU-side performance (independent thread)** — a managed tiled GEMM and/or
   an opt-in native BLAS bridge would close the ~5.6× Nivara-CPU deficit
   (possibly beating the GPU at these shapes); options, targets, and the
   M1/M2/M3 decision gate are captured in ROADMAP-SUGGESTION.md.

## 6. References

- [ILGPU.md](ILGPU.md) — probe verdicts: `Index1D`, padded-grid bounds checks,
  `CL_DEVICE_TYPE_GPU`, `AsContiguous().GetAsArray()`, `MathMode.Fast` off
- [COMPUTESHARP.md](COMPUTESHARP.md) — the complementary managed-D3D12 backend,
  kept in reserve
- [SMOLLM-GPU.md](SMOLLM-GPU.md) — why SmolLM is the follow-up, not the first
  scenario
- [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) — CPU/GPU-next roadmap, options,
  decision gate
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) —
  usage, `--gpu` quick start, benchmark tables, CPU baselines
- [`tests/Nivara.GpuProbe/README.md`](../tests/Nivara.GpuProbe/README.md) —
  seven-way backend comparison, `kernels` correctness gate
- Issues: **#435** (tiled-GEMM regression gate, open) · **PR #436** (this
  scenario)