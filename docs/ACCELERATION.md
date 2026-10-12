# Acceleration

This is the **acceleration index**: the story of Nivara's GPU work — why it was built, what it
measured, and where it is going — plus the map into the documents that carry the code-level detail.
For model-specific architecture and usage, see the model pages:

- [DISTELBERT.md](DISTELBERT.md) — DistilBERT, MiniLM, SST-2
- [MODERNBERT.md](MODERNBERT.md) — ModernBERT-large
- [SMOLLM.md](SMOLLM.md) — SmolLM-135M
- [QWEN.md](QWEN.md) — Qwen2.5-0.5B-Instruct
- [LAYA.md](LAYA.md) — Laya decision head
- [VISION.md](VISION.md) — MobileNetV2, ResNet-18

**Where the detail lives.** This page is the story and the index. The code-level implementation —
architecture, every kernel, geometry, design decisions and the durable lessons — is in
[ILGPU.md Part 2](ILGPU.md) (ILGPU/OpenCL) and [SILK.md Part 2](SILK.md) (the hand-authored `--silk`
backend). The probe verdicts are ILGPU.md Part 1 and SILK.md Part 1; the shipped streaming contract is
[STREAMING.md](STREAMING.md) and its design proposal is
[STREAMING-PROPOSAL.md](STREAMING-PROPOSAL.md). Backend case studies live in
[`tests/Nivara.GpuProbe/`](../tests/Nivara.GpuProbe/README.md).

---

# GPU Acceleration — ILGPU on Intel Arc

Status: **Implemented, gated, and measured** (2026-09-19; DistilBERT via PR #436, MiniLM via
`khurram/minilm-gpu`, ModernBERT via issue #449 on `khurram/449`). A second, hand-authored OpenCL
backend (`--silk`, Silk.NET) covers the BERT-family encoders and DistilBERT's SST-2 head — MiniLM,
DistilBERT and DistilBERT+SST-2; it is gated, and given an informal battery, same-session A/B against
ILGPU (no separation) — see "Second OpenCL backend" under §1.

## 1. Measured results

All F32-only, OpenCL iGPU via ILGPU 1.5.3. Sample-scoped (no `src/Nivara`, `Nivara.Extensions`, or
`src/Nivara.Gpu` changes). 128 tokens, 3-pass warmup + 10 timed, AC power (GPU↔CPU-Nivara same-session;
PyTorch = recorded CPU baseline):

| scenario | Nivara GPU (iGPU) | Nivara CPU | PyTorch CPU | vs Nivara CPU | vs PyTorch |
|---|---|---|---|---|---|
| distilbert | **65.3 ms** (62–73) | 194.7 ms (152–232) | 35 ms | **~3.0× faster** | ~1.9× slower |
| distilbert_sst | **64.0 ms** (61–71) | 166.4 ms (134–208) | 35 ms | **~2.6× faster** | ~1.8× slower |
| minilm | **26.8 ms** (25–29) | 76.3 ms (49–102) | 11 ms | **~2.9× faster** | ~2.4× slower |

The "vs PyTorch" column is **shape-specific, not a standing property of the iGPU.** At 128 tokens
and 66 M parameters a 128-EU iGPU is latency- and bandwidth-bound while 16 CPU threads have work to
fill. The same accelerator **ties** PyTorch at Laya's shape — 512 tokens, 1.4 B parameters, iGPU
3.00 s against PyTorch 2.97 s — where quadrupling `M` in every GEMM finally gives the device enough
parallel work per weight load. One shape is one data point and the §1 rows are unchanged, but read
the deficit as "at these shapes" rather than as a property to assume anywhere else; measure at the
shape you intend to ship. Detail in [docs/LAYA.md](LAYA.md).

**M2 — kernel fusion (issue #437) landed 2026-09-19** on `khurram/lazystream-gpu` (PR targets
`khurram/minilm-gpu`). Same session, AC power:

| scenario | pre-M2 (M1) | post-M2 | Δ |
|---|---|---|---|
| minilm | 26.8 ms | **24.3 ms** (23–25) | **1.10×** |
| distilbert | 65.3 ms | **63.1 ms** (61–67) | **1.04×** |
| distilbert_sst | 64.0 ms | **63.7 ms** (62–67) | ~1.00× |

What fused (all sample-scoped; parity gates byte-identical at every step — MiniLM maxAbs 1.87e-5 /
DistilBERT 1.53e-5 / SST 3.8e-6 + argmax 8/8): GEMM epilogue bias (+ exact-GELU for fc1, ReLU for the
head pre-classifier) via lean per-activation sibling kernels; `q/k/v` merged into **one** GEMM writing
a block-separable destination (`[q-block | k-block | v-block]`, dense SubViews for the unchanged
attention kernel); residual-add folded into LayerNorm; word/position/token-type embedding sums fused
into one gather; seqLen-deterministic posIds cached. Dispatches went **~113–119 → 44–48** per forward.

**Honest outcome:** the pre-M2 launch model was wrong. Removing ~70 dispatches moved MiniLM
26.8 → 24.3 ms, i.e. **~30 µs per dependent kernel** (not the ~0.18–0.2 ms first estimated — the
model was off ~6× because GEMM + non-GEMM compute dominate at these shapes). The #437 ">2× for
MiniLM" acceptance (~13.4 ms) is **not reachable via fusion on this iGPU** — it would need ~420 more
dispatches removed, and only ~50 exist. The real lever is GEMM *throughput*; even an optimistic
2–3× GEMM speedup lands MiniLM near ~22 ms at batch 1. The remaining micro-fusions were completed
because they were already designed; the numbers above are the honest result.

**Battery caveat confirmed again:** an intermediate session on battery produced contaminated
25.6–33 ms MiniLM numbers drifting downward as the charge drained — only AC numbers are valid. Do not
benchmark this path on battery.

Correctness gates — **all PASS**:
- `distilbert --ilgpu compare`: final hidden `[128,768]` vs same-process CPU — maxAbs 1.53e-5,
  maxRel 3.24e-6, **0/98304 violations** (bound `|gpu−cpu| ≤ 1e-3·(1+|cpu|)`).
- `distilbert_sst --ilgpu compare`: logits vs CPU maxRel 6.7e-7, vs PyTorch fixture maxRel 5.8e-7,
  **argmax 8/8** both (CPU's own doc'd bound vs HF is 9.5e-7 — same class).
- `minilm --ilgpu compare`: hidden `maxRel 1.04e-5`, pooled-embedding `maxRel 1.5e-7`,
  **0/245760 violations**, minimum pooled-embedding cosine **1.000000**.
- `--precision bf16|fp16` + `--ilgpu` → clear F32-only rejection (exit 1).

### ModernBERT GPU (issue #449)

`ModernBertGpuRunner` + `modernbert --ilgpu`. Same device (`Intel(R) Graphics`), same ILGPU 1.5.3
OpenCL path, still sample-scoped. This is the first runner whose architecture shares **no** structure
with DistilBERT or MiniLM: rotary instead of learned positions, a gated feed-forward instead of a
plain FFN, pre-norm instead of post-norm, and **per-layer attention geometry**. The kernel-level
changes it required — a `band` parameter, the `max == -inf → zeros` clamp, the removal of the
`[B, H, S, S]` score spill, and the new elementwise kernels — are in
[ILGPU.md §11–§12](ILGPU.md).

The gate is GPU-vs-CPU in-process, so it needs no PyTorch fixture (Phase 1 had already pinned the
CPU encoder to HuggingFace, so this pins the GPU runner transitively). Fixture seqLen 128, 26 valid
positions:

| region | maxAbs | maxRel | cosine | violations |
|---|---|---|---|---|
| valid (26 x 1024 = 26,624 values) | **2.861E-005** | 5.577E-004 | 1.0000001 | 0 |
| padding (104,448 values) | 6.866E-004 | 4.867E-003 | — | 0 |

Forward: **415–449 ms GPU vs a 7.3–7.5 s CPU reference** (the CPU figure is a 28-layer float forward
with no batching, so it is not a throughput claim — see below). Benchmark rows scale with sequence
length, as they must: 128 → 400 ms, 512 → 2.30 s, 2048 → 19.6 s, 4096 → 63.0 s. Those rows use
**1 warmup + 3 timed passes, median reported**, not the 3 + 10 of the §1 table — at S=4096 a single
pass is ~60 s — so they are not comparable to the §1 numbers and are not in that table; the mode
prints this caveat itself.

**This issue bought correctness, not speed — and the issue's premise was stale.** #449 expected
attention to become the bottleneck. [docs/LAYA.md](LAYA.md) (merged in PR #459, four hours after the
issue was filed) had already measured GEMM at **~99.9% of the arithmetic** with attention at 0.13% at
S=512, and named #440 (tile-32 / 2x2 GEMM) the leading open item. *(That nomination was later
measured and did not hold; and the arithmetic-share framing was itself the deeper error — a leg
profile measures `BatchedAttention` at 57% of the Laya forward, not 0.13%.)* ModernBERT has 1.4 B
parameters against DistilBERT's 66 M, so it is a GEMM-throughput problem exactly as DistilBERT was.
The value delivered here is that the architecture is *portable at all* — a banded RoPE'd gated
pre-norm encoder now runs on the accelerator and is pinned to the CPU encoder — not that it is faster.

**Banded attention (issue #447, done 2026-09-30)** took ModernBERT's remaining GPU lever: the band
now restricts the loop bounds of all three attention passes instead of only overwriting an
already-computed score with `-inf`, so a sliding layer's cost tracks its 129-wide window rather than
`seqLen`. Measured end to end on an **Intel Iris Xe** iGPU (the same device class as the #553
profile), AC power, F32, 1 warmup + 3 timed median, ModernBERT-large (28 layers, 10 full + 18 sliding,
band 64), against the same binary with the dense sweep restored:

| seq | dense (pre-#447) | banded | speedup |
|---|---|---|---|
| 128 | 1497.8 ms | 1397.7 ms | 1.07× |
| 512 | 9224.3 ms | 7264.8 ms | 1.27× |
| 2048 | 83342.7 ms | 57117.7 ms | 1.46× |
| 4096 | 279704.1 ms | 176001.2 ms | 1.59× |

Two cautions on that table. The 4096 row is the weakest number in it: the banded run's min and median
differ by 11.8% (157450.2 vs 176001.2 ms) where every other row's differ by under 1%, and the saving
there is ~21× the saving at 2048 where `S²` scaling of the attention work predicts ~4×. It is reported
because it was measured, not because it is understood; `modernbert --ilgpu benchmark` now takes
`--seq` (**#474**), so that one row can be re-measured on its own with `--seq 4096 --iters 10
--warmup 3`. **That re-measurement has not been run**, so the table stands as-is. The change is
provably output-invariant — `Keep` already suppressed every out-of-band key, and such a key
contributed `exp(-inf - max) = 0` and then `0 * v = ±0` to an accumulator starting at `+0` —
confirmed by reverting the kernel to the dense sweep and re-running `GpuAttentionBandTests` (all
green) and `modernbert --ilgpu` (0 of 10 sentences differ in any hidden-state statistic).
`modernbert --ilgpu compare` passes on this machine at max abs 3.052E-005, max rel 4.807E-004 (48% of
the 1e-3 bound), cosine 1.0000000000, 0 non-finite on either side.

**The Laya head (issues #460, #462) reached the GPU through the same seam.** `LayaHeadGpuRunner`
reuses this trunk through an additive `ForwardOnDevice`/`HiddenOnDevice`/`HiddenRows` seam, needing
**zero** new kernels — the head's ops are pre-norm, *biased* LayerNorm, ReLU and fused *biased* QKV,
all already present. Two traps worth naming: the head's norms are **biased** where every ModernBERT
norm is not (the shared zero beta would be a silent wrong answer), and the head's eps is
`LayaDecisionHead`'s `1e-5` default, not `config.NormEps`. **On the gate, the generalisable lesson:**
it runs at the same 1e-3 bound and passes at **1.1%** of it, while the encoder gate spends 56% — even
though the head sits *on top of* that encoder. A gate's tightness is a property of the quantity being
compared, not of the depth behind it: the encoder compares the raw `[L, 1024]` hidden state where 28
layers of reduction-order drift land undiluted, while this compares a `1024 → 1` projection taken
after a LayerNorm has re-normalized each row, so per-position errors average down by
`sqrt(1024) = 32` and the bias component is removed outright. Do not carry a hidden-state budget over
to a projected output. **Perf:** `laya --ilgpu benchmark` vs `laya benchmark`, whole model at
`max_len` 512, 1 warmup + 3 timed median — CPU ≈8.48 s vs iGPU ≈3.00 s, **≈2.83×** (2.62–2.95× per
question), the same 2.6–3.0× band as the §1 BERT-family rows and **tying** PyTorch at this shape
(3.00 s vs 2.97 s). Reflection in [docs/LAYA.md](LAYA.md).

### Second OpenCL backend: `--silk` (Silk.NET, hand-authored kernels)

A second GPU path runs the BERT-family encoder through OpenCL C written by hand and compiled
in-process by the driver (`Silk.NET.OpenCL` 2.23.0), reached with `--silk` instead of `--ilgpu`. It
exists to evaluate a vendor-neutral, non-ILGPU route — the same shore the `tests/Nivara.GpuProbe`
Silk.NET leg probes — on a real model rather than on microkernels.

- **Scope.** **MiniLM, DistilBERT and DistilBERT+SST-2**, F32 only, modes `default` / `benchmark` /
  `compare`. `--silk` and `--ilgpu` are mutually exclusive, and `--silk --precision bf16|fp16` is
  rejected with the same F32-only contract the ILGPU path uses. A checkpoint whose head weights the
  kernel set cannot serve is rejected at runner construction rather than answered with the encoder
  hidden state, which would be silent wrong output.
- **The kernel set** is eight kernels in `samples/Nivara.Samples/Gpu/SilkKernels.cs`, one per
  operation the family needs: `embed_sum`, `gather_rows`, `layer_norm_residual`, the four Row4 GEMM
  epilogues (`gemm_row4_bias`, `gemm_row4_relu`, `gemm_row4_gelu`, `gemm_row4_qkv`) and
  `batched_attention`. They are ports of the ILGPU kernels of the same shape — same K-reduction order,
  same tile geometry, same fused epilogues — and the Row4 GEMMs share one accumulation macro so their
  f32 text cannot diverge. `#pragma OPENCL FP_CONTRACT OFF` is load-bearing: ILGPU does not fuse
  `acc += a*b`, the OpenCL driver fuses by default, so without the pragma the Silk side computes the
  fma model and diverges at rounding scale. Detail in [SILK.md §11](SILK.md).
- **Host side** is `SilkKernels.cs` (source + geometry), `SilkRuntime.cs` (device select, in-process
  compile with the build log decoded on failure, buffers, dispatch) and `SilkBertEncoderRunner.cs`.
  Both runners implement `IBertEncoderRunner`, so the CLI's BERT-family entry points are written once
  and pick a backend at construction; the weight-layout maths (`GpuBuffers`) and the launch geometry
  (`SilkKernels`) are shared rather than re-derived per backend. The DistilBERT support is a
  `BertGpuNaming` constructor parameter on the same runner, not a fork.
- **q/k/v reach attention as read-only sub-buffers** (`clCreateSubBuffer`, the OpenCL analogue of
  ILGPU's `SubView`), created once per forward, so the already-gated fused attention kernel is
  untouched.
- **Gates.** `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` holds
  `gemm_row4_bias`, `gemm_row4_relu`, `gemm_row4_qkv` and `gather_rows` to **byte-identity** against
  ILGPU and the remaining kernels to the model's numeric bound, over both MiniLM and DistilBERT
  shapes; `minilm --silk compare`, `distilbert --silk compare` and `distilbert_sst --silk compare` are
  the end-to-end gates (`|gpu−cpu| ≤ 1e-3·(1+|cpu|)`, plus pooled-embedding cosine for MiniLM and
  argmax 8/8 for SST-2). `InferenceCliArgumentTests` pins the new rejects. The gate's timing column is
  explicitly *not a verdict*: on battery a few cells leave the informal x0.8..x1.2 band — consistently
  DistilBERT's larger GEMM shapes (qkv, fc1), four to six of the 32 in recent runs, worst ≈1.4× — and
  the exact set moves run to run. Flagged, not failed, and not tracked as an issue.
- **Informal A/B (battery; both legs on battery, same session).** The `--silk-kernels` gate runs the
  ILGPU originals and the Silk kernels **in one process**, so its timing column compares the two
  backends under one clock and one power state. On the **MiniLM** shapes the per-kernel ratio
  (Silk / ILGPU) fell in **0.79–1.19**; the drift is concentrated in the larger DistilBERT GEMM
  shapes (above). A back-to-back `minilm --silk benchmark` / `minilm --ilgpu benchmark` pair (same
  3-warmup + 10-timed protocol) compares the whole forward: it averaged **29.5 ms (Silk) vs
  31.4 ms (ILGPU), ≈0.94×**. Read these as "no separation", not as rows to quote: they are
  **battery** numbers, the gate's timing column is explicitly *not a verdict*, and the absolute
  milliseconds are not comparable to the AC-powered §1 table. (The gate prints the same battery
  warning at the top of every run.)

## 2. Where the implementation detail lives

| topic | document |
|---|---|
| ILGPU runtime, GEMM family (Row4, the #440 measured-null geometries, #468), attention, elementwise kernels, the BERT/ModernBERT/Laya runners, design decisions and the 19 durable lessons | [ILGPU.md Part 2](ILGPU.md) |
| the `--silk` runtime, translation unit, kernels, runner, CLI surface and gate; adding a model to the backend | [SILK.md Part 2](SILK.md) |
| probe verdicts for every backend leg (SPIR-V, SYCL, DX12, OpenVINO, ILGPU, ComputeSharp, Silk.NET) | [ILGPU.md Part 1](ILGPU.md) · [SILK.md Part 1](SILK.md) · [`tests/Nivara.GpuProbe/`](../tests/Nivara.GpuProbe/README.md) |
| the GEMM gate baseline and the repeatable benchmark harness | [`tests/Nivara.PerformanceTests/README.md`](../tests/Nivara.PerformanceTests/README.md) |
| Laya decision head (CPU + GPU) | [LAYA.md](LAYA.md) |
| shipped streaming contract · design proposal | [STREAMING.md](STREAMING.md) · [STREAMING-PROPOSAL.md](STREAMING-PROPOSAL.md) |

## 3. Where reality diverged from the pre-implementation estimate

Honest delta between the assessment's expectations and the measured outcome:

- **E2E landed at ~65 ms vs the estimated ~15–25 ms.** The GEMM extrapolation was right (Row4 rates ×
  5.44 GMAC ≈ 20–30 ms); the miss was the *unmeasured* portion — attention + elementwise legs **plus
  per-dispatch overhead × ~100 dispatches**, first over-estimated at ~0.4–0.7 ms/launch. M2 measured
  the truth: **~30 µs/dependent kernel**, so the ~70 dispatches fusion removed (MiniLM
  26.8 → 24.3 ms) matched the corrected model, and the remaining gap is kernel throughput — not a
  launch-count problem fusion can fix.
- **M2's >2× acceptance was not reachable.** The #437 ">2× MiniLM" target (~13.4 ms) assumed
  ~0.19 ms/launch; at the measured ~30 µs it would need ~420 fewer dispatches than exist. Re-scoped
  to an honest ~1.1× with the GEMM-throughput item promoted to the leading GPU follow-up.
- **The "clear win vs PyTorch CPU 35 ms" did not materialize.** The iGPU is ~1.9× *behind* PyTorch's
  MKL-backed CPU path here (PyTorch starts ~5.6× ahead of Nivara-CPU). We still won ~3.0× over our own
  CPU, which was the scenario's actual goal — but the assessment's PyTorch-vs-GPU framing was
  optimistic for a 128-EU iGPU.
- **The keystone GEMM decision gate held exactly as designed**: Row4 measured 303–379 GMAC/s
  (0.61–0.76 TFLOP/s), inside the 0.6–1.6 TFLOP/s *good* window, and the OpenVINO fallback was never
  needed.

# Not done / planned

Every open, deferred or future item, one line each — status, home document, issue number.

- **Promote the GPU paths into `src/Nivara.Gpu`** — pending; decide which backend, which project,
  bf16, which models. Nothing in core changes until then. Home: this doc.
- **#435 lasting GEMM regression harness** — open; partly delivered, since the `--gemm` gate now
  asserts byte-identity as well as the double-precision truth check. Home:
  [`tests/Nivara.PerformanceTests/README.md`](../tests/Nivara.PerformanceTests/README.md).
- **bf16 / fp16 on the GPU** — pending promotion-phase decision; neither precision gets native
  arithmetic from ILGPU 1.5.3, and the raw-OpenCL route offers a native BF16 *conversion* but no BF16
  multiply. Home: [SILK.md](SILK.md) §"Precision on this iGPU (BF16 and FP16)".
- **Transpose-free in-kernel GEMM** — deferred; worth revisiting only if upload time ever matters
  (it does not here — shared DRAM). Home: [ILGPU.md §16](ILGPU.md).
- **Regenerate the `last_hidden_state_py.bin` fixture** (absent in this checkout) to restore the full
  3-way GPU-vs-CPU-vs-PyTorch hidden-state gate, matching SST-2's coverage. Home:
  [DISTELBERT.md](DISTELBERT.md).
- **Driver-bump hygiene** — standing; re-run the probe `kernels` gate and the model compares after any
  Intel driver update (two compilers in the path). Home: [ILGPU.md Part 1](ILGPU.md).
- **#447 4096-token re-measurement** — pending; `modernbert --ilgpu benchmark --seq 4096 --iters 10
  --warmup 3` is now cheap, and the 4096 row of the banded table is the one unexplained number. Home:
  this doc §1.
- **#473 ModernBERT CPU `MaxDenseLength` lift (2048 → longer)** — a feasibility fix, not a speed one.
  Home: [MODERNBERT.md](MODERNBERT.md).
- **#467 LayerNorm one-work-item-per-row** — minor open GPU kernel improvement. Home:
  [ILGPU.md §12](ILGPU.md).
- **#468 ILGPU OpenCL `Allocate2D` lowering bug** — upstream; the "at most one shared allocation per
  kernel body" rule is the workaround and is gate-enforced. Home: [ILGPU.md §10.2](ILGPU.md).
- **#480 measure the scalar `ApplyMask` cost · #481 attention-mask gradient** — open CPU/AutoDiff
  follow-ups from the #448 mask-as-select work. Home: [AUTODIFF.md](AUTODIFF.md).
- **#457 GPU vision kernels · #458 CPU-side performance deficit** — open; the convolution kernels are
  naive nested loops, and a managed tiled GEMM and/or an optional native BLAS bridge would close the
  ~5.6× Nivara-CPU deficit (possibly beating the GPU at these shapes). Home:
  [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) · [VISION.md](VISION.md).
- **#325 streaming memory-budget *enforcement*** — design proposal, not implemented; the budget
  reaches the strategy (ADR-006) but a hard in-flight ceiling does not. Home:
  [STREAMING-PROPOSAL.md](STREAMING-PROPOSAL.md).
- **#325 Phase 2 — spill-to-disk for boundary operators · external sort / hash join** — future; only
  makes sense after sources obey backpressure, and needs operator-level spill abstraction. Home:
  [STREAMING-PROPOSAL.md](STREAMING-PROPOSAL.md).
- **Hard process-level memory limits · budget-adaptive chunk sizing** — future; .NET GC, runtime and
  native decoders all contribute memory outside Nivara's control. Home:
  [STREAMING-PROPOSAL.md](STREAMING-PROPOSAL.md).

## References

- [ILGPU.md](ILGPU.md) — probe verdicts (Part 1) and the production ILGPU/OpenCL implementation
  (Part 2): `Index1D`, padded-grid bounds checks, `CL_DEVICE_TYPE_GPU`, `AsContiguous().GetAsArray()`,
  `MathMode.Fast` off, the GEMM family and the durable lessons.
- [SILK.md](SILK.md) — the Silk.NET OpenCL probe leg (Part 1) incl. the measured BF16/FP16 answer, and
  the `--silk` production backend (Part 2).
- [SMOLLM.md](SMOLLM.md) — SmolLM GPU investigation.
- [STREAMING.md](STREAMING.md) — streaming execution strategy (shipped contract);
  [STREAMING-PROPOSAL.md](STREAMING-PROPOSAL.md) — the memory-budget enforcement proposal.
- [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) — CPU/GPU-next roadmap, options, decision gate.
- [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md) — usage, `--ilgpu`
  quick start, benchmark tables, CPU baselines.
- [`tests/Nivara.GpuProbe/README.md`](../tests/Nivara.GpuProbe/README.md) — eight-way backend
  comparison, `kernels` correctness gate. **The probe legs are not a backend-selection argument:**
  each runs a *naive* reference kernel (one work-item per row, serial reduction, no tiling), so their
  timings are not comparable to the production `Row4` GEMM above.
- Issues: **#325** (streaming memory budget) · **#435** (tiled-GEMM regression gate, open) ·
  **PR #436** (DistilBERT GPU) · **#437** (kernel fusion, done) · **#440** (tile-32/2x2 GEMM,
  **measured null** — closed) · **#447** (banded/sparse attention, **done** 2026-09-30) · **#448**
  (mask-as-select, **done** 2026-09-30) · **#449** (ModernBERT GPU path) · **#460** + **#462** (Laya
  decision head, CPU + GPU) · **#467** (LayerNorm one-work-item-per-row, minor) · **#468** (ILGPU
  OpenCL `Allocate2D` lowering, upstream).
