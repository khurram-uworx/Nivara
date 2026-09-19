# Nivara CPU/GPU performance roadmap — suggestion (captured 2026-09-19)

> **Status: suggestion only — not yet planned or executed.** Captured verbatim from
> the DistilBERT-GPU wrap-up discussion so the options aren't lost. This is a
> scratch/record doc (the iterative-work plan file `docs/TODO.md` was deleted at
> G2); if we pick this up later, it graduates into a proper plan (with issues,
> grounding, gates) before any implementation.

## 1. Where we are — milestone 1 done

DistilBERT GPU first scenario shipped on branch `khurram/distilbert-gpu`
(**PR #436**, open): `distilbert --gpu` / `distilbert_sst --gpu` via ILGPU 1.5.3
(F32-only), sample-scoped (no `src/Nivara` / `Nivara.Extensions` / `Nivara.Gpu`
changes), correctness gates PASS, measured on AC power.

Same-session/recorded numbers (128 tokens, 3-pass warmup + 10 timed, AC power;
PyTorch = recorded 2026-09-01 CPU baseline, aligned architecture):

| scenario | Nivara GPU (iGPU) | Nivara CPU | PyTorch CPU |
|---|---|---|---|
| distilbert | **65.3 ms** (62–73) | 194.7 ms (152–232) | 35 ms |
| distilbert_sst | **64.0 ms** (61–71) | 166.4 ms (134–208) | 35 ms |

- vs Nivara CPU: **~3.0× / ~2.6× faster** (GPU wins).
- vs PyTorch CPU: **~1.9× / ~1.8× slower** (GPU loses) — PyTorch starts ~5.6×
  ahead of Nivara-CPU; the iGPU closes most of that gap without fully catching up.
- Battery note: iGPU throttles to ~150–160 GMAC/s (all shapes); GPU timings only
  meaningful on AC (~110–135 ms/forward on battery vs 62–73 ms on AC).

A **second model shipped on the same infra** (`khurram/minilm-gpu`): the runner
generalized to a config/naming-driven `BertEncoderGpuRunner`, so MiniLM (BERT-style
keys, 384-dim) reused the kernel set unchanged at ~20% of the original effort —
26.8 ms iGPU vs 76.3 ms Nivara CPU (~2.9×). Details and the token-type lesson in
`docs/BERT-GPU.md` §5.5; launch-overhead follow-up is issue **#437**.

## 2. The gap we want to attack (CPU GEMM)

PyTorch's 35 ms CPU figure is MKL-class:

- **MKL** = Intel (one)MKL — Intel's tuned math library (GEMM/BLAS, FFTs, …).
  PyTorch CPU delegates matmuls to it. MKL wins via: AVX/AVX-512 SIMD (8–16
  floats/instruction/core), multi-threading across all logical processors, and a
  *per-shape* tuned-kernel selection (cache-blocked, shape-specific tile choices)
  that years of hand-tuned asm provide.
- Nivara CPU GEMM today: correct but untuned managed kernels (`TensorPrimitives`
  / `MatMulTransposedB` in `LlamaFusedKernels`; `ReverseGradOperations.MatMul`).
  No shape-tuned tiling, no `Vector512` register blocking.
- GEMM dominates transformer forwards (DistilBERT ≈ 5.44 GMAC/forward; attention
  is the next chunk). So CPU-path performance ≈ CPU-GEMM performance.

## 3. The options (all captured)

### A. Pure-managed tiled GEMM (CPU) — "MKL magic in the managed world"

Port the **same Row4 register-blocked tiled kernel we already proved on the GPU**
(`samples/Nivara.Samples/Gpu/GemmKernels.cs`) back to CPU with **`Vector512`
intrinsics** + cache blocking + optional `Parallel.For` threading.

- Realistic target: 2–4× over current GEMM → whole forward ~60–90 ms
  (single-threaded) / ~40–60 ms (threaded). Honest upper bound: community
  evaluations of hand-tuned managed GEMM (e.g., the ManagedMatrixMultiply series)
  land ~60% of OpenBLAS on **large** single-threaded GEMMs; DistilBERT's 128-row
  shapes are harder for managed code (per-call launch + small tiles favor native
  tuned kernels) — assume less than 60% MKL there until measured.
- Pros: zero new dependencies; fits the "dependency-free core" philosophy;
  reuses the exact pattern/tiling already validated; benefits every consumer
  (AutoDiff training, SmolLM/Llama/MiniLM samples, non-GPU users).
- Cons: managed can't fully match MKL; threading tuning (thread-pool contention
  in inference servers) is real work; may still land 1.5–2.5× behind PyTorch.

### B. Opt-in native BLAS bridge

P/Invoke into a native BLAS behind a small seam — **OpenBLAS (BSD-3) or Intel
oneMKL (free / redistributable)** — hosted in `Nivara.Extensions` (where the
other third-party deps already live), **not** in core.

- Realistic target: true MKL-class GEMM (the real ~35 ms class).
- Cons: native DLL + per-platform package deployment; contradicts pure-managed
  unless strictly opt-in; licensing tolerances need a decision (both permissive
  enough, but oneMKL deployments are large).

### C. Hybrid — managed default, native opt-in

Managed tiled GEMM stays the default; a provider seam (mirroring the existing
`KernelSelector.DetermineKernelType` dispatch pattern) lets users opt into the
native bridge. Measure both and publish both in the README table.

- Pros: best of both; honest user choice; core stays dependency-free.
- Cons: most work of the three.

### Comparison

| option | target (DistilBERT fwd) | cost | risk |
|---|---|---|---|
| A. managed tiled GEMM | ~40–90 ms | low (pure managed, reuses GPU pattern) | may only reach 1.5–2.5× behind PyTorch |
| B. native bridge | ~35 ms class | medium (native dep, deployment, policy) | philosophy/deployment tax |
| C. hybrid | ~35–90 ms (selectable) | highest (both paths) | most surface, split maintenance |

## 4. Where it plugs in

- **Seam**: GEMM funnels through `LlamaFusedKernels.MatMulTransposedB` +
  `ReverseGradOperations.MatMul`; `KernelSelector` already centralizes kernel
  choice by shape/CPU features — the natural dispatch point for managed-SIMD vs
  (optional) native provider.
- **Gate**: reuse the regression-harness idea from **issue #435** (lasting GEMM
  correctness + perf gate, still open): correctness vs double-precision truth
  (the GPU keystone run used `maxAbs 4.4e-5..1.6e-4` at K=768..3072) **and** a
  GMAC/s table for the CPU kernel side too, not just GPU.
- **Blast radius**: kernel files in `src/Nivara` (core) for A; a new (optional)
  provider project/Extensions surface for B/C. Not touching anything until a
  decision + issues are written.

## 5. Roadmap (draft, not started)

1. **M1 — Managed CPU-GEMM program**: CPU port of the Row4 pattern (Vector512 +
   cache blocking, single-thread first, then threaded); measure % of MKL reached
   on DistilBERT shapes; record honestly in the README table (same-session
   methodology). Highest leverage: benefits all transformer paths + training.
2. **M2 — GPU fusion follow-up** (from PR #436, independent of M1): lazy stream +
   per-op launch fusion to attack the 65 ms → ~25 ms gap (one kernel per op,
   ~100 dispatches/forward, is overhead-bound at 128-row shapes).
3. **M3 — Decide the native bridge only from M1's measured numbers**:
   - managed ≥ ~50% of MKL on these shapes → managed enough; skip native (keep
     pure).
   - managed ~20–30% → hybrid (C) becomes compelling; write B/C as a proper plan.

## 6. Decision gate / open questions

- A vs B vs C — decided *after* M1 measurements, not before.
- Threading policy: `Parallel.For` inside a GEMM in server/concurrent-inference
  contexts (thread-pool contention, oversubscription) — needs a knob.
- Dependency policy: native deps (if any) live in `Nivara.Extensions` /
  optional provider, never core; needs explicit sign-off given the pure-managed
  DNA.
- Benchmark methodology: keep same-session CPU/GPU/PyTorch where possible; AC
  power for iGPU; add a MKL/BLAS column once a bridge exists.

## 7. Reference notes

- **MKL / oneMKL**: Intel Math Kernel Library (oneAPI). Free for commercial use,
  redistributable. PyTorch CPU wheels ship MKL-backed; `torch ...+cpu` in our
  README figures.
- **BCL state (as of .NET 10)**: `TensorPrimitives` is elementwise/aggregate SIMD
  (200+ generic overloads) — **no GEMM in the BCL**; `Tensor<T>` doesn't expose
  matmul. So "MKL-class" needs our own kernel or a bridge.
- **RyuJIT**: explicit `Vector512` intrinsics available (.NET 8+); auto-vectorizes
  simple loops; managed GEMM literature (e.g., ManagedMatrixMultiply) is the
  reference point for what hand-tuned managed alcan hit (~60% of OpenBLAS on
  large matrices).
- **Existing assets to reuse**: `GemmKernels.TiledGemmKernelRow4` (GPU, proven
  303–379 GMAC/s, DistilBERT shapes), `MatMulTransposedB` (current CPU reference),
  `KernelSelector`, issue #435 harness concept, `BufferPool`/`ArrayPool` rules.

## 8. Related open items

- **#435** — promote tiled-GEMM correctness+perf harness into a lasting
  regression gate (probe or sample bench) — covers the gate half of M1.
- **#437** — M2 GPU kernel fusion / lazy stream to cut per-dispatch launch
  overhead (small encoders are launch-bound; MiniLM's GEMM legs are only ~5 ms).
- **PR #436** — DistilBERT GPU first scenario (merged when approved; M2 follow-up
  documented in `docs/BERT-GPU.md` §4.4).
- **PR (next)** — MiniLM GPU (`khurram/minilm-gpu`, retargets to `main` after
  #436 merges): second config-driven model on the shared runner.
- Any decision to *start* M1/M2/M3 should first be recorded as GitHub issues and
  a proper plan (grounding + gates), per the iterative-work conventions.