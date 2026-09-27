# Probe: CPU or GPU for Laya (ModernBERT-large + 2-layer decision head)

## Problem

Phase 2 of the Laya work (`docs/LAYA.md` §9) is blocked on an undecided question: **which
backend should Laya be implemented on?** The question was previously framed as a gate-strategy
question (GPU-vs-CPU vs GPU-vs-PyTorch), which is the wrong framing — the gate follows from the
implementation choice, not the other way round. The objective is simply the best implementation.

The blocker is that the existing measurements **cannot** settle it, and they bracket rather than
answer:

```
DistilBERT, 6L d=768 ffn=3072, S=128  ->  5.44 G MACs
  Nivara CPU    209.7 ms  ->   26 GMAC/s   (end-to-end, all ops)
  Nivara GPU     63.3 ms  ->   86 GMAC/s   (end-to-end, all ops)
  PyTorch CPU    32.8 ms  ->  166 GMAC/s   (end-to-end, all ops)
  GPU GEMM kernel alone (#440)         ->  303-379 GMAC/s

Laya, 28L d=1024 Wi=5248 ffn=2624, S=512  ->  175.7 G MACs  (32x DistilBERT)
```

Two readings point in opposite directions and both are wrong to assume:

1. **The GPU should do much better on Laya than the BERT rows suggest.** The BERT rows achieve
   only 86 GMAC/s end-to-end against a 303-379 GMAC/s isolated kernel, so ~75% of their time is
   attention / LayerNorm / dispatch, not GEMM. Laya's GEMM fraction is far higher (d=1024,
   28 layers, four large GEMMs per layer), so the 1.9x deficit vs PyTorch should mostly
   evaporate. The GEMM arithmetic says GPU ~0.52 s at S=512.
2. **The CPU number is not a hardware wall.** 26 GMAC/s is ~1% of this CPU's theoretical FMA
   peak, so something in the kernel, not the silicon, is responsible. Grounding (below)
   identified the shape of that something, and it is **not** a missing `Parallel.For`.

Neither has been measured at Laya's shapes. **#435** already asks for a lasting GEMM
regression harness and **#440** already identifies GEMM throughput as the GPU bottleneck, so
this probe extends existing, gated harnesses rather than inventing new ones.

## Grounding corrections (G1)

Grounding against MS Learn and the source invalidated three things this plan originally
assumed. They are recorded here rather than silently dropped, because two of them change what
the probe measures.

**1. `TensorPrimitives.Dot` is not a GEMM, and no BCL matrix multiply exists.**
`Dot<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y) → T` is the *vector* dot product. There are zero
`TensorPrimitives.MatMul` call sites in the repo, and `TensorsHelper.cs:30-36` already records
why: `Tensor.MatrixMultiply` has not shipped (upstream `dotnet/runtime#95863`). The original
plan proposed BCL `TensorPrimitives.Dot` as the CPU's "optimized reference" leg; that would
have measured a vector reduction, not a GEMM. Leg removed.

**2. #456 is stale — the CPU GEMM already has a `Parallel.For`.**
#456 asserts `GradKernels.MatMulTransposedB` is "a SIMD row kernel with **no** `Parallel.For`".
Today it is a one-line delegation (`GradKernels.cs:748`) to
`TensorsHelper.MultiplyCore(..., bTransposed: true)`, which reaches `Parallel.For(0, aRows, ...)`
at `TensorsHelper.cs:191`, gated by `ShouldParallelize` (`:312`,
`aRows >= 4 && aRows*aCols*bCols >= 2<<20`). The gate opens at every shape in this plan — the
smallest, MiniLM 128×384×1536, is 7.6e7 MACs against a 2.1e6 threshold. So "26 GMAC/s is a
missing `Parallel.For`" is false, and the plan's fix-the-CPU-then-compare framing collapses.
#456 is being closed with a comment carrying the evidence and the line refs it got wrong.

**3. The real CPU suspect is the kernel's shape, not its threading.**
`MultiplyRowFloat` computes every output element as its *own* `TensorPrimitives.Dot` over
`aCols` floats (`TensorsHelper.cs:293`). A 512×1024 @ 1024×3072 GEMM therefore runs
512×3072 = **1.57M independent horizontal reductions of length 1024**, each paying reduction
latency and flushing the accumulator, instead of holding an output tile in registers across the
K loop. That is a plausible cause of ~1%-of-peak throughput, and it is a *different* fix than
the one #456 asked for.

**Consequence.** The decision rule below no longer compares "fixed CPU vs unfixed GPU". The CPU
ceiling does not exist anywhere in-tree, so the probe has to supply it — which is why leg 3
writes a register-accumulator GEMM rather than calling a BCL API.

**Also cut on grounding.** The attention leg was dropped: at S=512 attention is 8.4M MACs per
layer against 6.3 G per layer for the four GEMMs, i.e. **~0.13%** of the work. It cannot
change the decision. Its one real hazard (NaN from `Exp(s - max)` on a fully-masked row, where
`max == -inf`) is filed against #448 now, without needing a benchmark to justify it.

## What this probe is NOT

It does not implement Laya. It does not build a GPU runner. It does not choose a gate
strategy. It produces three measurements and one decision. Laya itself is the next branch.

## Proposed changes

All of it lands in `tests/Nivara.PerformanceTests`, which the probe lifecycle designates for
"general throughput, allocation, and memory A/B work" and which already references
`Nivara.Samples` (hence `IlgpuRuntime` + `GemmKernels` via `InternalsVisibleTo`) and ILGPU
1.5.3.

**Lifecycle deviation, stated deliberately.** The probe lifecycle asks for a temp-dir
standalone project first. I am implementing in-repo instead, because: `GemmBenchmark.cs` *is*
the lasting GEMM regression harness #435 requests, so extending it satisfies an open issue
rather than creating a second harness; a temp project would have to duplicate the ILGPU
bootstrap and would therefore not measure the real runtime's characteristics; and the shape
additions are reusable under **either** probe outcome, so there is no waste risk from writing
them before the answer is known.

### 1. Allocation ceiling (binary gate)

New mode `--gpu-alloc`. The Laya checkpoint is F16 but `--gpu` is F32-only, so inference needs
`421,293,830 * 4 B = 1.69 GB` of F32 device buffers — against 66-110M params for every existing
GPU row. Allocate a stepped series (256 MB / 512 MB / 1 GB / 1.5 GB / 1.69 GB / 2 GB / 3 GB),
timing a fill of each, and report the largest that succeeds plus the device's reported global
memory. This is the one true binary gate: if 1.69 GB does not allocate, GPU is out regardless of
throughput and F16 device buffers become a prerequisite rather than a later optimisation.

### 2. GEMM at Laya's shapes, both backends

Extend `GemmBenchmark.s_shapes` with Laya's real per-layer shapes, keeping the existing
DistilBERT/MiniLM rows as controls (they are the only rows with published end-to-end numbers,
so they are what let us sanity-check the probe against history):

| name | M (rows) | K (aCols) | N (bCols) | source |
|---|---|---|---|---|
| `laya qkv` | 512 | 1024 | 3072 | `encoder.layers.N.attn.Wqkv` |
| `laya attn out` | 512 | 1024 | 1024 | `attn.Wo` |
| `laya fc1 (Wi)` | 512 | 1024 | 5248 | `mlp.Wi` (fused input‖gate) |
| `laya fc2 (Wo)` | 512 | 2624 | 1024 | `mlp.Wo` |
| `laya head ff1` | 512 | 1024 | 4096 | `head.layers.N.linear1` |
| `laya head ff2` | 512 | 4096 | 1024 | `head.layers.N.linear2` |
| `laya act 1` | 512 | 1028 | 256 | `act_head.0` |
| `laya scorer 1` | 8 | 1024 | 1024 | `scorer.1` (k options) |
| `laya scorer 2` | 8 | 1024 | 1 | `scorer.3` (thin — GEMV-shaped) |
| `laya qkv@128` | 128 | 1024 | 3072 | for comparison with the S=128 rows |

The last two are deliberately thin/skinny: at batch 1 they are weight-bandwidth-bound, not
compute-bound, and a GEMM benchmark that only measures fat shapes would miss that.

### 3. The CPU's achievable ceiling (the decisive leg)

New mode `--cpu-gemm`. At each Laya shape, time three CPU legs and report GMAC/s for each.
The first two are in-tree paths; the third is written by this probe, because as grounding
established, no in-tree path is a properly blocked GEMM.

| leg | what it is | what it tells us |
|---|---|---|
| `GradKernels.MatMulTransposedB` | the AutoDiff path Laya would actually call (`bTransposed: true`) | today's real CPU cost |
| `TensorsHelper.MatMul` | the same kernel with `bTransposed: false` | isolates the cost of the per-call B transpose |
| probe-local `Parallel.For` + `Vector<float>` | output tiles, register accumulators held across the K loop, A/B spans shared with no `RentCopy` | the CPU's **achievable** ceiling |

The third leg is the whole point of the probe, and it is ~20 lines: partition output rows
across workers, and for each output row walk K in `Vector<float>`-width steps accumulating into
register-resident vectors, storing only at the end. It is the shape of the fix that finding 3
implies, written where it measures a number instead of shipping a change to `src/Nivara`.

Two costs the in-tree legs carry that the reference leg does not, and which must be reported
separately because they are not GEMM work:

- `RentCopy(a)` — a full copy of A whenever `ShouldParallelize` opens. At M=512/K=1024 that is
  2 MB copied per call, for no reason: output rows are disjoint and A/B are read-only.
- the pooled `bT` staging buffer — `b.CopyTo` when `bTransposed`, `Transpose` otherwise. For
  `laya fc1` (Wi) that is 1024×5248 = 5.4M elements staged **per call**.

Whether to time steady-state (prep hoisted, as a real inference loop would do after one
transpose at load) or per-call (prep included, as the code does today) is reported for both,
because the honest end-to-end number depends on how often weights are reused. At batch 1 that
distinction is the difference between a memory-bound loop and a compute-bound one.

### 4. Record the evidence, then decide

Write the results into `samples/NivaraInference/README.md` (the canonical perf document) as a
new probe section, and the decision + its reasoning into `docs/LAYA.md` §9. Comment the measured
numbers back onto **#435** and **#440** so those open issues carry current evidence rather than
September estimates.

## Decision rule (fixed before the probe runs, so it cannot be rationalised afterwards)

Compare **GPU-as-it-is** against the **probe's register-accumulator CPU ceiling**:

- **GPU wins** -> the decision is **safe**, because #440's tile-32/2x2 is a 2-3x *speedup* and
  can only reinforce it. Laya goes GPU-only.
- **CPU wins** -> the decision is **not** safe, because GPU's cheap fix has not been written.
  Either implement #440 and re-probe, or choose CPU. Do not conclude "CPU" from a probe that
  measured an unfixed GPU against a fixed CPU.
- **`--gpu-alloc` fails at 1.69 GB** -> GPU is out on memory before throughput is even
  considered, and F16 device buffers become a prerequisite rather than a later optimisation.

This asymmetry is the whole reason the rule is written down first. Note the rule survived
grounding even though its justification did not: the CPU side is no longer "the fix that
already exists in-tree" but "the fix the probe writes", which is a strictly *stronger* CPU
showing and therefore a *conservative* test of the GPU. The direction of the bias is unchanged
and still favours concluding GPU.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` — builds clean.
2. `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm` — the **existing**
   gate must still pass, with the DistilBERT/MiniLM rows unchanged. This is the regression
   guardrail: if adding shapes perturbs existing numbers, the probe is measuring something else.
3. `dotnet run ... -- --gpu-alloc`, `--cpu-gemm` — the new modes, on **AC power only**.
   `GemmBenchmark` already prints AC/battery state via `GetSystemPowerStatus`; both new modes
   must do the same, because the 2026-09-27 retake established that battery contaminates the
   iGPU by 6-30%.
4. Each CPU leg must be gated against the same host double-precision truth the GPU legs use, so
   a fast-but-wrong reference leg cannot win the comparison. Max abs ≤ 1e-3, as `--gemm` does.
5. **Ask the human before running any `dotnet test`.**
6. Cross-check the probe against history: the DistilBERT shapes must reproduce the published
   303-379 GMAC/s band for the GPU. If they do not, the probe is wrong before its answer is.

## Blast radius

- `tests/Nivara.PerformanceTests/GemmBenchmark.cs` — **additive only**: new rows in
  `s_shapes`, no change to `GateMaxAbs`, the DP-truth gate, the kernel list, or the timing loop.
  The existing `--gemm` invocation must behave identically.
- **New files**: `CpuGemmProbe.cs`, `GpuAllocProbe.cs` + dispatch lines in `Program.cs` +
  `README.md` mode docs. All new modes are on-demand and excluded from the default scenario
  suite and the `--json`/`--compare` gate, exactly as `--gemm` is. `--gpu-alloc` is
  GPU-dependent; `--cpu-gemm` is not, and can run anywhere.
- `src/Nivara/**` — **untouched.** This probe measures; it does not fix. #440 stays open and
  unmodified; this branch supplies the evidence that decides whether to prioritise it.
- **#456 is closed** (human decision) with a comment carrying the evidence and the correction.
  Consequence to accept: after this, the ~6.1-6.5x Nivara-CPU-vs-PyTorch gap has **no open
  issue** tracking it. The closing comment points at the real remaining suspect, and the
  probe's leg-3 numbers will quantify it; whether to file a follow-up issue for the
  horizontal-reduction structure is a follow-up decision, deliberately not taken here.
- `samples/NivaraInference/README.md`, `docs/LAYA.md` — documentation only.
- No public API change, no library behaviour change, no dependency added.
- The only way this breaks existing behaviour is if the added shapes change `--gemm`'s reported
  numbers, which step 2 above exists to catch.

## Planned commits

1. `docs: plan the Laya CPU/GPU probe in TODO.md` (this file)
2. `docs: correct the probe plan after grounding` (this amendment — G1)
3. `perf: add Laya GEMM shapes to the #435 regression gate`
4. `perf: add --gpu-alloc to test the 1.69 GB Laya working set`
5. `perf: add --cpu-gemm to measure the CPU ceiling and the GEMM structure cost`
6. `docs: record the probe evidence and the backend decision`

## GitHub issues log

- [x] #435 — lasting GEMM regression harness (this probe *is* that harness; will be satisfied)
- [ ] #440 — iGPU GEMM throughput, tile-32/2x2 (evidence pending; not fixed here)
- [x] #456 — **closed**: the `Parallel.For` its body claims is missing already exists at
  `TensorsHelper.cs:191`; its line refs (`ReverseGradOperations.cs:431`,
  `TensorsHelper.cs:158`) are stale
- [ ] #448 — non-finite probabilities masquerading as model quality. `AttentionKernels`
  computes `Exp(s - max)` with `s = max = -inf` on a fully-masked row and has no guard, unlike
  the CPU `GradKernels` (`if (max == T.NegativeInfinity)`). Unreachable for Laya (`[CLS]` at
  position 0 is never masked) but a real hazard for any future model. Filing on the hazard
  alone, without needing the benchmark that was cut.
- [ ] #449 — GPU path for ModernBERT (depends on the decision; stays open either way)
- [ ] #437 — M2 kernel fusion, already shipped (regression check only)

Create any new issue immediately on discovery via
`gh issue create --repo khurram-uworx/Nivara` and record the number above — do not rely on
memory, as compaction during execution can lose it.
