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
   peak. `GradKernels.MatMulTransposedB` has no `Parallel.For` while `TensorsHelper.MatMul:191`
   does — that is issue **#456**, not silicon. If BCL `TensorPrimitives.Dot` reaches
   300-500 GMAC/s at these shapes, the CPU is very much alive and the 8x reading collapses.

Neither has been measured at Laya's shapes. **#435** already asks for a lasting GEMM
regression harness and **#440** already identifies GEMM throughput as the GPU bottleneck, so
this probe extends existing, gated harnesses rather than inventing new ones.

## What this probe is NOT

It does not implement Laya. It does not build a GPU runner. It does not choose a gate
strategy. It produces four measurements and one decision. Laya itself is the next branch.

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

New mode `--cpu-gemm`. At each Laya shape, time four CPU legs and report GMAC/s for each:

| leg | what it is | why it is in the probe |
|---|---|---|
| `GradKernels.MatMulTransposedB` | the AutoDiff GEMM Laya would actually call | today's CPU cost; single-threaded — **#456** |
| `ReverseGradOperations.MatMulTransposedB` | the autograd wrapper | isolates graph-node/dispatch overhead from the kernel |
| `TensorsHelper.MatMul` | has `ShouldParallelize` + `Parallel.For` | the CPU's "cheap fix" already in-tree |
| `TensorPrimitives.Dot` | BCL, multi-threaded + SIMD | the realistic CPU ceiling |

This answers the question nobody has asked: **how much of the CPU's 26 GMAC/s is the missing
`Parallel.For`?** Note `TensorsHelper.MatMul` `RentCopy`s all of A when parallel, so at
M=512/K=1024 it copies 2 MB per call — worth reporting, since it is a cost a `Parallel.For`
port of #456 would not pay.

### 4. Attention ceiling at S=512

New mode `--gpu-attn`, reusing `AttentionKernels.BatchedAttention` as-is. It recomputes every
score **three times** and materialises a `[B,H,S,S]` F32 buffer (16 heads x 512 x 512 x 4 B =
**16.8 MB per layer**) — both artifacts of mirroring the CPU kernel's accumulation order. Time
it at S=512 for band=-1 (global) and, for comparison, the arithmetic cost of a banded rewrite
that skips out-of-band keys:

```
naive:   28 x 512 x 3              = 43,008 score-units
banded:  (10 x 512 + 18 x 129) x 1 =  7,442 score-units   -> ~5.8x less
```

(10 global + 18 sliding from `global_attn_every_n_layers = 3`; band half-width 64 from
`local_attention = 128`.) That ratio is arithmetic, not measurement, and the probe must report
it as such. Also record whether the kernel returns `NaN` on a fully-masked row — it computes
`Exp(s - max)` with `s = max = -inf`, so it should, and the CPU `GradKernels` already guards
this (`if (max == T.NegativeInfinity)`). #448 is exactly "a NaN masquerading as a bad model", so
this is recorded whether or not Laya can reach that state.

### 5. Record the evidence, then decide

Write the results into `samples/NivaraInference/README.md` (the canonical perf document) as a
new probe section, and the decision + its reasoning into `docs/LAYA.md` §9. Comment the measured
numbers back onto **#435**, **#440** and **#456** so the three open issues carry current
evidence rather than September estimates.

## Decision rule (fixed before the probe runs, so it cannot be rationalised afterwards)

Compare **GPU-as-it-is** against **CPU-with-its-cheap-fix-already-in-tree** (`TensorsHelper.MatMul`
and `TensorPrimitives.Dot` — both exist today, so this needs no new code):

- **GPU wins** -> the decision is **safe**, because #440's tile-32/2x2 is a 2-3x *speedup* and
  can only reinforce it. Laya goes GPU-only.
- **CPU wins** -> the decision is **not** safe, because GPU's cheap fix has not been written.
  Either implement #440 and re-probe, or choose CPU. Do not conclude "CPU" from a probe that
  measured an unfixed GPU against a fixed CPU.

This asymmetry is the whole reason the rule is written down first.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` — builds clean.
2. `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm` — the **existing**
   gate must still pass, with the DistilBERT/MiniLM rows unchanged. This is the regression
   guardrail: if adding shapes perturbs existing numbers, the probe is measuring something else.
3. `dotnet run ... -- --gpu-alloc`, `--cpu-gemm`, `--gpu-attn` — the new modes, on **AC power
   only**. `GemmBenchmark` already prints AC/battery state via `GetSystemPowerStatus`; all
   three new modes must do the same, because the 2026-09-27 retake established that battery
   contaminates the iGPU by 6-30%.
4. **Ask the human before running any `dotnet test`.**
5. Cross-check the probe against history: the DistilBERT shapes must reproduce the published
   303-379 GMAC/s band for the GPU. If they do not, the probe is wrong before its answer is.

## Blast radius

- `tests/Nivara.PerformanceTests/GemmBenchmark.cs` — **additive only**: new rows in
  `s_shapes`, no change to `GateMaxAbs`, the DP-truth gate, the kernel list, or the timing loop.
  The existing `--gemm` invocation must behave identically.
- **New files**: `CpuGemmProbe.cs`, `GpuAllocProbe.cs`, `GpuAttentionProbe.cs` + dispatch lines
  in `Program.cs` + `README.md` mode docs. All new modes are on-demand, GPU-dependent, and
  excluded from the default scenario suite and the `--json`/`--compare` gate, exactly as
  `--gemm` is.
- `src/Nivara/**` — **untouched.** This probe measures; it does not fix. #456 and #440 stay open
  and unmodified; this branch only supplies the evidence that decides whether to prioritise them.
- `samples/NivaraInference/README.md`, `docs/LAYA.md` — documentation only.
- No public API change, no library behaviour change, no dependency added.
- The only way this breaks existing behaviour is if the added shapes change `--gemm`'s reported
  numbers, which step 2 above exists to catch.

## Planned commits

1. `docs: plan the Laya CPU/GPU probe in TODO.md` (this file)
2. `perf: add Laya GEMM shapes to the #435 regression gate`
3. `perf: add --cpu-gemm to measure the CPU ceiling behind #456`
4. `perf: add --gpu-alloc to test the 1.69 GB Laya working set`
5. `perf: add --gpu-attn to establish the attention ceiling at S=512`
6. `docs: record the probe evidence and the backend decision`

## GitHub issues log

- [ ] #435 — lasting GEMM regression harness (this probe *is* that harness; will be satisfied)
- [ ] #440 — iGPU GEMM throughput, tile-32/2x2 (evidence pending; not fixed here)
- [ ] #456 — AutoDiff `MatMulTransposedB` has no `Parallel.For` (evidence pending; not fixed here)
- [ ] #448 — non-finite probabilities masquerading as model quality (`--gpu-attn` NaN row)
- [ ] #449 — GPU path for ModernBERT (depends on the decision; stays open either way)
- [ ] #437 — M2 kernel fusion, already shipped (regression check only)

Create any new issue immediately on discovery via
`gh issue create --repo khurram-uworx/Nivara` and record the number above — do not rely on
memory, as compaction during execution can lose it.
