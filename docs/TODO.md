# #447 — Banded/sparse attention kernel for the GPU encoder path

Branch `khurram/447`, off `main` at `c42bfac`.

## Problem

`AttentionKernels.BatchedAttention` (`samples/Nivara.Samples/Gpu/AttentionKernels.cs:29-112`)
carries the ModernBERT sliding-window band as *mask data* rather than as a control-flow
restriction. It has three unconditional `for (int j = 0; j < seqLen; j++)` passes, and `band`
reaches only `Keep()` (`:126-127`), which overwrites an already-computed score with `-inf`:

```csharp
static bool Keep(ArrayView<float> mask, int maskBase, int qPos, int j, int band)
    => mask[maskBase + j] >= 0.5f && (band < 0 || XMath.Abs(qPos - j) <= band);
```

The score for an out-of-band key is computed, then discarded. **No choice of `band` changes the
kernel's iteration count**, so the sliding window buys no arithmetic saving on the GPU path.

`band` is already plumbed per layer (`Gpu/ModernBertGpuRunner.cs:408`, `Band = config.Band(index)`),
so nothing needs to change in the signature or at any call site — the kernel simply ignores it.

### Why this matters now (measured, AC power, this machine)

`--gemm-legs` on an **Intel Iris Xe** iGPU (edge class) with .NET 11.0.0-rc.1, F32, AC line,
ILGPU/OpenCL, 512 threads/group, 65536 B shared/group:

| Laya-large leg | Arc 140T (issue comment) | **Iris Xe (this branch)** |
|---|---|---|
| **BatchedAttention** | **57.0%** | **57.8%** |
| GEMM (4 shapes) | 39.2% | 40.6% |
| total ms/fwd | 2221 | 17110 |

ModernBERT base: BatchedAttention **60.3%**, GEMM 38.2%, 12479.83 ms/fwd.

The share reproduced within 0.8pp across two GPU classes, a ~7.7x difference in absolute
throughput, and two power states (battery 57.1% / AC 57.8%). That is a structural property of the
kernel, not a device artifact. The mechanism is consistent with the code: GEMM is bandwidth-bound
while this kernel is latency-bound, which is why it dominates *more* on a lower-bandwidth edge part.

### The CPU case is closed — do not use it to justify this work

`modernbert benchmark` (AC, same session, F32): seq 128 median 4924.1 ms (175.86 ms/layer),
seq 256 median 6960.5 ms (248.59 ms/layer) → **1.41x for a 2x length increase**, sub-quadratic.

Arithmetic for the measured config (28 layers, d=1024, ffn=2624, 10 full + 18 sliding, band 64):

| L | GEMM | attn (dense) | attn share of attn+gemm |
|---|---|---|---|
| 128 | 43.92 GMAC | 0.94 GMAC | 2.1% |
| 256 | 87.85 GMAC | 3.76 GMAC | 4.1% |

Even making all 18 sliding layers infinitely cheap saves **0.9%** of attn+gemm MACs. That is why the
measured ratio is 1.41x and not the 1.87x the issue originally claimed (its author has since
retracted 1.87x/1.43x as superseded). **The CPU perf rationale is not sound; the GPU one is.**

The 2048 cap (`ModernBertMasks.MaxDenseLength`) is a real but **CPU-only** limit — the GPU path
already reaches 4096. Lifting it would mean adding a `band` parameter to the public
`MultiHeadAttention` (`src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:496`). Out of scope
here; filed separately.

## Proposed change

### 1. Kernel — `samples/Nivara.Samples/Gpu/AttentionKernels.cs`

Compute the row's key range once, then use it in all three passes:

```csharp
int jFirst = band < 0 ? 0 : Math.Max(0, qPos - band);
int jLast  = band < 0 ? seqLen - 1 : Math.Min(seqLen - 1, qPos + band);
```

Retarget each of the three loops to `for (int j = jFirst; j <= jLast; j++)`. `Keep()` still applies
the padding mask unchanged, so the mask contract is untouched.

**The fully-masked-row guarantee needs no new code.** `Gpu/AttentionKernels.cs:70-75` already returns
zeros when `max == float.NegativeInfinity`. An empty band range means the first pass never runs,
`max` stays `-inf`, and that existing guard fires. This is why #454 does not gate this work: the
guard is structural, not incidental. What *does* change is reachability — narrowing makes
fully-masked rows more common — so it must be covered by a test, not merely reasoned about.

### 2. Hard constraint: exactly one shared-memory allocation per kernel

`tests/Nivara.Tests/Gpu/SharedMemoryAllocationTests.cs` scans compiled IL and fails any kernel body
with more than one `SharedMemory.Allocate*` call. It exists because of **#468**, an ILGPU OpenCL
lowering bug that landed in `c42bfac`: two allocations with disagreeing extents produce *plausible
wrong numbers* rather than a compile error. Do **not** stage the score row in shared memory — that
option is ruled out by the gate. Banding needs no second allocation.

### 3. Tests — new `tests/Nivara.Tests/Gpu/GpuAttentionBandTests.cs`

| gate | comparison | bound |
|---|---|---|
| regression | `band = -1`, before vs after | **exact** |
| banding | banded vs dense, all-valid mask | **exact** |
| CPU parity | banded GPU vs `MultiHeadAttention` + `ModernBertMasks.Build` | tolerance |

Exactness is justified, not aspirational: an out-of-band term contributes
`XMath.Exp(-inf - max) = 0` and `0 * v = ±0`, neither of which can perturb an ascending float sum,
and the surviving terms keep their original ascending order. Per AGENTS.md, reproduce the reference
exactly rather than with a tolerance band.

The CPU-parity gate needs a tolerance because `RowScore`'s scalar `acc +=` is not an FMA while
`MatMulTransposedB` may be — the same split documented in `GpuElementwiseParityTests`
(maxRel 3.2e-6). The two exact gates isolate the band change so a tolerance can never mask a band
bug. Cases: empty band row (zeros, not NaN), `band = -1` byte-identical, band smaller than
`headDim`, a band that clips at both sequence ends, and multi-head/multi-batch.

**Gap being closed:** there is currently no automated test that runs the GPU `BatchedAttention`
kernel. `GpuElementwiseParityTests` calls only host-side scalar helpers; `SharedMemoryAllocationTests`
reads IL. The only real gate today is end-to-end `modernbert --gpu compare`, which needs weights and
a GPU. This adds the first. It must skip cleanly where no OpenCL device exists, with the skip
counted in a **separate counter** from pass/fail so an unhostable run is never reported as numeric.

### 4. Docs

- `docs/ACCELERATION.md:155` and `:456` — the band now reduces work; record the measured saving.
- `docs/MODERNBERT.md:17` — same.
- Correct four inaccuracies carried in the issue's own text (the issue body is left alone; the repo
  docs are the durable record):
  1. The dense masks are **two, built once per forward** (`ModernBertModel.cs:435-436`), not one
     "per layer, per head group". At L=8192 that is **537 MB**, not 268 MB.
  2. The issue's option 1 ("score masking only") does **not** lift the 2048 cap, because the mask
     is still materialized. Only the blocked/true-banded options do.
  3. `modernbert benchmark --seq 256` is not a real command. CPU benchmarks hardcode `{128, 256}`
     (`samples/NivaraInference/ModernBert.cs:168`); GPU hardcodes `{128, 512, 2048, 4096}` (`:589`).
  4. The L^2 mask-size table describes the CPU path only; the GPU never materializes `[L, L]`.

### 5. Correct the record on this machine's OpenCL

An earlier claim in this session — that the GPU path could not run here because
`HKLM:\SOFTWARE\Khronos\OpenCL\Vendors` is absent — was **wrong**. That registry key is not how the
ICD resolves on this box. `IlgpuRuntime.SelectGpu` (`samples/Nivara.Samples/Gpu/IlgpuRuntime.cs:57-69`)
matches on `"Intel"` + `CL_DEVICE_TYPE_GPU`, **not** `"Arc"`, so the "Intel Arc iGPU" wording in the
`UNBUILT` messages is cosmetic and those paths never fire here. The `Nivara.GpuProbe` ILGPU leg
passes 3/3 kernel gates on the Iris Xe.

## Verification

1. `dotnet build Nivara.slnx -c Release`
2. New `GpuAttentionBandTests` (ask the human before `dotnet test`).
3. Existing GPU guards stay green: `SharedMemoryAllocationTests`, `GpuElementwiseParityTests`.
4. `modernbert --gpu compare` — end-to-end HF parity, unchanged (fixture bound 1.62e-5 max abs diff,
   cosine 1.0000000000, including the padding region).
5. `--gemm-legs` on **AC** against this machine's baseline: BatchedAttention 57.8%, GEMM 40.6%,
   17110 ms/fwd. Expect the attention share to drop and the ratio (BatchedAttention / GEMM) to fall.

### Expected result, and what a miss would mean

Projected from iteration counts (10 full + 18 sliding, S=512, band 129/512):
`10*512 + 18*512 = 14336` → `10*512 + 18*129 = 7442` units, a **48% cut in attention
iterations**; at a 57.8% attention share that is **~28% end-to-end**, rising to ~45% at S=2048.

This is arithmetic assuming cost is linear in iteration count. The three passes are structurally
identical loops, which supports the assumption, but it is an assumption. **If the measured result
lands materially off, the linearity assumption is what is wrong** — report that rather than moving
the target.

## Planned commits

1. `docs: plan the #447 banded GPU attention kernel in TODO.md`
2. `perf(gpu): make the attention band restrict iteration, not just the score`
3. `test(gpu): pin the banded attention kernel against the dense path`
4. `docs: record the banded attention result and correct the dense-mask figures`

## Blast radius

- **Changed:** `samples/Nivara.Samples/Gpu/AttentionKernels.cs` — `BatchedAttention` only. Same
  signature, no caller changes.
- **Callers (unaffected, verified):** `samples/Nivara.Samples/Gpu/ModernBertGpuRunner.cs:126`
  (`acc.LoadKernel<...>(AttentionKernels.BatchedAttention)`), and by extension
  `ModernBert.RunGpu/BenchmarkGpu/CompareGpu`. Also used by the DistilBERT and MiniLM GPU runners,
  which pass `GlobalAttentionBand` — so `band < 0` must stay bit-identical or those regress.
- **Tests touched:** new file only; two existing GPU guards must stay green.
- **Public API:** none. `AttentionKernels` is `internal static` in `Nivara.Samples`.
- **Not touched:** the CPU AutoDiff path, the dense mask, `MaxDenseLength`, and the #447 issue body.

## GitHub issues log

- [x] #447 — banded/sparse attention kernel (this branch)
- [x] #450 — fused pre-norm block, explicitly blocked on #447
- [x] #454 — fully-masked row under `finfo.min`; **not** a blocker here (the `max == -inf` guard
      already covers the GPU path), but narrowed bands raise its reachability, so a test pins it
- [ ] lift the CPU 2048 cap — CPU-only feasibility follow-up; needs a `band` parameter on the
      public `MultiHeadAttention`. Filed during execution.
