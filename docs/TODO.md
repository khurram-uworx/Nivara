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

**The planned "banded vs dense, all-valid mask, exact" gate is not expressible, and the reason
matters.** The kernel's mask is indexed `mask[b * seqLen + j]` with no query dimension, so a
per-query band cannot be encoded there at all — which is precisely why `band` exists as a
parameter. So the "before" kernel is not reproducible in one build, and comparing a `band = 2`
run against a `band = -1` run asserts something false (`band = -1` means attend to everything).

**The stronger property, established by mutation rather than by assertion: the outputs are
invariant under the loop-bound change.** `Keep` already suppressed every out-of-band key, so an
out-of-band key contributed `exp(-inf - max) = 0` and then `0 * v = ±0` to an accumulator that
starts at `+0`, and round-to-nearest leaves `+0` alone. Reverting the kernel to the pre-#447 dense
sweep leaves every test green — there is no observable output difference to find.

That makes the change purely a performance change, and it dictates the gate design: the **only**
way the new bounds can be wrong is by **under-reaching** (dropping a key that should have been
kept), so that is what the fixture attacks. Four gates, each confirmed to fail on its own mutation:

| gate | band | bound | fails when |
|---|---|---|---|
| `band = 0` == the V row (closed form) | 0 | **exact** | bounds under-reach by one |
| fully-masked row is zeros, not NaN | 1 + padding | **exact** | the `max == -inf` guard is removed |
| vs CPU `MultiHeadAttention` + `ModernBertMasks.Build` | 2, and 1 | 1e-5 | bounds under-reach |
| `band = -1` vs a band spanning the sequence | -1 / 8 | **exact** | nothing (bound fidelity only) |

The `band = 0` gate is exact because a single visible key collapses the softmax to `p = 1`, so the
output row *is* `V[b, qPos]` — a closed form needing no reference implementation. The CPU gates
carry a tolerance because `RowScore`'s scalar `acc +=` is not an FMA while the CPU dot product may
be (the split `GpuElementwiseParityTests` documents). The exact gates have been shown to fail, so
the tolerant ones are not the only thing standing between a band defect and a green run.

**Gap being closed:** there is currently no automated test that runs the GPU `BatchedAttention`
kernel. `GpuElementwiseParityTests` calls only host-side scalar helpers; `SharedMemoryAllocationTests`
reads IL. The only real gate today is end-to-end `modernbert --gpu compare`, which needs weights and
a GPU. This adds the first. It skips cleanly where no OpenCL device exists, with the skip counted
in a **separate counter** from pass/fail so an unhostable run is never reported as numeric.

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

**Correction to step 5, found while executing: `--gemm-legs` cannot measure this change.**
`GemmLegBenchmark.cs:166-168` deliberately pins `band = GpuBuffers.GlobalAttentionBand` (-1),
documented as "the more expensive of ModernBERT's 3:1 global:sliding layer mix, so the attention
leg here is the pessimistic one." A global-attention leg is bit-identical before and after #447, so
the 57.8% baseline is expected to reproduce **unchanged** and would say nothing about this work.
Measuring the change needs the real model, which carries per-layer bands
(`ModernBertGpuRunner.cs:408`), so the A/B is `modernbert --gpu benchmark` on this branch against
the same build with the dense sweep restored.

Arithmetic that shapes what to expect at each length (band 64 → a 129-wide window):

- **S=128 must show no change at all** — 129 ≥ 128, so the band spans the whole sequence. A free
  control row: any movement there is measurement noise, not signal.
- S=512 → 129/512 = 25% of keys kept, so a 75% cut on the 18 sliding layers.
- S=2048 → 129/2048 = 6.3% of keys kept, a 93.7% cut.

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
      already covers the GPU path), but narrowed bands raise its reachability, so a test pins it —
      and that test is the only one that fails when the guard is removed
- [x] #473 — lift the CPU 2048 cap. CPU-only feasibility: attention is 2.1%/4.1% of MACs at
      L=128/256, so a banded CPU kernel saves ~0.9%, not time. What it buys is the ability to run
      8192-token context at all, and not allocating 537 MB of mask. Needs a `band` parameter on
      the public `MultiHeadAttention`. Created while executing.
- [x] #474 — `modernbert benchmark` has no `--seq` flag. Blocked the A/B in step 5: the GPU path
      hardcodes `{128, 512, 2048, 4096}` (`ModernBert.cs:589`) and the 4096 row is ~1000x the
      attention work of the 128 row, so one row of that table cannot be measured without paying for
      the rest. Created while executing.
