# Plan — #449 GPU (ILGPU/OpenCL) path for ModernBERT

Branch: `khurram/449` (off `main` @ `6daed80`)

## Problem

`modernbert --gpu` does not exist. `BertEncoderGpuRunner` is hard-wired to post-LN
BERT with a classification head, and the four primitives ModernBERT needs are
missing or unreachable on the GPU path:

1. **No RoPE kernel exists at all.** Every GPU encoder so far is a bidirectional
   BERT with learned position embeddings, so rotary was never needed.
   `ElementwiseKernels` (9 kernels) has no `Rotary`.
2. **No GeGLU.** `GemmKernels.TiledGemmKernelRow4Gelu` folds a *plain* GELU into the
   GEMM epilogue, which cannot gate.
3. **Post-norm fused.** `LayerNormResidual1D` is post-norm and not reusable; the
   pre-norm restructure is composable from `LayerNorm1D` + `Add` but unbuilt.
4. **No band parameter.** `AttentionKernels.BatchedAttention` takes a `[B, S]` 0/1
   *padding* mask only, so the 18 sliding-window layers have no distance limit.

## Corrections to the issue as written

Established during planning against the source; each changes the work.

- **The gate bound must be `1e-3`, not `~1e-5`.** `docs/BERT-GPU.md` §3.6 records
  the measured F32 floor: two valid reduction orders disagree at
  ~4.4e-5…1.6e-4 for K=768…3072, "so a `1e-6`-class gate is impossible — the
  `1e-3·(1+|x|)` bound is right". ModernBERT's GEMM K values are 1024 (QKV/O/Wi) and
  2624 (Wo) — inside that range. The issue's 1.62e-5 is the CPU-vs-PyTorch **max
  abs** figure, not a maxRel target; the GPU introduces a *third* reduction order.
  `Program.cs:19` already sets `GateRelTol = 1e-3` for exactly this reason.
- **The band makes an all-`NaN` bug reachable.** `ModernBertMasks` intersects the
  band with padding, so sliding layers leave query rows past
  `valid_length + 64` with no visible key. CPU handles this at
  `GradKernels.cs:487` (`if (max == T.NegativeInfinity) { clear; return; }`).
  `AttentionKernels.BatchedAttention` has **no such clamp** — grep for
  `IsFinite|NaN|IsNaN` across `samples/Nivara.Samples/Gpu/` returns zero hits. With
  `max = -inf`, `s - max` is `NaN` and the whole output goes NaN from that layer
  on. This is the failure #448 documents; `docs/LAYA.md:667-670` already flags it
  as this kernel's live hazard. **Hard prerequisite, not mentioned in the issue.**
- **Items 2b and 5 are unnecessary.** `UploadTransposed` already transposes any
  pre-fused `[out, in]` weight, and ModernBERT ships `Wqkv` as `[3072, 1024]` and
  `Wi` as `[5248, 1024]` already fused. No `UploadGateUpConcat`, no fused-QKV
  upload helper.
- **The "attention will dominate" premise is stale.** `docs/LAYA.md:681-683`
  (merged in PR #459, four hours after the issue was filed) measured GEMM at
  ~99.9% of the arithmetic with attention at 0.13% (S=512), and names **#440**
  (tile-32/2x2 GEMM) the highest-value Phase 3 item. The issue's "measure before
  optimising the MLP path" advice is right; its expectation of *where* time goes is
  not. **This issue buys correctness, not speed** — say so plainly in the PR.
- **The Laya head is out of scope.** Phase 2 (`LayaDecisionHead<T>`) does not exist
  — no such symbol, no `LayaPromptBuilder`, no `laya` mode, no tracking issue. With
  no CPU head, the head gate would have to be GPU-vs-PyTorch directly, dragging in
  all of Phase 2's fixture apparatus. The head needs zero new kernels (pre-norm,
  *biased* LayerNorm, ReLU, fused *biased* QKV) and becomes a small follow-up once a
  CPU head is gated. Tracked in the issues log.

## Net kernel work

Two new kernels, one modified, everything else reused:

| Piece | Status |
|---|---|
| `ElementwiseKernels.GeluExact(float)` | promoted from the duplicated `GemmKernels.Gelu` |
| `ElementwiseKernels.Rotary` | **new** |
| `ElementwiseKernels.GeGlu` | **new** |
| `AttentionKernels.BatchedAttention` | **modified** — `band` param, `-inf` clamp, drop `scores` spill |
| `ElementwiseKernels.Gather` | reused (token embeddings only) |
| `ElementwiseKernels.LayerNorm1D` | reused with a shared zero-filled beta view |
| `ElementwiseKernels.Add` | reused for both residual adds |
| `GemmKernels.TiledGemmKernelRow4` | reused (no-bias variant, `GemmKernels.cs:67`) |
| `UploadTransposed` | reused unchanged |
| `RotaryEmbedding<T>.GetPositionTables` | `internal` → `public` |

## Proposed changes

### 1. Consolidate the duplicated GELU polynomial

`GemmKernels.Gelu(float)` (`:428`) and `ElementwiseKernels.Gelu` (`:145`) are
byte-identical A–S 7.1.26 erf ports. Promote the scalar to one authoritative
`internal static float GeluExact(float)` on `ElementwiseKernels`; point the vector
kernel and all `GemmKernels` epilogues at it (AGENTS.md rule 8). Zero behavior
change; `GeGlu` needs the scalar form regardless.

**Correction found while implementing.** The two *GPU* copies were identical, but
neither is bit-identical to the CPU: `GradKernels.Erf<T>` chains
`T.FusedMultiplyAdd` (`:826-829`) while the GPU port does separate
multiply-then-add, because `XMath` exposes no FMA. Measured agreement is better
than `1e-6` relative, so the new cross-check test asserts `1e-6` — not equality —
and the plan's earlier "bit-identical to the CPU" wording for GELU was wrong. This
does **not** move the gate: DistilBERT's f32 FFN already routes through this kernel,
and the recorded baseline passes at `maxRel 3.238E-006` with it in the path.

### 2. `ElementwiseKernels.Rotary`

One thread per `(row, head, freqIndex)`, writing both halves of the pair, so
`rotate_half` needs no cross-thread communication:

```
idx = Grid.GlobalIndex.X;  if (idx >= rows * numHeads * (headDim/2)) return;
i   = idx % half;  h = (idx / half) % numHeads;  row = (idx / half) / numHeads;
t   = posIds[row] * half + i;          // per-row table lookup
b   = row * D + h * headDim;
x0 = x[b + i];  x1 = x[b + i + half];  c = cos[t];  s = sin[t];
y[b + i]        = x0 * c - x1 * s;
y[b + i + half] = x0 * s + x1 * c;
```

Operation-for-operation identical to `GradKernels.RotaryForward`
(`GradKernels.cs:250`), so bit-identical to the CPU. Reuses the `posIds` array the
runner already builds and caches. `XMath` only, never `Math`.

### 3. `ElementwiseKernels.GeGlu`

`y[r, i] = GeluExact(x[r, i]) * x[r, i + intermediate]` for `i` in
`[0, intermediate)`. The **first** half is the activated one — per
`ModernBertModel.cs:336-338` and HF `modeling_modernbert.py:89-91`
(`input, gate = Wi(x).chunk(2, -1)`; `act(input) * gate`). Getting this backwards
yields plausible-looking but wrong output.

### 4. `BatchedAttention` — three changes, one kernel (no overload)

- **`band` parameter** (`-1` = global) folded into a single
  `Keep(mask, maskBase, qPos, j, band)` predicate at the same three sites the mask
  is currently applied (`:54`, `:62`, `:70`). Matches `ModernBertMasks.Build`'s
  `max(0, i-band)` / `min(validLength-1, i+band)` exactly, because the 0/1 mask
  already encodes the padding bound. `XMath.Abs`, not `Math.Abs`.
- **`max == -inf → zeros` clamp**, written before the sum pass, mirroring
  `GradKernels.cs:487`.
- **Drop the `scores` spill.** Accumulate `headDim` probability-weighted V rows in
  a shared-memory tile and write `attnOut` once at the end. Same three score
  recomputes, and the per-`d` accumulation over `j` stays 0..S-1 ascending, so it
  is **bit-identical** to today's spill version while removing the entire
  `[B, H, S, S]` allocation.

  | S | scores buffer today (16 heads, F32) | after |
  |---|---|---|
  | 512 | 16.8 MB | 0 |
  | 2048 | 268 MB | 0 |
  | 8192 | 4.3 GB | 0 |

  **Two implementation constraints, both found by running the gate (not by reading):**
  1. ILGPU requires a `SharedMemory.Allocate` size to be **statically known**, and `headDim`
     is a runtime argument — `SharedMemory.Allocate<float>(headDim)` throws
     `NotSupportedException: The allocation size of type 'Float32' must be statically known`.
     The tile is over-allocated to `GpuBuffers.MaxHeadDim = 64` (compile-time const) and only
     the first `headDim` rows are touched. All three encoders have headDim 64, so this costs
     nothing. `ValidateAttentionLocalMemory` rejects a larger headDim rather than letting it
     read past the tile.
  2. **ILGPU's `SharedMemory` is per work *group*, not per work item.** The first attempt gave
     each thread its own `SharedMemory.Allocate<float>(64)` and every thread in the group wrote
     the same 64 floats — a race that the gate caught immediately as 97809 violations at
     maxRel 3.42, i.e. structurally wrong rather than rounding-level. The tile is therefore
     2-D: `[MaxHeadDim rows × AttentionGroupSize columns]`, with column `Group.IdxX` the
     thread's private slice and `DenseX` keeping each thread's row block contiguous.

  **Device limit queried, not assumed.** `Accelerator.MaxSharedMemoryPerGroup` (and
  `MaxNumThreadsPerGroup`) are checked at construction. Note the property is
  `MaxSharedMemoryPerGroup`, not `MaxLocalMemorySize` — the plan's original name was wrong.

### 5. `BertEncoderGpuRunner` — follow the signature change

Drop the `scores` buffer and its `8 * numHeads * 128 * 128` allocation, pass
`band: -1`, group size 256 → 64. Its `Forward` keeps the `batch<=8, seqLen<=128`
cap — lifting it is #447's job on the CPU side and out of scope.

### 6. `RotaryEmbedding<T>.GetPositionTables` → `public`

Best available tables are the CPU's own, bit-identical by construction, so RoPE
contributes *nothing* to the parity delta instead of adding an `XMath.Cos` vs
host-libm term. The runner constructs
`new RotaryEmbedding<float>(config.HeadDim, config.MaxPositionEmbeddings, theta)`
(ctor stores three ints; the cache is lazy) and calls
`GetPositionTables(0, seqLen, out cos, out sin)`, then copies the spans out (they
alias the instance cache). No formula is duplicated, so there is no drift.

`public` rather than an `InternalsVisibleTo` entry for `Nivara.Samples`: the
method already has an in-tree consumer (`LlamaDecoderBlock.cs:253,370`) and a
doc comment written as a general-purpose accessor, so `public` is the honest
signature and it keeps core's friend list to first-party projects.

### 7. `ModernBertGpuRunner` — new file

Real shapes from `samples/data/modernbert/config.json`: hidden 1024, 16 heads,
headDim 64 (half 32), 28 layers, intermediate 2624, eps 1e-5,
`global_attn_every_n_layers 3` → 10 full + 18 sliding, theta 160000 full / 10000
sliding, `local_attention 128` → inclusive band 64, max_position 8192.
`attention_bias`/`mlp_bias`/`norm_bias` all false — everything bias-free.

Per forward:
- `Gather(tokenIds, wordEmb, x)` — token embeddings only; ModernBERT has **no**
  position-embedding and **no** token-type tensor (positions come purely from RoPE).
- `LayerNorm1D(x, embedNorm.gamma, ZERO, x)`.
- 28 layers, each: `LN(attnNorm)` (skipped for layer 0 — identity, no weights) →
  `TiledGemmKernelRow4(Wqkv)` → `SplitColumns` x3 → `Rotary(q)`,`Rotary(k)` →
  `BatchedAttention(band)` → `Gemm(Wo)` → `Add` → `LN(mlpNorm)` → `Gemm(Wi)` →
  `GeGlu` → `Gemm(Wo)` → `Add`.
- `LayerNorm1D(x, finalNorm.gamma, ZERO, x)`.

Bias-free norms pass a **shared zero-filled beta view** (ILGPU views cannot be
null). This matches the CPU exactly, which also adds `0.0` via its zero-init Beta.

Per-layer theta and band come from `config.RopeTheta(i)` / `config.Band(i)`. Four
table buffers (cos/sin x full/sliding) of `seqLen * 32` floats, grown on `seqLen`
change like the existing `cachedPosSeqLen`. 16 KB each at S=128; 1 MB at S=8192.

Device footprint ~1.7 GB, consistent with the 1.685 GB measured by `--gpu-alloc`.

**Two things the plan missed, both found by running the gate and neither visible
by reading the CPU encoder:**

1. **The fused QKV projection is row-major, so `[q | k | v]` are interleaved per
   row.** The plan assumed contiguous sub-views into the fused buffer would
   isolate the three blocks. They do not: row `r` of the device buffer is
   `[q(r) | k(r) | v(r)]`, so a contiguous sub-view returns the right values for
   row 0 and the wrong values for every row after it. `ElementwiseKernels.SplitColumns`
   (`dst[r, c] = src[r, part*blockCols + c]`, launched three times) walks rows
   instead, into a `qkvSplit` buffer holding three dense `[rows, hidden]` blocks.
   `GeGlu` needed no equivalent: it indexes the fused gate/up buffer directly, so
   its interleave was already handled.
2. **The embedding norm's output is the residual stream.** The first draft wrote
   it to `normed` and left the raw embedding in `x`, so layer 0's QKV and the
   final norm both read pre-norm values. It runs **in place** now. Safe because
   `LayerNorm1D` owns a whole row across both reduction passes before writing any
   of it, so a work item never reads an element another has already overwritten.

### 8. Dispatch + gate

`modernbert` `--gpu` branch at `Program.cs:270` and a GPU-vs-CPU compare mode
modeled on `RunMiniLmGpuCompare` (`Program.cs:1121`) — in-process, identical
tokenization, so it runs without the PyTorch fixtures. Bound `GateRelTol = 1e-3`.
Reuse `ModernBert.LoadConfig` / `LoadTokenizer` / `PadTo` so both sides tokenize
identically.

**Correction — diff the whole buffer, not the valid prefix only.** The plan
followed `ModernBert.cs:12-17` and dismissed the padding region. That reasoning
is right against PyTorch and wrong against the CPU encoder. A fully-masked row
produces a zero *attention output*, but the row's final value is that folded
into 28 layers of residual adds, so the region is neither zeros nor an
implementation-defined constant — it is deterministic on both sides and there is
nothing to dismiss. The region is now gated separately and reports
maxAbs 6.866E-004. Finiteness is the load-bearing half of that gate: a missing
`-inf` clamp shows up as `NaN`, not as a wrong number.


### 9. Docs

`docs/BERT-GPU.md` — a §3-style "what we learned" entry (the local-memory budget
and the F32 summation-order floor are both non-obvious and belong there) plus the
ModernBERT row. `samples/NivaraInference/README.md` — model table, mode list, the
`--gpu` capability line, and the Phase 3 status.

## Grounding (G1) — done, cleared

Every file the plan touches was read end to end. The plan's claims all held, with two
corrections.

**Confirmed by reading, not assumption.** `BatchedAttention` applies the mask at exactly
`:54/:62/:70`, spills at `:66`, has no `-inf` clamp, and `grep IsFinite|NaN|IsNaN` over
`samples/Nivara.Samples/Gpu/` returns zero hits. The spill removal is **bit-identical**: the
current inner loop is `for d { acc=0; for j ascending: acc += p_j * v_jd }`; keeping `d`
outer, `j` inner ascending, and recomputing `p` by the same expression preserves every
per-`d` addition order, and `p` is the same f32 whether stored to the spill or used
directly. `ModernBertMasks.Build` confirms `band < 0` means global — the `band: -1`
convention matches the CPU's own. `ModernBertMlp.cs:336-338` confirms the first
`Wi` half is activated; `ModernBertAttention` applies RoPE to **q and k only**, not v;
layer 0's attention norm is a true identity with no checkpoint weights.
`TiledGemmKernelRow4` (no bias) exists at `GemmKernels.cs:67`.

**Correction A — extract a shared `GpuBuffers` helper.** `UploadTransposed`,
`UploadPlain`, `Alloc`, `AllocInt`, `Req`, `Ensure`, `Readback`, `Cfg` and the GEMM
grid/groupSize math are all `private static` on `BertEncoderGpuRunner`, so
`ModernBertGpuRunner` cannot reach any of them, and it needs the identical
`(ceil(rows/16), ceil(cols/64))`-of-`16x16` launch. ~40 lines get extracted into one
authoritative helper both runners call (AGENTS.md rule 8) rather than duplicated.

**Correction B — query the local-memory limit, do not hard-code it.** MS Learn has no
OpenCL local-memory reference (it redirects to the Khronos registry), so the "32 KB
full-profile minimum" figure is not verifiable in-repo. `runtime.Accelerator
.MaxLocalMemorySize` is reachable, so the runner **asserts
`groupSize * headDim * 4 <= MaxLocalMemorySize`** at construction and derives the group
size from the queried value. Strictly safer than hard-coding 64 or citing the spec.

**Baseline recorded** (this branch is still identical to `main`). GPU reachable:
`Intel(R) Graphics (Intel(R) Corporation)`; distilbert GPU forward 75 ms vs CPU 1434 ms.

| Gate | maxAbs | maxRel | violations |
|---|---|---|---|
| `distilbert --gpu compare` (seqLen 128) | 1.526E-005 | 3.238E-006 | 0/98304 |
| `minilm --gpu compare` hidden | 1.872E-005 | 1.037E-005 | 0/245760 |
| `minilm --gpu compare` pooled | 1.788E-007 | 1.471E-007 | 0/1920 |

Both PASS. Because the spill removal is claimed to be bit-identical, these must come back
**identical to the digit** after the change — the strongest available regression check.

## Verification

1. ~~Pre-flight baseline~~ **done** — see Grounding (G1) above.
2. ~~`dotnet build Nivara.slnx` clean, no new warnings.~~ **done** — 0 warnings, 0 errors.
3. ~~`ElementwiseGernels` GELU promotion is a no-op.~~ **done** — distilbert/minilm
   gates below are unchanged **to the digit**, which is the whole check.
4. ~~Post-change regression.~~ **done**:

   | Gate | maxAbs | maxRel | violations | vs baseline |
   |---|---|---|---|---|
   | `distilbert --gpu compare` (seqLen 128) | 1.526E-005 | 3.238E-006 | 0/98304 | identical |
   | `minilm --gpu compare` hidden | 1.872E-005 | 1.037E-005 | 0/245760 | identical |
   | `minilm --gpu compare` pooled | 1.788E-007 | 1.471E-007 | 0/1920 | identical |

   The spill removal, the group-size change and the `band` parameter are all
   reached through the one `LoadKernel` delegate in `BertEncoderGpuRunner`, which
   passes `GlobalAttentionBand = -1`, so bit-identity is the expected outcome and
   it held.
5. ~~New NUnit test: the promoted `GeluExact` scalar equals CPU
   `GradKernels.GeluExact`.~~ **done** — `GpuElementwiseParityTests` has two:
   `GeluExact_GpuScalar_AgreesWithCpuKernel` (asserts `1e-6` *relative*, not
   equality — see the FMA correction above) and
   `GeluExact_GpuScalar_OddPartRecoversTheInput` (the `gelu(v) - gelu(-v) = v`
   identity, which needs no CPU reference at all).
6. ~~`modernbert --gpu compare`.~~ **done — PASS.** Fixture seqLen 128, 26 valid:

   | Region | maxAbs | maxRel | cosine |
   |---|---|---|---|
   | valid (26 x 1024 = 26,624 values) | 2.861E-005 | 5.577E-004 | 1.0000001 |
   | padding (104,448 values) | 6.866E-004 | 4.867E-003 | — |

   0 values beyond `1e-3·(1+|cpu|)` in either region; 0 non-finite on either side.
   GPU forward 415–449 ms against a 7.3–7.5 s CPU reference forward.
7. ~~`modernbert --gpu` default + `benchmark`.~~ **done.** 10 sample sentences,
   89–160 ms each. Benchmark rows: 128 → 400 ms, 512 → 2.30 s, 2048 → 19.6 s,
   4096 → 63.0 s. The 4096 row is past the CPU's `MaxDenseLength` of 2048 on
   purpose — see §8.
8. ~~Long-sequence / `-inf` clamp check.~~ **done, and stronger than planned.** The
   fixture's `validLength = 26` with band 64 leaves query rows 90..127 with no
   visible key, so 38 of 128 rows take the clamp; the gate reports 0 non-finite
   values in that region and diffs it against the CPU rather than asserting a
   constant (see the §8 correction).
9. ~~Ask before running the full `dotnet test` suite.~~ **pending** — not yet asked.

## What the gate found that reading did not

Recorded because the shape of it is the reusable lesson, not the individual bugs.

**Three of the five "bugs" chased during implementation were faults in the
throwaway stage diagnostic, not in the GPU.** In order:

1. The `qkv` stage reference concatenated CPU q/k/v **block-major** while the
   device buffer is row-major. Row 0 matched, everything after it did not.
2. The `rope` stage reference interleaved q and k **per row** while the
   post-`SplitColumns` readback is **block-contiguous** — the same class of
   mistake, opposite direction, in the diagnostic written to check the fix for #1.
3. `geglu` read as a perfect 100% relative error, which is the signature of a
   missing activation rather than a kernel fault: the reference computed
   `inputProj(x) * gateProj(x)` and omitted the `GeluExact` on the first half.

The two genuine defects were the row-interleave `SubView` (now `SplitColumns`) and
the non-in-place embedding norm, both listed in §7.

**Method that worked:** build the reference with the *exact* layout the device
buffer has, and verify a suspected kernel defect with an isolated probe before
blaming the runner. `TiledGemmKernelRow4` at 128x1024x3072 was confirmed exact
(maxAbs 1.5e-5, 0 bad) on the same accelerator, before and after the runner
constructor, and the runner's own `Wqkv` device buffer was confirmed bit-exact
against the host transpose (0/3,145,728 mismatches) — which is what ruled the
kernel and the upload out and pointed at the reference.

**One red herring, recorded so it is not re-investigated:** the split→rotary
dependency was suspected to be unordered. `AcceleratorStreamFlags` does not exist
in ILGPU 1.5.3, `Accelerator` exposes only `CreateStream()` and `DefaultStream`
with no ordered-stream option, and `accelerator.CreateStream()` is what
`BertEncoderGpuRunner` and the GpuProbe ILGPU leg both already use. Once the
diagnostic reference was corrected the dependency was never a problem. **The
stream is in order; do not go looking for an ordering bug here.**


## Blast radius

| Change | Affects | Downstream | Test cover |
|---|---|---|---|
| `BatchedAttention` signature + `band` + clamp + spill removal | `Gpu/AttentionKernels.cs` | `BertEncoderGpuRunner` **only** (reached via one `LoadKernel` delegate, `BertEncoderGpuRunner.cs:220`) → distilbert, distilbert_sst, minilm GPU scenarios | none (no NUnit tests exist for GPU kernels); CLI compare modes are the gate |
| `GeluExact` promotion | `Gpu/ElementwiseKernels.cs`, `Gpu/GemmKernels.cs` | every GEMM epilogue + GeGlu | new NUnit cross-check vs CPU |
| `Rotary`, `GeGlu`, `SplitColumns` (new) | `Gpu/ElementwiseKernels.cs` | `ModernBertGpuRunner` only | new code, gated by compare mode |
| `GetPositionTables` → public | `src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs` | **widens core's public API** | existing `RotaryEmbeddingTests` (2 suites) |
| `BertEncoderGpuRunner` update | `Gpu/BertEncoderGpuRunner.cs` | distilbert, distilbert_sst, minilm GPU | CLI compare modes |
| `ModernBertGpuRunner` (new) | `Gpu/ModernBertGpuRunner.cs` | nothing existing | new |
| Dispatch + compare mode | `NivaraInference/Program.cs`, `ModernBert.cs` | `modernbert` mode only | new |
| Docs | `docs/BERT-GPU.md`, sample README | — | — |

**No `src/Nivara` behavior changes** except the one visibility widening. All
numerics live in `samples/Nivara.Samples` and the sample CLI.

## Planned commits

1. `docs: plan #449 ModernBERT GPU path in TODO.md` — `97885a0`
2. `refactor: extract the shared GPU upload/launch helpers into GpuBuffers` — `037fa79`
3. `refactor: promote the duplicated GPU GELU polynomial to one scalar helper` — `88e5c39`
4. `fix: clamp a fully-masked attention row to zeros on the GPU path` — `3a4c0de`
5. `perf: drop the [B,H,S,S] score spill from the fused GPU attention kernel` — `41aa867`
6. `feat: add the RoPE and GeGLU elementwise kernels for the GPU path` — `d9b5580`
7. `feat: make the RoPE position tables reachable from a GPU runner` — `94193d2`
8. `feat: add ModernBertGpuRunner with per-layer band and rope theta` — `863d29b`
9. `feat: wire modernbert --gpu and add the GPU-vs-CPU compare gate` — `a33e053`
10. `docs: record the ModernBERT GPU path in BERT-GPU.md and the sample README` — pending

Commits 4 and 5 are separate on purpose: the clamp is a correctness fix that stands
alone, and the spill removal is the perf/memory change. Both touch the same kernel, so they
may land as one if the intermediate state does not build.


## GitHub issues log

As each task executes, if deferred work or a concern is found, create it
immediately (`gh issue create --repo khurram-uworx/Nivara`) and record the number
here. Do not rely on memory — compaction can lose it.

- [x] #449 — this work (GPU path for ModernBERT)
- [x] #440 — tile-32/2x2 GEMM; LAYA.md's stated top Phase 3 item, untouched here
- [x] #447 — banded/sparse attention kernel; also the reason the CPU
      `MaxDenseLength` cap and the `BertEncoderGpuRunner` `seqLen<=128` cap stay
- [x] #448 — mask-as-select; **not** a prerequisite (the local `-inf` clamp is),
  but the same class and should be closed by the same reasoning
- [x] #460 — Phase 2 Laya head (`LayaDecisionHead<T>` + `LayaPromptBuilder` + `laya`
  mode + PyTorch gate). Discovered while scoping #449; filed. The GPU head needs zero
  new kernels and becomes a small follow-up once the CPU head is gated.
