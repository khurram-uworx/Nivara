# TODO — #440 GEMM throughput (tile-32 / 2×2) on the ILGPU encoder path

Plan for [issue #440](https://github.com/khurram-uworx/Nivara/issues/440), branch `khurram/440` off `main`.

> **Phase 1 result (2026-09-29) — see the "Phase 1 result" section at the end. It partially
> invalidates the premise below and escalated a re-scope decision to the human before Phase 2
> was started.** The arithmetic in "Problem" was a derivation; the measurement is the number.

## Problem

After M2 fusion (#437) the launch-count lever is exhausted (~44–48 dispatches/forward, ~30 µs
dependent-kernel latency). The remaining gap is kernel throughput, and #440 names a tile-32 /
2×2 register-blocked GEMM family as the lever. The issue's own comment supplies a falsifiable
claim: the `--gemm` gate (#435) computes double-precision truth per shape, so a 2–3× claim can be
measured rather than estimated.

**A premise needs checking before the kernel work starts.** The 2026-09-27 comment argues that at
Laya's shapes GEMM is ~99.9% of the arithmetic, so "kernel throughput is essentially the whole
cost". That is a share of *arithmetic*, not of *time*. Doing the division:

- Encoder GEMM = 175.7 G MAC (28 layers × 4 GEMMs at S=512, d=1024: 512·1024·3072,
  512·1024·1024, 512·1024·5248, 512·2624·1024 = 6.28 G MAC/layer × 28).
- At the gate's measured ~200 GMAC/s → **~0.88 s**.
- Measured whole-model Laya GPU forward → **3.00 s** (`docs/LAYA.md:255`).

So GEMM is roughly **30% of wall-clock at Laya shapes**, not 99.9%. This is corroborated
independently by the dispatch structure: `ModernBertGpuRunner.cs:237-288` launches **15 kernels
per encoder layer — 4 GEMM, 2 LayerNorm, 3 `splitColumns`, 2 `rotary`, 1 attention, 1 `geGlu`,
2 `add`** — putting GEMM at 4/15 ≈ 27% of launches. Two independent routes to ~30%.

Consequence: a 2–3× GEMM win lands Laya near ~2.4–2.55 s (≈1.2× end-to-end), not at the 2×-class
the issue title targets. **Phase 1 exists to measure the split and may reorder the rest of this
work.** A measured null or negative result is a legitimate outcome and will be recorded as one.

### Structural suspect in the non-GEMM remainder

`ElementwiseKernels.LayerNorm1D` / `LayerNormResidual1D` are launched `GpuBuffers.Cfg1D(rows)` —
one work item per row, serial column loops. At Laya's 512 rows with
`GpuBuffers.LinearGroupSize = 256` that is **2 work groups** covering a 512×1024 row-reduction,
each thread serially sweeping 1024 floats three times (sum, sumSq, write). Two such launches per
layer × 28 layers. On an iGPU with a few hundred vector lanes this is single-digit-percent
occupancy, and it is a much smaller diff than four GEMM kernels. `BatchedAttention` has the same
one-work-item-per-row shape (`AttentionKernels.cs:30`, one item per (b,h,q), three serial score
passes). Both are what the issue's step 5 asks about; this plan moves them **ahead** of the kernel
work rather than after it.

## Proposed changes

### Phase 1 — measure the GEMM / non-GEMM split (no kernel changes)

Extend `tests/Nivara.PerformanceTests/GemmBenchmark.cs` with a leg-timing pass, reusing the
existing `TimeBest` helper and the same `IlgpuRuntime`:

- Time `AttentionKernels.BatchedAttention`, `ElementwiseKernels.LayerNorm1D`,
  `LayerNormResidual1D`, `Rotary`, `GeGlu`, `SplitColumns`, `Add` at the Laya and ModernBERT
  shapes already in `s_shapes`.
- Multiply per-leg µs by the per-forward launch counts read off the runners, sum, and report the
  GEMM share against a same-session `modernbert --gpu benchmark` / `laya --gpu benchmark`.
- Output: a measured GEMM-vs-rest percentage and a ranked list of which leg to attack.

No DP truth is needed for these legs (they are not gated on numerics here), so this is a timing
pass only.

### Phase 2 — the variant family (`samples/Nivara.Samples/Gpu/GemmKernels.cs`)

Add the blocking lattice next to the Row4 tree, sized so a win is *attributable* — if a bundled
kernel loses, we can tell whether the register block or the shared-memory footprint caused it.

| variant | thread tile | shared/group | isolates |
|---|---|---|---|
| `2x2` @ KTile16 | 32×32 | ~5 KB | register-blocking effect alone |
| `2x2` @ KTile32 | 32×32 | ~8 KB | + barrier halving, occupancy cost |
| `4x2` @ KTile32 | 32×64 | ~12 KB | asymmetric vs K-heavy shapes |
| `1x8` @ KTile16 | 16×128 | ~9 KB | the N-heavy extreme |

Each gets the lean per-epilogue sibling set (plain / Bias / Gelu / Relu / Qkv) — no runtime
activation byte, per the existing design note at `GemmKernels.cs:123-133` (a single kernel with a
runtime byte would inline all three activation paths and bloat registers on the hot lean launches).

Geometry constants live in `GpuBuffers` so `GpuBuffers.GemmCfg` stays the single authority for
grid math; runner call sites keep their shape and only the constants they read change.

**Occupancy is the risk.** Shared memory per group goes 5 KB → 8–12 KB, and the issue's
"occupancy/register headroom exists" is asserted, never measured. The four-variant matrix is what
makes that falsifiable.

**The 2–3× claim is not derived from the arithmetic.** 2×2 is 4 shared loads per 4 MACs against
Row4's 5 — a 20% cut. The barrier halving is real. Neither yields 2–3× on its own. The gate
decides; the title must not pre-commit the result.

### Phase 3 — gate wiring (`tests/Nivara.PerformanceTests/GemmBenchmark.cs`)

- Extend `GemmVariant`, `VariantsFor`, `ApplyEpilogue` and the launch switch for the new variants.
- **Two checks per cell**, both cheap since both kernels' buffers are already in the same loop:
  1. **Byte-identity against Row4.** Any tiling that accumulates strictly ascending `k` produces a
     bit-identical f32 result per output element, because the per-element sum order is unchanged.
     This is a *provable invariant*, so assert it rather than observe it. A violation means the
     accumulation order changed — a design bug, not a rounding difference. This is strictly
     stronger than the existing maxAbs check.
  2. Existing `maxAbs ≤ 1e-3` against double-precision truth, unchanged.
- Add 2–3 shapes stressing the new geometry's edges (K and N not multiples of 32, and an N < 8
  case). Cells ~70 → ~110; runtime roughly 2.5 min.

### Phase 4 — swap the winner

Through `GpuBuffers.GemmCfg` + the three runners' kernel loads. Only if a variant wins. A null
result gets recorded in the docs as a null result, not quietly dropped.

### Phase 5 — gates + benchmark, AC power only

`minilm | distilbert | distilbert_sst | modernbert | laya --gpu compare` must stay GATE PASS at
the existing byte-identity maxAbs floors. `--gemm` produces the GMAC/s table. AC line only — an
intermediate battery session in #437's history produced contaminated 25.6–33 ms results.

### Phase 6 — documentation

- `docs/ACCELERATION.md` §1 (measured results), §5.2 (the #440 item), §5.13 (ModernBERT lever)
- `docs/LAYA.md` "What's next" #1
- `docs/ROADMAP-SUGGESTION.md` §8
- `tests/Nivara.PerformanceTests/README.md` — new GMAC/s baseline table

## Verification steps

1. `dotnet build Nivara.slnx` — must be clean after each change unit.
2. `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm` (AC power) — all cells
   PASS, byte-identity vs Row4 holds, GMAC/s recorded.
3. `dotnet run --project samples/NivaraInference -c Release -- <model> --gpu compare` for
   minilm / distilbert / distilbert_sst / modernbert / laya — GATE PASS at existing floors.
4. `dotnet run --project samples/NivaraInference -c Release -- <model> --gpu benchmark` (AC) for
   the honest end-to-end delta.
5. `dotnet test` — **ask the human before running** (AGENTS.md).

### Verification outcome (2026-09-29)

| # | step | result |
|---|---|---|
| 1 | `dotnet build Nivara.slnx` | **clean** at every change unit, 0 warnings, 0 errors |
| 2 | `--gemm` (AC) | **PASS** — 163 cells, byte-identity PASS, no geometry skipped |
| 3 | `--gpu compare` × 5 | **PASS** — every figure identical to `ACCELERATION.md` §1, ModernBERT's `5.577E-004` bit for bit |
| 4 | `--gpu benchmark` (AC) | **not run — no delta exists to report.** Step 4 existed to measure the Phase 4 swap, and Phase 4 was cancelled, so there is no new configuration to benchmark. Running it would have re-measured the already-recorded baseline for a few minutes of machine time. |
| 5 | `dotnet test` | put to the human, not run unattended (AGENTS.md) |

**Step 4 was substituted rather than skipped, and the substitution is the point.** The plan's own
blast-radius section flags `GpuBuffers.GemmCfg` as a 14-call-site chokepoint where "a bad geometry
constant is a 3-model failure, not a 1-kernel failure" — and step 4 was the only step that would
have caught it *at speed*, since step 3 checks numerics and step 2 checks the gate's own launch
config, not the runners'. So the substitution targets the same risk: Row4's own GMAC/s in the
2026-09-29 gate run, against the 2026-09-27 baseline.

| shape | 09-27 | 09-29 | Δ |
|---|---|---|---|
| laya qkv | 202 | 201 | −1 |
| laya attn out | 202 | 202 | 0 |
| laya fc1 (Wi) | 205 | 199 | −6 |
| laya fc2 (Wo) | 197 | 197 | 0 |
| laya head ff1 | 204 | 204 | 0 |
| laya head ff2 | 155 | 154 | −1 |
| laya act 1 | 188 | 189 | +1 |
| laya scorer 1 | 59 | 59 | 0 |
| laya qkv@128 | 193 | 193 | 0 |
| distilbert fc1 | 183 | 186 | +3 |
| distilbert fc2 | 177 | 184 | +7 |

−6 to +7 GMAC/s on a 155–205 range is inside the session-to-session drift the bench README
already documents (it records +5 to +6 on unchanged rows between two earlier sessions). So
`GemmCfg`'s new 2-arg delegation resolves to the same geometry at the runners, at both the
numerical and the throughput level. **The chokepoint risk the plan named is closed, and the
closing argument is the numbers rather than the reasoning about the delegation.**

## Blast radius

| change | files | depends on | tests |
|---|---|---|---|
| Phase 1 leg timing | `GemmBenchmark.cs` | `AttentionKernels`, `ElementwiseKernels`, `GpuBuffers.Cfg1D` | `--gemm` (manual, AC) |
| Phase 2 kernels | `Gpu/GemmKernels.cs`, `Gpu/GpuBuffers.cs` | `ElementwiseKernels.GeluExact`, `GpuBuffers.MaxHeadDim`-style shared-mem contract | `--gemm`; `--gpu compare` per model |
| Phase 3 gate | `GemmBenchmark.cs` | Phase 2 kernels, `DpCore`, `ApplyEpilogue` | `--gemm` |
| Phase 4 swap | `Gpu/GpuBuffers.cs` (14 call sites), `BertEncoderGpuRunner.cs`, `ModernBertGpuRunner.cs`, `LayaHeadGpuRunner.cs` | `GpuBuffers.GemmCfg` — the single geometry authority, so all three runners move together | all five `--gpu compare` gates |

- **Sample-scoped**: no `src/Nivara`, `Nivara.Extensions`, or `src/Nivara.Gpu` changes. Nothing in
  core is affected, so `Nivara.Tests` has no new coverage obligation beyond staying green.
- **`GpuBuffers.GemmCfg` is the shared chokepoint.** 14 call sites across 3 runners route through
  it; changing its constants moves all three at once. That is the intent (one authority), but it
  means a bad geometry constant is a 3-model failure, not a 1-kernel failure.
- **`TiledGemm` / `IlgpuGemmVariant` (GemmKernels.cs:428-511) are dead code** — referenced nowhere
  outside their own file. Left alone unless a change makes them relevant; noted so a reviewer does
  not read them as the live path.
- **Shared-memory footprint** must be validated against `Accelerator.MaxSharedMemoryPerGroup` at
  construction, the way `GpuBuffers.ValidateAttentionLocalMemory` does for the attention tile.

## Planned commits

1. `docs: plan #440 GEMM throughput work in TODO.md` — `46efa73`
2. `Add a leg-timing pass to the --gemm gate for the non-GEMM kernels` — `f54dbad` (the
   `--gemm-legs` probe, split from the measurement so the probe and its result are separate
   commits), and `57116a9` for the measurement itself
3. `Add the tile-32 / 2x2 register-blocked GEMM family to GemmKernels` — `d06b8c8`, then
   `05fc6bb` for the #468 fix that commit's kernels turned out to need
4. `Wire the new GEMM variants into the --gemm gate with a byte-identity check` — `d06b8c8`
5. ~~`Route the runners at the winning GEMM variant`~~ — **cancelled, no winner exists**
6. `Record #440's measured outcome in ACCELERATION.md, LAYA.md, ROADMAP-SUGGESTION.md, and the
   performance-tests README` — `90f2b56` (finding), `e4f0e96` (docs), `b74ce26` (source labels)

Steps 3 and 4 landed as one commit, which the plan's own Phase 2 heading already merged them
into. Step 5 was conditional on a winner and there is none, so the branch ends with a recorded
negative result rather than a repointed runner. Two commits exist that the plan did not
anticipate: `05fc6bb` (a real defect found by the byte-identity check, not a planned unit) and
`b74ce26` (labelling the negative baseline in the source so the finding is legible at the point
of use).

## GitHub issues log

- [x] **#467** — LayerNorm1D / LayerNormResidual1D are one work item per row; 2 work groups
      cover a 512×1024 row-reduction at Laya shapes. Created during G1 grounding, not after the
      work — `Cfg1D(512)` = `(2, 256)` confirmed at `ModernBertGpuRunner.cs:235` and
      `BertEncoderGpuRunner.cs:355-359`, with three serial passes over 1024 floats per item.
      Split out of #440 so it is tracked independently of #440's measured outcome.
      **Phase 1 measured it at 1.2% of the Laya forward — a minor cleanup, not a lever. The
      occupancy argument holds; the magnitude in the original writeup was wrong.**
- [x] **#447** *(pre-existing, not created here)* — banded/sparse attention kernel. Phase 1
      measured `BatchedAttention` at **57.0%** of the Laya forward (1266.87 of 2221.12 ms, AC),
      the single largest leg and
      larger than all four GEMMs combined. This is where the remaining GPU time actually is, and
      #440 should not absorb it.
- [x] **#468** — ILGPU 1.5.3 OpenCL: two `SharedMemory.Allocate2D` calls in one kernel whose
      extents disagree (KT-wide A tile vs TileCols-wide B tile) get the second tile mis-placed by
      the lowering. The kernel reads a neighbour's staged data and returns plausible garbage;
      `1x8@KT16` failed at `LoadKernel` with a bare `CLException`. Created during Phase 2 at
      discovery, not after. Not a Nivara defect — the four #440 kernels sidestep it with a single
      `SharedMemory.Allocate<float>`, which is what lesson 15 already prescribes. Filed because any
      future Nivara kernel that widens a shared tile will hit it.
- [ ] #NNN — *(pending: any follow-up surfaced during Phase 4–6, created at discovery time.)*

> As each task executes, if deferred work or a concern surfaces that is outside this plan, create
> the issue immediately with `gh issue create --repo khurram-uworx/Nivara` and record its number
> above. Do not rely on memory — compaction during execution can lose it.

## Phase 1 result (2026-09-29) — the issue's premise is inverted

`--gemm-legs` measurement, Intel Arc iGPU, **AC line**. An earlier battery run gave
attention 51.3% / GEMM 45.4% on Laya; the clean AC run shifts it to 57.0% / 39.2%, so
battery was compressing the gap rather than inventing it. The conclusion got *stronger*,
not weaker, so nothing downstream of it needed revisiting.

Laya large (rows=512, hidden=1024, heads=16, ffn=2624, layers=28), 2221.12 ms/forward:

| leg | shape | ms/fwd | share |
|---|---|---|---|
| **BatchedAttention** | [16x512x512x64] | **1266.87** | **57.0%** |
| GEMM Wi | [512x1024x5248] | 378.06 | 17.0% |
| GEMM qkv | [512x1024x3072] | 223.25 | 10.1% |
| GEMM WoMlp | [512x2624x1024] | 195.02 | 8.8% |
| GEMM attn out | [512x1024x1024] | 74.51 | 3.4% |
| LayerNorm1D | [512x1024] | 28.15 | 1.3% |
| SplitColumns | [512x1024] | 27.86 | 1.3% |
| GeGlu | [512x2624] | 11.77 | 0.5% |
| Add | [512x1024] | 9.06 | 0.4% |
| Rotary | [512x16x64] | 6.49 | 0.3% |
| **total** | | **2221.12** | 39.2% GEMM / 60.8% non-GEMM |

ModernBERT base (d=768, 22L), 1536.53 ms/forward: **BatchedAttention 939.22 ms = 61.1%**,
GEMM 35.3%, LayerNorm1D 1.1%, SplitColumns 1.1%.

**Two findings, both against the plan:**

1. **GEMM is ~39% of wall-clock, not the ~99.9% of arithmetic the issue comment claims.** The
   arithmetic-vs-time distinction flagged in planning is real, but even the corrected 30%
   estimate was low — the four GEMM shapes sum to 870.84 ms of a 2221 ms forward. The
   `175.7 G MAC / ~200 GMAC/s ≈ 0.88 s` derivation under-counted because the gate's GMAC/s was
   itself taken on a differently-loaded session; the direct timing here is the better number.

2. **`BatchedAttention` is the single largest leg at 57.0%** — larger than all four GEMMs
   combined. One work item per (b,h,q) = 8192 items for Laya, each serially sweeping `seqLen`
   three times with a `headDim`-wide inner loop and a shared-memory accumulator tile. That is
   the same latency-bound shape as the LayerNorm suspect, but an order of magnitude more
   expensive. Already tracked as **#447**, so #440 should not absorb it.

**LayerNorm correction.** The G1 writeup above ranked LayerNorm a strong structural suspect and
filed #467 on that basis. It measures at 1.2% — real, but not a lever. #467 stays open as a minor
cleanup; the occupancy argument is sound, the magnitude was wrong.

## Consequence for the remaining phases

Phase 2 (four GEMM geometries) attacks a 39% term. Even the best case the theory allowed — a
full 20% register-blocking gain, which the measurement showed is not achievable — would have
moved the Laya forward by ~8% end-to-end. The 57% attention term is the larger prize by a
factor of seven and is already tracked at #447.

**Decision escalated to the human before Phase 2 was started** — proceed with the GEMM family as
scoped, or re-scope #440 against the measurement. Recorded because G2 must be able to see that
the plan was amended by evidence, not silently.

## Also decided during execution

- **Variant matrix narrowed** (human call, 2026-09-29): 4 geometries as *plain* kernels to run
  the attribution, then fuse the winner's Bias/Gelu/Relu/Qkv siblings — 8 bodies, not 20. The
  epilogue is one extra register add over the same accumulators in the same ascending-`k` order,
  so it is orthogonal to tile geometry and a geometry win transfers.
- **G1 grounding result**: the microsoft-learn MCP has no coverage for ILGPU/OpenCL (its only
  OpenCL hit is a stub pointing at khronos.org). Real grounding came from code-memory navigation
  plus `docs/ACCELERATION.md` lesson 15 — `MaxSharedMemoryPerGroup`, not `MaxLocalMemorySize`,
  and per-*group* not per-work-item shared memory. The device reports 1024 threads/group and
  65536 B shared/group, so none of the four planned geometries is at a limit.

## Phase 2–3 result (2026-09-29) — a null result, and the reason is legible

Four geometries built, gated, and measured on **AC power** (Intel Arc iGPU). All four are
byte-identical to Row4, so the numerics are settled and only throughput is in question.

Ratio to Row4 (GMAC/s, higher is better). Row4 is the `1x4@KT16` incumbent at 5 KB shared/group.

| shape | Row4 GMAC/s | 2x2@KT16 | 2x2@KT32 | 4x2@KT32 | 1x8@KT16 |
|---|---|---|---|---|---|
| laya qkv | 201 | 0.74x | 0.60x | 0.76x | 0.82x |
| laya fc1 (Wi) | 202 | 0.71x | 0.58x | 0.73x | 0.79x |
| laya fc2 (Wo) | 197 | 0.75x | 0.60x | 0.77x | 0.81x |
| laya attn out | 203 | 0.74x | 0.59x | 0.74x | 0.84x |
| laya head ff1 | 204 | 0.76x | 0.58x | 0.76x | 0.82x |
| **laya head ff2** | **155** | 0.89x | 0.67x | **0.98x** | **1.01x** |
| laya qkv@128 | 193 | 0.73x | 0.58x | 0.77x | 0.78x |
| distilbert fc1 | 189 | 0.72x | 0.59x | 0.76x | 0.78x |
| minilm qkv/o | 135 | 0.68x | 0.62x | 0.64x | 0.64x |
| edge padded rows | 61 | 0.64x | 0.74x | 0.43x | 0.43x |
| laya scorer 1 | 59 | 0.41x | 0.42x | 0.25x | 0.37x |

**No geometry wins. The best cell is 1.01x** — `1x8@KT16` on `laya head ff2`, the one shape
already known to be weak at 155 GMAC/s, and that is a wash inside run-to-run noise. Every other
cell is a regression, typically 0.6–0.85x. The narrow, launch-bound shapes (`head`, `scorer`)
regress hardest, as expected: a wider tile means fewer groups, so fewer items to hide launch
latency behind.

**Phase 4 is therefore cancelled — there is no winner to route the runners at** — and Phase 2b
(fusing the winner's Bias/Gelu/Relu/Qkv siblings) is cancelled with it. The plan's own null-result
clause applies: recorded, not dropped.

### Why, which is the part worth keeping

The theory the family was built on is *shared-memory traffic per MAC*: 2×2 is 4 shared reads per 4
MACs against Row4's 5, a 20% cut. The measurement says the binding constraint is not traffic, it
is **shared-memory capacity per group**.

The cleanest evidence is the pair that isolates it. `2x2@KT16` and `2x2@KT32` have *identical*
blocking factors, so identical shared reads per MAC, and differ only in shared memory per group
(4 KB vs 8 KB). `2x2@KT32` is consistently ~18–25% slower than `2x2@KT16` — and both are slower
than Row4, which holds 5 KB. Going the other way, `4x2@KT32` has the *best* traffic ratio in the
family (0.75 shared reads/MAC) and the *worst* footprint (12 KB), and it lands mid-pack. The
metric that predicted the ordering was the footprint, not the traffic.

So on this device the two effects roughly cancel and the occupancy loss wins: a 5 KB → 8–12 KB
tile cuts co-resident groups by more than the 20–40% traffic saving returns. That is the opposite
of the issue's "occupancy headroom exists" — not merely unmeasured, but pointing the wrong way.

### Bug found en route (#468)

All four kernels were initially wrong (maxAbs 20–119) and `1x8@KT16` would not compile at all.
Root cause is an ILGPU OpenCL lowering bug, not a kernel bug: two `SharedMemory.Allocate2D` calls
in one kernel whose extents disagree get the second tile mis-placed. `2x2@KT32` passed throughout
and the reason is the tell — its two tiles happen to share an extent of 32×32, so the lowering has
nothing to disagree about. Fixed by a single `SharedMemory.Allocate<float>` with hand-computed
offsets. Filed upstream as **#468**.

Worth noting for the byte-identity check: neither the failure nor the crash is visible from the
source. Every extent, stride and element index is in bounds and the kernel is self-consistent, so
review passes and the numbers look like a real dot product against the wrong elements. **The
byte-identity invariant is what caught it** — a `maxAbs` tolerance test alone would have been
argued with. That is the check earning its keep on the very first shape.

## Phase 5 result (2026-09-29) — all five model gates PASS, unchanged

Phase 4 was cancelled, so no runner points at a new kernel. But `GpuBuffers.GemmCfg` is the
14-call-site geometry chokepoint and its 2-arg form was changed to delegate to `Row4Geometry`, so
the gates were re-run rather than assumed.

| model | gate | maxAbs | maxRel | notes |
|---|---|---|---|---|
| minilm | PASS | 1.872E-005 | 1.037E-005 | 0/245760 violations; pooled cosine 1.000000 |
| distilbert | PASS | 1.526E-005 | 3.238E-006 | 0/98304 violations |
| distilbert_sst | PASS | 3.815E-006 | 6.677E-007 | 0/16; argmax 8/8 |
| modernbert | PASS | 2.861E-005 | 5.577E-004 | 0 non-finite; cosine 1.0000001 |
| laya | PASS | 1.657E-005 | — | 1.1% of the 1e-3 bound (choice2); act matched 4/4 |

Every figure is **identical to the value already recorded in `docs/ACCELERATION.md` §1** —
ModernBERT's `5.577E-004` bit for bit — which is the point: the `GemmCfg` refactor resolved to
the same geometry, so the chokepoint change is confirmed inert rather than merely unexamined. The
`minilm | distilbert | distilbert_sst` gate the plan names as mandatory stays PASS.

`--gemm` on AC: **163 cells, GATE PASS, byte-identity PASS**, no geometry skipped.

