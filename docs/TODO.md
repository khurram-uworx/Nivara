# TODO — #440 GEMM throughput (tile-32 / 2×2) on the ILGPU encoder path

Plan for [issue #440](https://github.com/khurram-uworx/Nivara/issues/440), branch `khurram/440` off `main`.

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

1. `docs: plan #440 GEMM throughput work in TODO.md`
2. `Add a leg-timing pass to the --gemm gate for the non-GEMM kernels`
3. `Add the tile-32 / 2x2 register-blocked GEMM family to GemmKernels`
4. `Wire the new GEMM variants into the --gemm gate with a byte-identity check`
5. `Route the runners at the winning GEMM variant` *(only if Phase 4 has a winner)*
6. `Record #440's measured outcome in ACCELERATION.md, LAYA.md, ROADMAP-SUGGESTION.md, and the
   performance-tests README`

## GitHub issues log

- [ ] #NNN — *(pending: the LayerNorm/attention occupancy finding, if Phase 1 confirms it —
      likely deserving of its own issue ahead of any GEMM kernel work. Created at discovery time,
      not after the plan.)*
- [ ] #NNN — *(pending: any follow-up surfaced during Phase 2–5, created at discovery time.)*

> As each task executes, if deferred work or a concern surfaces that is outside this plan, create
> the issue immediately with `gh issue create --repo khurram-uworx/Nivara` and record its number
> above. Do not rely on memory — compaction during execution can lose it.
