# BF16 / Half SIMD Acceleration — Probe & Investigation

## Problem

MiniLM inference runs **~26× slower** with BF16/Half vs F32 on CPU
(measured ~3658 ms vs ~142 ms in `samples/NivaraInference`). The BCL
`TensorPrimitives` routes these narrow types through **scalar fallback loops**
because `Vector<BFloat16>.IsSupported == false` on .NET 11. The memory win
(half weight size) is real; the CPU compute cost currently negates it.

This is a **probe/investigation** to determine whether .NET 11 Hardware
Intrinsics (`System.Runtime.Intrinsics` — `Vector128`, `Widen`, F16C) can
implement faster BF16/Half kernels via the widen-compute-narrow strategy:
widen 16-bit values to `float` lanes, run genuine SIMD float math, then narrow
back.

## Proposed Approach

The strategy: **Load 8× BF16/Half as `Vector128<ushort>` → widen to float lanes
→ SIMD compute in float → narrow back to 16-bit.**

- **BFloat16** is losslessly the top 16 bits of float32 → widen via `<< 16`
  bit shift + reinterpret; narrow via `>> 16` (rounding: truncation, matching
  current `T.CreateChecked` truncation in the scalar path). No hardware-specific
  intrinsic needed; this is the clean primary path.
- **Half** requires a real conversion. **An important grounding discovery:** .NET 11
  does **not** expose an `F16C` batch intrinsic (`F16C.ConvertToVector128Single`
  does not exist in the net11 `System.Runtime.Intrinsics` surface — the scalar
  JIT consumes `vcvtph2ps`/`vcvtps2ph` internally, but there is no public batch
  intrinsic class). The Half SIMD path therefore widens/narrows via a portable
  element-wise `BitConverter` conversion while still accumulating in float SIMD.

Probe lives at `tests/Nivara.SimdProbe/` as a net11.0 console app (mirrors the
`Nivara.PerformanceTests` pattern — run manually, **not** part of CI's NUnit
run).

## Measured Results (Release, X64, .NET 11)

All correctness checks pass (DotBf16/DotHalf/AddBf16/MultiplyBf16/RmsNormBf16).

Dot product (the matmul hot path), median of 7 trials × 5000 reps:

| n     | BF16 scalar | BF16 SIMD | speedup | Half scalar | Half SIMD | speedup |
|-------|------------|-----------|---------|-------------|-----------|---------|
| 128   | 1456 ns    | 1542 ns   | slower  | 2123 ns     | 2588 ns   | slower  |
| 384   | 4090 ns    |  200 ns   | 20.4×   | 3424 ns     |  563 ns   | 6.1×    |
| 768   | 6643 ns    |  384 ns   | 17.3×   | 6901 ns     | 1105 ns   | 6.2×    |
| 1536  | 22529 ns   |  956 ns   | 23.6×   | 18892 ns    | 2942 ns   | 6.4×    |
| 3072  | 47120 ns   | 3940 ns   | 12.0×   | 44360 ns    | 9669 ns   | 4.6×    |

Element-wise (n=3072): AddBf16 **2.4×** (SIMD now matches the F32 reference),
MulBf16 **1.7×**.

**Headline: BFloat16 SIMD dot runs ~12–24× faster than the scalar BCL fallback**
at the vector lengths MiniLM uses (384/768/1536), directly attacking the ~26×
MiniLM slowdown. Half wins ~4.6–6.4× (slower at n=128 where conversion overhead
dominates).

## Proposed Changes (probe)

- `tests/Nivara.SimdProbe/Nivara.SimdProbe.csproj` — net11.0 console app
  referencing Nivara core + Samples.
- `NarrowSimdKernels.cs` — widen-compute-narrow SIMD kernels:
  - `DotBf16` / `DotHalf` (matrix multiply row-dot hot path)
  - `AddBf16` / `AddHalf`, `MultiplyBf16` / `MultiplyHalf`
  - `RmsNormBf16` (per-row RMSNorm)
  - (GELU dropped — no `MathF.Erf`/`Vector128.Erf` in BCL; not a matmul hot path)
- `Correctness.cs` — compare SIMD results against scalar baseline.
- `Benchmark.cs` — median-of-trials timed harness scalar vs SIMD.
- `Program.cs` — CLI: `correctness`, `benchmark`, or `all`.

## Verification Steps

1. `dotnet build tests/Nivara.SimdProbe` succeeds.
2. Correctness: SIMD kernels match scalar within float tolerance. ✅
3. Benchmark: SIMD row-dot vs scalar — measured 12–24× BF16, 4.6–6.4× Half. ✅
4. Next: wire SIMD matmul into a MiniLM BF16 forward and measure end-to-end
   (target < 200 ms vs ~3658 ms scalar).

## Planned Commits

1. `probe: scaffold Bfloat16SimdProbe project + plan in TODO.md` ✅
2. `probe: add BFloat16 SIMD dot-product kernel (widen-compute-narrow)`
3. `probe: add Half SIMD dot-product kernel (portable conversion, no F16C batch)`
4. `probe: add element-wise + RMSNorm SIMD kernels`
5. `probe: correctness validation + benchmark harness`
6. `probe: end-to-end MiniLM BF16 benchmark + results`

(Commits will be adjusted as the probe's findings dictate — this is an open
investigation, not a fixed implementation.)

## Blast Radius

Entirely **additive and sample/test-scoped** — the probe lives under
`tests/Nivara.SimdProbe/` and references (but does not modify) Nivara core and
`Nivara.Samples`. Nothing in `src/` is touched. The only risk is if the probe is
later promoted into `src/Nivara` kernels (e.g. `TensorsHelper`,
`RMSNormKernel`, `Adam`), which would be a separate follow-up.

## Blockers / Red Flags

- **No F16C batch intrinsic on .NET 11** — resolved: Half uses a portable
  element-wise conversion in the widen/narrow step (still wins 4.6–6.4×);
  BFloat16 bit-shift needs no intrinsic at all and is the primary target.
- Small-vector (n < 128) dot: SIMD widening overhead exceeds the benefit — the
  scalar path remains appropriate there.
- BFloat16 result rounding: SIMD truncates via `>> 16`, matching the scalar
  `T.CreateChecked` truncation; validated within float tolerance.

## GitHub issues log

- (none yet — probe work; any follow-up refactors/limitations discovered will be
  filed here during execution)
