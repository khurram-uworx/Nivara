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
  current `T.CreateChecked` truncation in the scalar path).
- **Half** requires a real conversion. .NET 11 JIT already uses the **F16C**
  hardware instructions (`vcvtph2ps` / `vcvtps2ph`) for scalar Half↔float;
  the batch intrinsic path (`F16C.ConvertToVector128Single`) exists.

Probe delivers a standalone console app `samples/Bfloat16SimdProbe/` measuring
kernels in isolation and then, if promising, an end-to-end MiniLM BF16
benchmark.

## Proposed Changes (probe)

- `samples/Bfloat16SimdProbe/Bfloat16SimdProbe.csproj` — net11.0 console app
  referencing Nivara core + Samples.
- `SimdKernels.cs` — widen-compute-narrow SIMD kernels:
  - `DotProductBFloat16` / `DotProductHalf` (matrix multiply row-dot hot path)
  - `AddBFloat16` / `AddHalf`, `Multiply*`
  - `RmsNormBFloat16` (per-row SIMD RMSNorm)
  - `GeluBFloat16`
- `CorrectnessTests.cs` — compare SIMD results against scalar BCL path.
- `BenchmarkHarness.cs` — timed loops scalar vs SIMD.
- `Program.cs` — CLI to run individual probes.

## Verification Steps

1. `dotnet build samples/Bfloat16SimdProbe` succeeds.
2. Correctness: SIMD kernels match scalar within float tolerance.
3. Benchmark: SIMD row-dot vs scalar — measure speedup (expect 4–8×).
4. If promising: wire SIMD matmul into a MiniLM BF16 forward and measure
   end-to-end (target < 200 ms vs ~3658 ms scalar).

## Planned Commits

1. `probe: scaffold Bfloat16SimdProbe project + plan in TODO.md`
2. `probe: add BFloat16 SIMD dot-product kernel (widen-compute-narrow)`
3. `probe: add Half SIMD dot-product kernel (F16C path)`
4. `probe: add element-wise + RMSNorm + GELU SIMD kernels`
5. `probe: correctness validation + benchmark harness`
6. `probe: end-to-end MiniLM BF16 benchmark + results`

(Commits will be adjusted as the probe's findings dictate — this is an open
investigation, not a fixed implementation.)

## Blast Radius

Entirely **additive and sample-scoped** — the probe lives under
`samples/Bfloat16SimdProbe/` and references (but does not modify) Nivara core
and `Nivara.Samples`. Nothing in `src/` is touched. The only risk is if the
probe is later promoted into `src/Nivara` kernels (e.g. `TensorsHelper`,
`RMSNormKernel`, `Adam`), which would be a separate follow-up.

## Blockers / Red Flags

- Whether the F16C batch intrinsic (`F16C.ConvertToVector128Single`) is exposed
  in .NET 11 and yields a measurable win over element-wise scalar conversion —
  to be measured in the probe, not assumed.
- BFloat16 result rounding: the current scalar `T.CreateChecked` path truncates;
  the SIMD path should match that (or be validated as an acceptable improvement).

## GitHub issues log

- (none yet — probe work; any follow-up refactors/limitations discovered will be
  filed here during execution)
