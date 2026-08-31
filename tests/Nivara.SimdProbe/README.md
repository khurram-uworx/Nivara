# Nivara.SimdProbe

Probe: can .NET 11 hardware intrinsics accelerate **BFloat16 / Half** compute on
CPU via a **widen-compute-narrow** strategy, when the BCL `TensorPrimitives` path
runs scalar loops for these types?

## Background / Motivation

MiniLM inference runs **~26× slower** with BF16/Half weights vs F32 on CPU
(measured ~3658 ms vs ~142 ms in `samples/NivaraInference`). The memory win
(half the weight size) is real, but the CPU compute cost currently negates it.

Root cause: `Vector<BFloat16>.IsSupported == false` and `Vector<Half>.IsSupported
== false` on .NET 11. The BCL `TensorPrimitives` routes these narrow types
through **scalar fallback loops**.

This is a standalone, run-manually console app (mirrors the
`tests/Nivara.PerformanceTests` pattern — **not** run as part of CI's NUnit
suite). It compares hand-written `Vector128` SIMD kernels against the scalar BCL
baseline in isolation, to decide whether fast BF16/Half kernels are worth
promoting into `src/Nivara`.

## Build & Run

```bash
dotnet run -c Release --project tests/Nivara.SimdProbe -- correctness   # validate SIMD vs scalar
dotnet run -c Release --project tests/Nivara.SimdProbe -- benchmark     # timed scalar vs SIMD
dotnet run -c Release --project tests/Nivara.SimdProbe                  # both
```

Always benchmark in `-c Release` — the SIMD payloads are inlined/optimized paths
and Debug numbers are meaningless. The probe is **self-contained** (only the
`System.Numerics.Tensors` package for its scalar BCL baseline), so it builds
fast and isn't coupled to Nivara internals.

## The Strategy: Widen-Compute-Narrow

Load 8× BF16/Half as a single `Vector128<ushort>`, widen to two `Vector128<float>`
vectors (4 float lanes each), run genuine SIMD float math, then narrow back to
16-bit. This recovers SIMD even though the 16-bit types themselves are not
SIMD-vectorizable.

### BFloat16 — the clean win (no hardware intrinsic needed)

A BFloat16 bit pattern is exactly the **top 16 bits of float32**. So widening is
a pure bit-shift (no conversion), and narrowing is a shift back:

- widen:  `ushort bits << 16` → reinterpret the `uint` as `float`
- narrow: `float bits >> 16`  → truncates the mantissa (matches the scalar
  `T.CreateChecked` truncation)

Because this is pure integer bit manipulation, it is lossless, portable, and
needs **no x86-specific intrinsic**. It is the primary target.

### Half — portable conversion (no F16C batch intrinsic on .NET 11)

Half requires a real cross-format conversion. **Important grounding finding:**
.NET 11 does **not** expose an `F16C` batch intrinsic — there is no
`F16C.ConvertToVector128Single` in the net11 `System.Runtime.Intrinsics` surface
(verified against the net11 reference XML). The scalar JIT consumes
`vcvtph2ps`/`vcvtps2ph` internally, but no public batch intrinsic class exists.

The Half SIMD path therefore widens/narrows via a portable element-wise
`BitConverter` conversion (`UInt16BitsToHalf` / `HalfToUInt16Bits`) while still
**accumulating in float SIMD**. It wins, but less than BFloat16 because the
conversion step dominates.

## Results (Release, X64, .NET 11)

Correctness: all checks pass (`DotBf16`, `DotHalf`, `AddBf16`, `MultiplyBf16`,
`RmsNormBf16`) — SIMD output matches the scalar baseline within float tolerance.

### Dot product (the matmul hot path)

Median of 7 trials × 5000 reps:

| n     | BF16 scalar | BF16 SIMD | speedup | Half scalar | Half SIMD | speedup |
|-------|------------|-----------|---------|-------------|-----------|---------|
| 128   | 1456 ns    | 1542 ns   | slower  | 2123 ns     | 2588 ns   | slower  |
| 384   | 4090 ns    |  200 ns   | 20.4×   | 3424 ns     |  563 ns   | 6.1×    |
| 768   | 6643 ns    |  384 ns   | 17.3×   | 6901 ns     | 1105 ns   | 6.2×    |
| 1536  | 22529 ns   |  956 ns   | 23.6×   | 18892 ns    | 2942 ns   | 6.4×    |
| 3072  | 47120 ns   | 3940 ns   | 12.0×   | 44360 ns    | 9669 ns   | 4.6×    |

### Element-wise (n = 3072)

| op      | scalar | SIMD  | speedup | notes |
|---------|--------|-------|---------|-------|
| AddBf16 | 14589  | 5970  | 2.4×    | SIMD now ≈ F32 reference (~8.3 µs) |
| MulBf16 | 13749  | 7937  | 1.7×    | SIMD ≈ F32 reference |

The element-wise SIMD results now match the F32 reference speed (~8 µs), meaning
BF16-side compute is no longer a penalty relative to F32.

## Findings

1. **BFloat16 SIMD dot products run ~12–24× faster** than the scalar BCL fallback
   at the vector lengths MiniLM actually uses (384 / 768 / 1536). This directly
   targets the ~26× MiniLM slowdown.
2. **Half wins ~4.6–6.4×**, constrained by the portable conversion in the widen/
   narrow step (no F16C batch intrinsic is available on .NET 11).
3. **Small vectors (n < 128) are slower** for both types — the widen overhead
   exceeds the SIMD benefit. The scalar path should remain for tiny dots.
4. **Dropped GELU from the probe**: BCL has no `MathF.Erf` / `Vector128.Erf`, and
   GELU is not the matmul hot path. If needed later, use an erf approximation or
   a widened-float + `TensorPrimitives` approach.
5. At larger n (3072+) speedup plateaus toward ~12× as both arrays exceed cache,
   moving into the memory-bandwidth regime — this is the realistic matmul regime
   and the win still holds.

## Recommendations

These kernels are validated and fast in isolation. If BF16/Half inference matters
for Nivara, the natural follow-up is an **end-to-end MiniLM BF16 forward**: wire
the SIMD row-dot matmul (plus SIMD RMSNorm) into the existing BF16 MiniLM path and
measure wall-clock vs F32. The ~26× scalar regression should collapse toward ~1×
(BF16 matching F32) given the ~12–24× dot and ~2.4× element-wise gains.

Because the kernels are **memory-bandwidth-sensitive** and the whole model's
weights must be resident, an end-to-end measurement is required to confirm the
real-world number (target < 200 ms vs ~3658 ms scalar) — the standalone numbers
above strongly suggest it is achievable.

If promotion into `src/Nivara` is pursued later, the natural homes (per the
ADR-001 span-ified design) are `TensorsHelper` (matmul) and `RMSNormKernel`
(per-row RMSNorm), gated by a length check so small vectors keep the scalar path.

## Files

- `NarrowSimdKernels.cs` — the SIMD `Widen*`/`Narrow*` helpers and kernels
  (`DotBf16`, `DotHalf`, `Add*`, `Multiply*`, `RmsNormBf16`).
- `Correctness.cs` — scalar-vs-SIMD validation.
- `Benchmark.cs` — median-of-trials timed harness.
- `Program.cs` — CLI entry (`correctness` / `benchmark` / `all`).
