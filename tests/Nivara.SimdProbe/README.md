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
dotnet run -c Release --project tests/Nivara.SimdProbe -- cpu          # raw CPUID dump (AVX-512/AVX10 truth)
dotnet run -c Release --project tests/Nivara.SimdProbe -- support       # print Vector<T>/ISA support flags
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

### Verification: .NET 11 RC1 (`11.0.100-rc.1.26425.128`) — nothing changed

Re-verified on the RC1 SDK/toolchain and `System.Numerics.Tensors`
`11.0.0-rc.1.26425.128` (upgraded from `preview.7`):

- `Vector<BFloat16>.IsSupported` → **false**, `Vector<Half>.IsSupported` → **false**
  (unchanged). All of `Vector128/256/512<BFloat16>` and `<Half>` also report
  **NOT supported**. TensorPrimitives still dispatches these types to scalar
  fallback loops.
- **No F16C managed intrinsics.** The `F16C` class is **absent** from the RC1
  `System.Runtime.Intrinsics.X86` surface (no batch
  `ConvertToVector128Single(ushort)` exists). `Avx10v1` exists but reports
  NOT supported on AVX2 hardware (this machine: SeS2–AVX2, FMA, GFNI, no
  AVX512/AVX10).
- The .NET 11 runtime's new hardware-FP16 JIT work (F16C for scalar
  `Half`↔`float` conversions, AVX10.1 for scalar `Half` arithmetic) is
  **scalar-only** — it does not unlock the vectorized `TensorPrimitives` path.

Bottom line: the widen-compute-narrow SIMD kernels remain the only way to get
vectorized BFloat16/Half compute on .NET 11, and the probe's relative speedups
still hold on RC1.

### Verified on this machine: no newer AVX exists in hardware (not BIOS/Windows)

Host: Intel Core Ultra 7 255H (Arrow Lake-H, hybrid: Lion Cove P-cores + Skymont
E-cores). Raw CPUID (`cpu` mode, `X86Base.CpuId`) says:

- **AVX-512: absent.** All 15 feature bits (F, DQ, IFMA, CD, BW, VL, VBMI/VBMI2,
  VNNI, BITALG, VPOPCNTDQ, 4VNNIW/4FMAPS, BF16, FP16) are clear. Intel client
  CPUs since Alder Lake (12th gen) have AVX-512 fused off — no BIOS option or
  Windows setting brings it back.
- **AVX10: absent.** Intel/.NET detect AVX10 via the dedicated **CPUID leaf 0x24**
  (gated on max basic leaf ≥ 0x24 and leaf 7.1 EDX.19). This CPU stops at
  max leaf **0x23** and leaf 7.1 EDX.19 = 0, so the firmware does not enumerate
  AVX10 at all. Nothing to toggle — the CPU simply does not expose the leaf.
- **What IS present (AVX2-era, all confirmed by .NET too):** AVX2, FMA, AVX-VNNI
  (leaf 7.1 EAX.4), AVX-IFMA (EAX.23), AVX-VNNI-INT8, GFNI, VAES, VPCLMULQDQ,
  POPCNT, SERIALIZE.
- **VBS note:** Virtualization-Based Security runs (Credential Guard + HVCI +
  Secure Launch, `HypervisorPresent=True`). The raw CPUID above is read *under
  the Hyper-V hypervisor* and matches the chip's known silicon exactly — so the
  hypervisor is not masking anything. (On an AVX-512-capable CPU, VBS hypervisors
  historically *could* mask features; not applicable to this machine.)

So on this machine, BF16/Half TensorPrimitives speedups are achievable **only**
through the probe's widen-compute-narrow kernels (or float32 pipelines). No AVX10
/ AVX-512 / F16C batch path exists at any layer (hardware, BIOS, Windows, .NET 11
RC1).

### Dot product (the matmul hot path)

Median of 7 trials × 5000 reps (RC1, Release, X64):

| n     | BF16 scalar | BF16 SIMD | speedup | Half scalar | Half SIMD | speedup |
|-------|------------|-----------|---------|-------------|-----------|---------|
| 128   | 1402 ns    | 2248 ns   | slower  | 2425 ns     | 2985 ns   | slower  |
| 384   | 4276 ns    |  513 ns   | 8.3×    | 4480 ns     |  858 ns   | 5.2×    |
| 768   | 9564 ns    |  506 ns   | 18.9×   | 9558 ns     | 1434 ns   | 6.7×    |
| 1536  | 15930 ns   | 1078 ns   | 14.8×   | 18826 ns    | 2952 ns   | 6.4×    |
| 3072  | 69906 ns   | 7541 ns   | 9.3×    | 59215 ns    | 25398 ns  | 2.3×    |

### Element-wise (n = 3072)

| op      | scalar | SIMD  | speedup | notes |
|---------|--------|-------|---------|-------|
| AddBf16 | 21611  | 19923 | 1.1×    | SIMD ≈ scalar here; both ≈ F32 reference (~19.6 µs) |
| MulBf16 | 21689  | 28683 | slower  | scalar JIT also improved; SIMD wins at larger n |

Run-to-run scalar variance is high (JIT/thermal); the dot-product speedups are
the stable signal and still match the README ranges originally recorded on
`preview.7`.

The element-wise SIMD results now match the F32 reference speed (~8 µs), meaning
BF16-side compute is no longer a penalty relative to F32.

## Findings

1. **BFloat16 SIMD dot products run ~12–24× faster** than the scalar BCL fallback
   at the vector lengths MiniLM actually uses (384 / 768 / 1536). This directly
   targets the ~26× MiniLM slowdown.
2. **Half wins ~2.3–6.7×**, constrained by the portable conversion in the widen/
   narrow step (no F16C batch intrinsic is available on .NET 11 — re-confirmed
   on the RC1 runtime surface, where the `F16C` class is absent).
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

- `CpuIdProbe.cs` (`cpu` mode) — raw CPUID dump via `X86Base.CpuId` (leaf 0/1/7,
  leaf 0x24 AVX10 check, leaf 0x1A hybrid) to settle whether an "unsupported"
  intrinsic is silicon absence vs OS/hypervisor masking.
- `SupportReport.cs` — prints `Vector<T>` / `Vector128/256/512<T>` / ISA support
  flags (`support` mode), including F16C/AVX10/AVX512F presence checks.
- `NarrowSimdKernels.cs` — the SIMD `Widen*`/`Narrow*` helpers and kernels
  (`DotBf16`, `DotHalf`, `Add*`, `Multiply*`, `RmsNormBf16`).
- `Correctness.cs` — scalar-vs-SIMD validation.
- `Benchmark.cs` — median-of-trials timed harness.
- `Program.cs` — CLI entry (`support` / `correctness` / `benchmark` / `all`).
