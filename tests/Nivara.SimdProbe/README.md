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
dotnet run -c Release --project tests/Nivara.SimdProbe -- scalar       # Qwen decode scalar hot-path: RoPE + attention V-phase (AVX-512 probe)
dotnet run -c Release --project tests/Nivara.SimdProbe -- transpose    # #136 transpose A/B: tiled kernel vs BCL view+flatten
dotnet run -c Release --project tests/Nivara.SimdProbe -- tensor-api   # reading one row of a rank-2 Tensor<T> (docs fixes, #528)
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

## Qwen decode scalar hot-path probe (`scalar` mode)

Qwen2.5-0.5B F32 decode (~441 ms/token on this machine) is already
TensorPrimitives-backed end to end — and TensorPrimitives is vectorized
(AVX-512-capable) inside the .NET runtime, so replacing those calls would be
reinventing the wheel. A kernel-by-kernel audit of the per-token hot path found
exactly **two scalar (non-TensorPrimitives) kernels**: RoPE forward
(`GradKernels.RotaryForward`) and the decode-attention V-weighted accumulation
(`AttentionKernels.DecodeAttention`). This mode benchmarks hand-rolled
`Vector512` branches for both at the exact Qwen shapes, plus the current
TensorPrimitives GEMV path as an "already vectorized" baseline.

Host for the numbers below (documented by the probe itself): 11th Gen Core
i5-1135G7 (Tiger Lake), **AVX-512 present** (F/DQ/L/BW/VL etc.,
`Vector512<float>.Count = 16`). Earlier README sections describe an Arrow Lake
host — that is a different machine; this section records the real one.

Results (Release, X64, .NET 11, median of 7 trials):

### Correctness
All PASS — AVX-512 RoPE/V-phase match scalar references to ≤ 3e-6, including
non-multiple-of-16 widths (p=37, headDim=40 tail handling).

### Baseline: current TensorPrimitives GEMV path (already AVX-512 via runtime)

| shape            | ms    | GB/s | GFLOPS |
|------------------|-------|------|--------|
| lm_head 151936×896 | 46.30 | 11.8 | 5.9 |
| gate/up 4864×896   |  1.52 | 11.5 | 5.7 |
| qkv/o 896×896      |  0.11 | 29.6 | 14.8 |
| down 896×4864      |  1.64 | 10.6 | 5.3 |

The LM head reads 545 MB/token at 11.8 GB/s — **memory-bandwidth-bound**, not
compute-bound. No AVX-512 GEMV branch can move this; the dot path is already at
the achievable ceiling.

### RoPE forward — no change (below promotion bar)

| measure | scalar | avx512 | speedup |
|---------|--------|--------|---------|
| one rotation (p=32) | 1486 ns | 336 ns | 4.42× |
| per token (24 layers) | 0.571 ms | 0.129 ms | 4.42× (0.13% of 441 ms) |
| prefill (216 tok) | 123.3 ms | 27.9 ms | 4.4× (one-time) |

4.4× on the kernel, but only ~0.13% of per-token time — **below the ≥ ~1%
promotion bar. RoPE stays scalar.**

### Decode-attention V-phase — promotable (A/B confirms ≥2× and ≥1% of token time)

| kvLen | scalar | avx512 | speedup | per token (×24 lyr) | share of 441 ms |
|-------|--------|--------|---------|---------------------|-----------------|
| 216   | 469 µs | 72 µs | 6.49× | 11.26 ms → 1.74 ms | 2.55% → 0.39% |
| 376   | 641 µs | 96 µs | 6.70× | 15.39 ms → 2.30 ms | 3.49% → 0.52% |

The scalar V-phase costs 2.6–3.5% of per-token wall time and grows with cache
length; the AVX-512 d-blocked broadcast-FMA kernel is 6.5–6.7× faster,
recovering ~2–3% of decode time. This is the one branch worth promoting into
`src/Nivara` — done. The float path now uses a three-tier chain: `Vector512`
(AVX-512, gated on `Vector512.IsHardwareAccelerated`) → `Vector<float>`
(portable variable-width SIMD — SSE2/AVX2/NEON on non-AVX-512 machines) →
existing scalar loop (non-float `T` / unaccelerated runtimes). Verified by
existing attention/KV-cache/prefill/PyTorch-parity tests.

## Transpose route A/B (`transpose` mode) — issue #482

Question: `Tensor.Transpose<T>` ships in .NET 11 but returns a zero-copy strided
**view**, so the #136 swap route must pay `Tensor` construction + `FlattenTo`
materialization per call. Does that still cost more than the handwritten
cache-tiled kernel in `TensorsHelper.Transpose`?

**Always run this mode with `-c Release`** — see the build-configuration note below.

Design: the unit-test version of this gate was flaky (3/5 failures on a clean
tree, once on a 0.2% margin), so the probe fixes the *measurement*, not the
threshold — warm up both routes, interleave A/B rounds, alternate which route goes
first so first-measured drift cancels, and report a per-shape distribution
(median ratio + win rate) over five shapes instead of one point estimate. It also
reprints the old best-of-5 methodology per shape for contrast.

### Results (Release, X64, .NET 11, Core Ultra 7 255H)

| shape | tiled median | bcl median | median ratio | tiled win rate |
|-------|--------------|------------|--------------|----------------|
| 1024x1024 (the #482 gate shape) | 3.5–4.1 ms | 9.0–9.2 ms | ~0.38 | 30/30 |
| 512x512 | 0.67 ms | 1.12 ms | ~0.59 | 29–30/30 |
| 2048x1024 | 6.0 ms | 21.0 ms | ~0.29 | 30/30 |
| 1024x2048 | 7.5 ms | 41.5 ms | ~0.25 | 30/30 |
| 129x257 (tail handling) | 0.05 ms | 0.41 ms | ~0.12 | 30/30 |

**All shapes: tiled wins ~99–100% of rounds, median ratio ~0.30 — a 2.2–3.4× win.**
The tiled kernel is earning its keep; the #136 swap stays parked.

### Build configuration is the #482 root cause

The gate failed because it was normally run in **Debug** (`CONTRIBUTING.md`
documents plain `dotnet test`, which defaults to Debug). This probe's code is
compiled with whatever configuration it runs in — but `Tensor.Transpose` and
`FlattenTo` ship **ReadyToRun** and stay optimized regardless. In Debug that
cancels the tiled kernel's entire advantage:

| Build | tiled | bcl | ratio | outcome |
|-------|-------|-----|-------|---------|
| Release | 3.5 ms | 9.1 ms | **0.35–0.51** | 15/15 PASS |
| Debug   | ~10 ms | ~10 ms | **0.80–1.24** | coin-flips across 1.0 |

In Debug the two routes sit at **parity**, so a sub-1%-margin ordering assertion
fails roughly half the time — reproducing the issue's 3/5 failures exactly. The
kernel never regressed; the comparison was void. The probe now prints a
build-configuration check and refuses to report a Debug "tiled lost" reading as a
failure.

### Hypotheses tested and ruled out

1. **Machine contention** — under a 16-thread memory-pressure generator the
   interleaved probe still reported tiled winning 149/150 rounds (median ratio
   0.220). Contention *widens* the gap.
2. **JIT tier-0 / missing warmup** — per-call timings over the first 40 calls show
   the tier-0 penalty is ~2.1x on tiled and ~2.45x on bcl; it hits both routes, so
   the ratio is unchanged. Warmup is still correct, but it is not the cause.
3. **Generic-vs-concrete dispatch** — `Transpose<T> where T : struct, INumber<T>`
   measured identical to a concrete `float` specialization (4.13 ms vs 4.11 ms).

## `Tensor<T>` row extraction (`tensor-api` mode) — issue #528

The other modes answer performance questions. This one is a **correctness probe for
the docs**: it exists so a documentation fix never has to guess at a BCL API, and
never gets certified by a gate that only compiles.

```bash
dotnet run -c Release --project tests/Nivara.SimdProbe -- tensor-api
```

It has no SIMD content at all — no timing, no Release requirement (it passes in
Debug, and says so, because nothing it measures is affected by optimization). It
is self-contained, referencing only `System.Numerics.Tensors`, because the subject
is the BCL contract the docs are written against.

**Why a probe at all, for a two-line doc fix.** Because the obvious fixes are all
wrong in ways a compiling gate cannot see:

| Candidate | Compiles | Runs |
|---|---|---|
| `tensor.Span.Slice(i * dims, dims)` — what #528 prescribed | no, `CS1061` | — |
| `tensor.GetSpan([i], dims)` | **yes** | **no — `ArgumentOutOfRangeException`** |
| `tensor.GetSpan([i, 0], dims)` | yes | yes |

`DocumentationSnippetTests` compiles; it does not run. So the middle row — which
looks correct, reads correct, and passes the gate — throws on the first row of the
first document. That is the failure this mode exists to make visible, and it asserts
the third row's output against the flat row-major slice exactly, with no tolerance
band, because a wrong row offset is structural rather than a precision question.

**What this table is and is not evidence for.** The "compiles" column for the two rows
that fail to compile comes from compiling them, in a scratch project, at the time the
issue was analysed — this mode cannot re-derive it, because a form that does not compile
cannot appear in a file that must itself compile. Those two rows are covered reflectively
instead (step 2 of the probe below). Everything the mode reports about *running* is
re-verified on every invocation. Likewise the `CS0121` / `CS9174` diagnostics quoted for
`[]` and `Slice([i], ..)` are from that same compile check, not from this mode.

**What it does**

1. Reflects and prints the resolved `Tensor<T>` public surface, with generic
   arguments expanded — `Span<Single> GetSpan(ReadOnlySpan<nint>, Int32)` and the
   `ReadOnlySpan<NIndex>` twin are otherwise both `ReadOnlySpan\`1` and
   indistinguishable.
2. Asserts `Tensor<T>` has **no** public `Span`, `Memory`, `AsSpan` or `AsMemory`.
   Issue #528 said "the property is `.Span`"; that does not exist. The `CS1929` the
   gate reports is the compiler having resolved `AsSpan` to
   `MemoryExtensions.AsSpan(string?)`. This check fails loudly if a future release
   adds one of those members and the docs should be revisited.
3. Runs every candidate form that *can* compile, comparing each row against the flat
   row-major slice. The non-compiling forms cannot appear in a file that must itself
   compile, so their status is covered by step 2 instead. The tensor is built from a
   **clone** of the reference array, because `Tensor.Create` aliases its input rather
   than copying — verified, not assumed. Comparing an extracted row against an array
   the tensor also points into would be correct by construction, and nothing would
   demonstrate the comparison discriminates. A negative control therefore asserts that
   a deliberately wrong row is rejected, so a passing row check means something.
4. Runs both `EXAMPLES.md` scoring snippets end to end on the Act 4 dataset and
   asserts the descending ranking is `doc-101, doc-103`, which is what the document
   claims. The numpy values are printed alongside for a human to compare — they are
   not asserted, because a float tolerance would let a real bug hide behind it.

## `Tensor.Create` aliases its input

Found while building this mode, and it is a trap for any zero-copy reasoning: passing a
`float[]` to `Tensor.Create` does **not** snapshot it. Mutating the array afterwards
changes the tensor, and writing through the tensor changes the array — in both
directions. So a "reference" array that the tensor also points into is not a reference.

## Tensor<T> findings (`tensor-api` mode)

- **`Tensor<T>` has no `Span`/`Memory`/`AsSpan()`.** To read one row of a rank-2
  tensor, use `GetSpan(startIndexes, length)`.
- **`GetSpan` takes one index per dimension.** `GetSpan([i], dims)` on a `[rows, cols]`
  tensor compiles and throws. Use `GetSpan([i, 0], dims)`.
- **`GetSpan`'s length parameter is `int`, not `nint`**, while `Tensor.Lengths` is
  `ReadOnlySpan<nint>` — so the natural `nint dims = tensor.Lengths[1]` is `CS1503`.
- **`nint` vs `NIndex` is a real ambiguity.** `GetSpan` is overloaded on both and
  `int` converts to each; a collection expression (`[i, 0]`) resolves, while `[]` and
  `Slice([i], ..)` do not.

**Use this mode when a doc names a `Tensor<T>` member that does not compile.**
Issues #524–#532 are nine more of the same class, and the recurring finding is that
the filed prescription is not the fix. Run the mode and read the surface rather than
implementing an issue body verbatim. See `docs/TODO.md`.

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
- `ScalarKernelProbe.cs` (`scalar` mode) — Qwen decode scalar hot-path probe:
  RoPE + attention V-phase vs hand-rolled `Vector512`, plus TensorPrimitives GEMV
  baseline.
- `TransposeKernelProbe.cs` (`transpose` mode) — #136 A/B of the cache-tiled
  `TensorsHelper.Transpose` kernel against the BCL view + `FlattenTo` route,
  interleaved with alternating order, plus the build-configuration check that
  explains the #482 flakiness. Run with `-c Release`.
- `TensorApiProbe.cs` (`tensor-api` mode) — reflects the real `Tensor<T>` surface
  and asserts that it has no `Span`/`Memory`/`AsSpan`, then runs every row-extraction
  form that *can* compile against the flat row-major slice and against the `EXAMPLES.md`
  ranking. A correctness probe for doc fixes, not a benchmark — run it before
  implementing any filed issue that names a `Tensor<T>` member.
- `Program.cs` — CLI entry (`support` / `correctness` / `benchmark` / `scalar` /
  `transpose` / `tensor-api` / `all`).
