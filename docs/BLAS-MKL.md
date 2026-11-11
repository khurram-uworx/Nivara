# OpenBLAS / oneMKL — options and the measurement-first plan

> **Status update (2026-11-12):** the seam shipped, opt-in via `--blas`. `--threads` is the
> Nivara row-block count and `--blas-threads` is `openblas_set_num_threads`; both optional,
> default 1. Part 2 non-GEMM routines are not bound; `IGemvProvider` waits on a measured
> GEMV gap. The sections below are the 2026-11-10 probe record and are not rewritten.
>
> **Status: probe run 2026-11-10.** No package reference. The probe cleared its gate
> (`maxAbs <= 1e-3` on every row, all four runs) and the recommendation is recorded below.
> This document remains the record of that measurement.

This page is the CPU-BLAS counterpart of the GPU story in
[ACCELERATION.md](ACCELERATION.md). The earlier capture of the same three options (managed
tiled GEMM, native bridge, hybrid) is [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) §3–§5.
Where that doc and this one disagree about the dispatch point, this one is the later reading
of the code — see §2.

---

## 1. The question

PyTorch's CPU matmul is MKL-class. Nivara's is a managed kernel. The published gap is
~6.1–6.5× on the BERT-scale forwards in `samples/NivaraInference/README.md` (issue
[#458](https://github.com/khurram-uworx/Nivara/issues/458)). The question is not "can we
call OpenBLAS or MKL" — that is straightforward — it is whether a native BLAS is the thing
that closes the gap, and whether it beats a dependency-free managed kernel at the shapes we
actually ship.

Two libraries, same C ABI:

| | OpenBLAS | Intel oneMKL |
|---|---|---|
| Licence | BSD-3 | Intel Simplified Software License, redistributable |
| What it wins on | AVX/AVX-512, its own thread pool, per-shape tuned kernels | same, plus Intel-specific kernel selection; this is what PyTorch's CPU wheel calls |
| Cost | native DLL per RID; smaller than MKL | native DLL per RID; the redistributable is large |
| Not a fit for | Half, BFloat16, null masks, transpose-as-a-copy | same |

oneMKL's GPU surface (SYCL-only) was already dismissed for the iGPU path — see
[`tests/Nivara.GpuProbe/OPENVINO.md`](../tests/Nivara.GpuProbe/OPENVINO.md). This page is
CPU BLAS only.

## 2. Where a bridge would plug in

Every dense matmul in the repo already funnels through one internal method. A seam there
covers inference, AutoDiff training, and attention. Nothing else needs a call-site edit.

| call | file | what it does |
|---|---|---|
| `TensorsHelper.MultiplyCore<T>` | `src/Nivara/Tensors/TensorsHelper.cs` | the kernel. `bTransposed` selects `A·B` vs `A·Bᵀ`. `aRows == 1` is a GEMV fast path (no rent, no transpose). float/double/generic dispatch; `Parallel.For` behind `ShouldParallelize` |
| `GradKernels.MatMul` / `MatMulTransposedB` | `src/Nivara/AutoDiff/Operations/GradKernels.cs` | one-line wrappers over `MultiplyCore` |
| `Linear<T>.Forward` | `src/Nivara/AutoDiff/Nn/Linear.cs` | `ReverseGradOperations.MatMulTransposedB` — weight is `[out, in]`, so every `Linear<T>` is the transposed-B form |
| BERT / ModernBERT / Laya | `src/Nivara.Extensions/AI/` | built from `Linear<T>` plus `ReverseGradOperations`; attention calls `MatMulTransposedB` directly |

`TensorsHelper` already names the swap target: `Tensor.MatrixMultiply` has not shipped
(dotnet/runtime#95863, BLAS epic dotnet/runtime#93286, noted at the top of that file). There
is no BCL GEMM to delegate to. `TensorPrimitives.Dot` is the vector dot product, and it is
what the row kernel uses today.

**Not the dispatch point.** [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) §4 names
`KernelSelector` as the natural place to choose managed-SIMD vs native. `KernelSelector`
(`src/Nivara/KernelSelector.cs`) chooses scalar vs vectorized for elementwise column ops. It
does not see a matmul. Do not route a BLAS provider through it.

**Stays managed even if a bridge lands.** `Transpose<T>` (same file — BLAS has no
"materialize a transpose" routine; the in-tree tiled copy is the physical transpose).
Null-masked multiply. Half and BFloat16 (`MultiplyCoreGeneric` / `WidenPrimitives`). The
`aRows == 1` GEMV fast path is a separate decision: `cblas_sgemv`, or `cblas_sgemm` with
`M = 1`. The probe in §5 does not exercise it — no shape in `GemmBenchmark.s_shapes` has
`M = 1`.

## 3. The options

### A. Managed tiled GEMM — no native dependency

Port the register-blocked pattern (the GPU Row4 idea, on the CPU with `Vector512<T>` or
`Vector<float>`) into `MultiplyCore`. Zero dependencies, portable, benefits every consumer.

The probe already contains this as `Leg.Blocked` (`CpuGemmProbe.BlockedGemm`): `Parallel.For`
over disjoint output rows, four output columns held in `Vector<float>` across the K loop, A
and B read in place. At Laya's S=512 shapes it **lost** to the in-tree kernel (2529 ms vs
2414 ms projected forward, [#458](https://github.com/khurram-uworx/Nivara/issues/458)). At
the S=128 shapes the published benchmarks use, the same comparison went the other way
(6–10×, one run, not yet re-measured with enough rounds to trust). So "managed is enough"
and "managed cannot close the small-M gap" are both still open, and they are different
answers at different `M`.

### B. Direct P/Invoke of the CBLAS ABI

`[LibraryImport]` (not `[DllImport]` — source-generated, AOT/trim friendly; see
[source generation for platform invokes](https://learn.microsoft.com/dotnet/standard/native-interop/pinvoke-source-generation))
on `cblas_sgemm` / `cblas_dgemm`, plus `cblas_sgemv` if the single-row path is worth it.
One binding serves both libraries: `NativeLibrary.SetDllImportResolver` /
`NativeLibrary.TryLoad` picks the DLL
([native library loading](https://learn.microsoft.com/dotnet/standard/native-interop/native-library-loading)).

CBLAS is row-major-capable, so `MultiplyCore`'s two layouts map without a copy:

| Nivara call | CBLAS |
|---|---|
| `MultiplyCore(..., bTransposed: true)` — B is already `[N x K]` | `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M=aRows, N=bCols, K=aCols, 1, A, lda=K, B, ldb=K, 0, C, ldc=N)` |
| `MultiplyCore(..., bTransposed: false)` — B is `[K x N]` | same, with `TransB = CblasTrans` and `ldb = N` |

`CblasRowMajor = 101`, `CblasNoTrans = 111`, `CblasTrans = 112` (netlib `cblas.h`). Getting
these wrong is a silent numeric failure, not a crash — the probe gate in §5 is what catches
it.

Host the native implementation in `Nivara.Extensions`, never in core. Core stays
dependency-free; Extensions is where Silk, Arrow, Parquet, and ML.NET already live.

### C. Wrap a package instead of writing the P/Invoke

`MKL.NET` (+ a RID runtime package), `MathNet.Numerics` with `MathNet.Numerics.MKL.Win-*`,
or `OpenBLAS.NET`. Less binding risk, at the cost of a dependency and of crossing a
span/array boundary the package may not accept as a span. Same deployment and licensing
questions as B.

### D. Hybrid — managed default, native opt-in

A is the default. B or C is selected through a provider registered from Extensions, the same
shape as `IExecutionStrategy` and as the GPU seam in `src/Nivara.Extensions/AI/GpuSeam.cs`.
This is the contract [issue #444](https://github.com/khurram-uworx/Nivara/issues/444) already
asks for on the GPU side (matmul + optimizer steps, "adopt whatever
[#133](https://github.com/khurram-uworx/Nivara/issues/133) resolves"). A CPU BLAS provider is
a second consumer of that seam, not a second seam. Do not invent the contract in a BLAS
change — [#133](https://github.com/khurram-uworx/Nivara/issues/133) is the decision, and it
is still open.

### E. Wait for the BCL

`Tensor.MatrixMultiply` is the named swap target in `TensorsHelper` and it has not shipped.
No date. Not a plan.

### Comparison

| option | closes a measured gap? | new dependency | when it is the right call |
|---|---|---|---|
| A. managed tiled | unknown at S=128; measured no at S=512 | none | if A captures the small-M win that [#458](https://github.com/khurram-uworx/Nivara/issues/458) saw |
| B. P/Invoke CBLAS | the thing the probe in §5 exists to answer | native DLL, opt-in | if B beats A by enough to pay for the DLL |
| C. package wrapper | same as B | NuGet + native DLL | if we would rather not own the binding |
| D. hybrid | whichever of A/B wins, selectable | native, opt-in | if B wins and we still refuse to make it the default |
| E. wait | no | none | not a plan |

## 4. Why the probe comes before the dependency

[#458](https://github.com/khurram-uworx/Nivara/issues/458) already ran the managed half of
this comparison, via `--cpu-gemm` in `tests/Nivara.PerformanceTests`. Two results, and they
do not transfer to each other:

- **S=512 (Laya).** In-tree GEMM sustains 66–80 GMAC/s. The blocked managed reference did
  not beat it. End-to-end throughput is ~3× below the GEMM rate, so a large share of the
  ~6× PyTorch gap is **not GEMM**. A BLAS bridge cannot close that share.
- **S=128 (DistilBERT / MiniLM — every published transformer number).** One run showed the
  in-tree path 6–10× behind the blocked reference on the two biggest rows (`distilbert qkv/o`
  12.3 ms vs 1.2 ms, `distilbert fc1` 40.0 ms vs 6.4 ms). Single rows vary up to 6.5× run
  to run; the roll-up is the trustworthy number, the row is not. Re-measure before treating
  it as stable.

[ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) §5 already set the gate this page is executing:
go native only if the managed kernel measures well below MKL on the shapes we ship. That
measurement has not been run. Until it has, adding a native dependency is a guess.

A BLAS bridge also does **not** fix [#457](https://github.com/khurram-uworx/Nivara/issues/457).
The vision gap is convolution (naive nested loops, no SIMD, no parallelism), and convolution
does not go through `MultiplyCore`.

## 5. Decision as recorded (2026-10-15)

The decision recorded before the probe ran. Items 2 and 3 are what the probe
changed: see §5.1.

1. **Probe, not a seam.** Add a BLAS leg to the existing `--cpu-gemm` probe
   (`tests/Nivara.PerformanceTests/CpuGemmProbe.cs`). The P/Invoke stays in that project.
   `src/` does not change. This is the same rule #444 states for the GPU seam: introduce the
   contract when the consumer lands, not before.
2. **Probe-only package.** Reference a win-x64 MKL runtime package (`MKL.NET.win-x64`, or
   `MathNet.Numerics.MKL.Win-x64` if its binding is the one that actually exports
   `cblas_sgemm`) from `Nivara.PerformanceTests.csproj` only, `PrivateAssets="all"`. Confirm
   the binding surface against the package before writing the call — do not assume the
   managed wrapper's method names. Condition the reference so a non-Windows restore of that
   project still succeeds; the leg prints `not available` and is skipped, never a silent
   pass and never a failed build.
3. **One library answers the decision; it does not rank the two.** MKL is the probe vehicle
   because it is what the PyTorch number is. A win says "tuned native BLAS beats us". It
   does not say OpenBLAS would, or would not. Swapping the DLL behind the same `cblas_sgemm`
   binding is a second run, not a second design.
4. **If it clears, the seam covers everything `MultiplyCore` covers** — inference, training,
   attention. Not a model-layer-only hook. The contract is the [#133](https://github.com/khurram-uworx/Nivara/issues/133)
   / [#444](https://github.com/khurram-uworx/Nivara/issues/444) provider, not a BLAS-specific
   one.

## 5.1. What the probe actually found (2026-11-10)

### Where the DLL comes from

- **Source.** The official OpenBLAS GitHub release
  [`v0.3.34`](https://github.com/xianyi/OpenBLAS/releases/tag/v0.3.34), asset
  `OpenBLAS-0.3.34-x64.zip` (39 MB). It contains a single `libopenblas.dll` (49 MB).
- **Why this and not MKL.** MKL is 481 MB on disk and ships no official NuGet; OpenBLAS is
  49 MB and its import table is `KERNEL32.dll` + `msvcrt.dll` only (no gcc/gfortran/quadmath/
  winpthread). Both answer the same question; the lean one is the candidate. This probe does
  not measure MKL and does not claim to.
- **How to get it.** Download the zip, extract `libopenblas.dll` to `lib/libopenblas.dll`
  at the repository root (that directory is gitignored). That is the only location the probe
  looks in. Nothing is committed to the repository and nothing is restored.
- **Recommendation (acquisition).** If the seam lands, ship the DLL as an opt-in native asset in
  `Nivara.Extensions` (RID-specific, default off), never in core. Core stays dependency-free.

The probe ran on the i5-1135G7 laptop (4C/8T, AVX-512 enabled, 31.7 GB RAM,
.NET 11.0.100-rc.3, AC power, Balanced plan), against OpenBLAS 0.3.34
(`DYNAMIC_ARCH NO_AFFINITY MAX_THREADS=64`, LP64, imports `KERNEL32.dll` +
`msvcrt.dll` only). Four runs, all `Gate PASS` on every row, all against the
same double-precision truth the other legs use.

### Measured table — projected Laya forward, GEMM only (total ms / GMAC/s)

| leg | SkylakeX (AVX-512) | GMAC/s | Haswell (AVX2) | GMAC/s |
|---|---|---|---|---|---|
| AutoDiff (managed) | 14286 | 13 | 14147 | 13 |
| BTranspose (managed) | 16889 | 11 | 16545 | 11 |
| Blocked (managed) | 14788 | 12 | 14364 | 13 |
| Blas 1t | 8988 | 21 | 11038 | 17 |
| Blas 1t-NT | 9368 | 20 | 11224 | 16 |
| Blas Nt | 2191 | 84 | 2612 | 71 |
| Blas hybrid | 2112 | 87 | 2358 | 78 |

Run-to-run spread for the projection: AutoDiff 11238-15018 ms, Blas 1t 4851-11038 ms,
Blas 8t 2178-4838 ms, Blas hybrid 1878-2261 ms. The hybrid's ratio to the managed kernel is
the stable quantity (6.0-6.8x); the absolute GMAC/s figures are not.

### What changed relative to the plan

1. **MKL was not measured and is not needed.** §5.3's assumption that the probe vehicle
   had to be MKL (because that is what the PyTorch number is) does not survive contact with
   the leanness requirement: MKL is 481 MB on disk (82.8 MB rt+core, 245 MB kernel variants,
   84 MB VML, 69.2 MB thread layers) and ships no official NuGet. OpenBLAS 0.3.34 is a
   single 49 MB DLL whose import table is `KERNEL32.dll` + `msvcrt.dll` only. Both answer
   the same question — "how fast is a tuned native BLAS here" — and the lean one is the
   candidate. The write-up therefore cannot claim "MKL measured on this box"; it says so.
2. **No package reference at all.** §5.2's `MKL.NET.win-x64` / `MathNet.Numerics.MKL.Win-*`
   assumption is gone. The DLL is resolved at runtime from `lib/libopenblas.dll`; there is no
   build-time gating, no environment variable, no copy step. Default builds restore and
   define nothing extra.
3. **OpenBLAS needs no preload dance.** `openblas_set_num_threads(1)` works before any GEMM
   (unlike MKL, whose `mkl_set_num_threads` / `mkl_set_threading_layer` crash before init).
   Setting the thread count through the CRT (`putenv`) is silently ignored; a real process
   environment variable flips `openblas_get_corename()` from `SkylakeX` to `Haswell`.
4. **The CBLAS mapping differs from §3B.** That table has the two layouts swapped relative to
   the binding that actually ran. The binding in `OpenBlasInterop.cs` uses:
   - B as `[N x K]` (the `MatMulTransposedB` layout): `TransB = Trans`, `ldb = K`
   - B as `[K x N]` (the `MatMul` layout): `TransB = NoTrans`, `ldb = N`
   `RowMajor = 101`, `NoTrans = 111`, `Trans = 112`. The probe gate caught the swap; the
   doc's §3B table is left as the earlier reading and is superseded by this one.
5. **The driver-BLAS property does not transfer to CPU BLAS.** Intel graphics drivers ship
   OpenCL, Level Zero and oneVPL and no MKL, so "MKL is already on every machine like
   OpenCL" is false. Torch's MKL is statically linked into `torch_cpu.dll` (290.9 MB,
   59,776 C++-mangled ATen exports; `hasattr(dll, "cblas_sgemm")` is False) and is not
   hookable. Neither is a usable acquisition vehicle.
6. **The 8-thread leg is not a candidate.** Its `min` and `median` disagree by up to 30x
   because waking OpenBLAS's pool costs more than the GEMM at M=128. `Blas hybrid` (Nivara's
   `Parallel.For` over row blocks, one OpenBLAS thread per call) matched or beat it in 3 of
   4 runs and never had a bad one, so it is the integration shape. See §8.
7. **AVX-512 buys about 10-15%.** Haswell is 78 GMAC/s against SkylakeX's 82-87 on the
   hybrid. The AVX2 fallback is usable; the AVX-512-transference worry largely dissolves.

### Findings

1. **Native BLAS closes the gap.** Hybrid is 6.0-6.8x over the managed kernel on the
   GEMM-only forward, which is the size of the #458 deficit (~5.6x against PyTorch CPU).
   Single-threaded native is 1.6-2.4x, so even the conservative configuration wins.
2. **`Blocked` still does not beat the in-tree kernel** on this box (14788 vs 14286 ms), so
   the "MultiplyRowFloat's one-Dot-per-element is the bottleneck" hypothesis remains
   falsified. The small-M win is not available to a managed kernel.
3. **The 8-thread leg is erratic, not slow.** It is fast when it is fast (84-85 GMAC/s
   projected) and collapses to 38-45 when the process is not continuously busy. Its `min`
   stays fast in every run, which is the signature of wake-up latency, not of a slow kernel.

### Recommendation

Adopt the seam, with the **hybrid** configuration as the default behaviour of the provider:
Nivara owns the row parallelism, OpenBLAS is called single-threaded from each chunk. The
provider is off by default (Extensions, opt-in), and the managed path remains the default and
bit-identical fallback. `Blas 1t` is the conservative configuration for callers who do not
want a second pool in their process at all.

### Decision rule, applied

| result | do |
|---|---|
| BLAS clearly ahead of `Blocked` at both M=128 and M=512 | file the seam issue. Scope is §7. One commit records the probe table on [#458](https://github.com/khurram-uworx/Nivara/issues/458) first, so the issue exists before the design |
| BLAS ≈ `Blocked`, or the win is M=128 only and `Blocked` already has it | no native dependency. The small-M fix is a managed-kernel item under [#458](https://github.com/khurram-uworx/Nivara/issues/458) |
| BLAS ≈ in-tree at both | no GEMM change. [#458](https://github.com/khurram-uworx/Nivara/issues/458)'s non-GEMM lead stands |

The hybrid is clearly ahead of `Blocked` at both regimes (2112 vs 14788 ms at M=512, and
1.46 ms vs 7.02 ms on `distilbert qkv/o` at M=128), and the 8-thread leg's own pool is the
reason it is not the configuration to ship. **First row applies.** The seam issue is to be
filed against [#458](https://github.com/khurram-uworx/Nivara/issues/458) before any design work.

"Clearly ahead" means ahead of the noise, not ahead of zero. The roll-up moves ~2% run to
run for the in-tree legs and ~9% for `Blocked`. The hybrid's 6.0-6.8x is not in that noise.

---

## 6. The probe (as run)

The plan as written assumed MKL as the vehicle and a single `Blas` leg. The probe that
actually ran is documented in §5.1 and `tests/Nivara.PerformanceTests/README.md`. Deviations
from the plan below:

- **Vehicle.** OpenBLAS 0.3.34, not MKL. No package reference; the DLL is resolved at
  runtime from `lib/libopenblas.dll` (see `OpenBlasInterop.CandidatePaths`). See �5.1.
- **Legs.** Four BLAS legs, not one: `Blas 1t`, `Blas 1t-NT`, `Blas Nt`, `Blas hybrid`.
  See `CpuGemmProbe.cs` and the README leg table.
- **Protocol.** The probe already used 1 untimed warmup and 20 interleaved rounds (the README's
  "5 untimed warmups" line was stale and is corrected). The BLAS legs report `min ms` alongside
  `median ms`, because `Blas Nt`'s min and median diverge by up to 30x.
- **AVX2 control.** `OPENBLAS_CORETYPE=Haswell` as a real process env var, confirmed by
  `corename: Haswell` in the header. No new CLI flag.
- **LP64 assertion.** `TryInitialize` rejects an ILP64 build by searching for `USE64BITINT`
  in `openblas_get_config()`.

The original plan text below is retained as the record of intent.

```bash
dotnet run -c Release --project tests/Nivara.PerformanceTests -- --cpu-gemm
```

Release only. Debug compares an unoptimized Nivara against an optimized native library and
reports a false gap. AC power — the iGPU caveat does not apply (this probe is CPU), but the
same-session rule does: do not compare a BLAS number from this run with a PyTorch number
from another.

**Fourth leg, three comparators.** `AutoDiff`, `BTranspose`, and `Blocked` already exist and
are gated `maxAbs(leg − double-precision truth) <= 1e-3`. Add `Blas` beside them. The
decision column is **BLAS vs `Blocked`**, not BLAS vs `AutoDiff`. `Blocked` is the
dependency-free alternative; if it captures the small-M win, the DLL is not worth it.

**Call mapping.** The probe already builds `bt` as `[N x K]` for the `AutoDiff` leg and
keeps `b` as `[K x N]` for `BTranspose`. BLAS consumes those two buffers as the two rows of
the table in §3. No extra transpose on the BLAS side — a transpose staged for the native
call would be measuring the copy.

**Threads.** Print the thread count on every row. Sweep 1 (like-for-like against a
single-threaded kernel) and `Environment.ProcessorCount` (against the threaded in-tree
path). Set it explicitly — `mkl_set_num_threads`, or the package's wrapper around it. Left
unset, MKL starts its own pool and the process oversubscribes against `Parallel.For` and
against the query engine. That oversubscription is the production hazard too; the probe is
where it gets a number.

**Protocol.** The probe's current best-of-5, each leg timed on its own, is the #482
flakiness pattern (best-of-N taken separately per route). Use it as a screening pass. If
BLAS lands close to either in-tree leg, re-measure with interleaved rounds and compare
medians before writing the conclusion down. Read the projected roll-up, not a single row —
[#458](https://github.com/khurram-uworx/Nivara/issues/458) recorded 6.5× run-to-run swing on
`distilbert fc1`.

**Shapes that matter.** `GemmBenchmark.s_shapes` already has both regimes: DistilBERT and
MiniLM at M=128, Laya at M=512, plus `laya qkv@128` as the S=128 control at Laya's width.
The edge rows (`edge padded rows`, `edge 2x2 K=17`, …) are there so a wrong `lda` fails the
gate instead of passing on a multiple of the tile.

**Gate.** Correctness is the pass/fail: `maxAbs <= 1e-3` against the same
double-precision truth the other legs use. The milliseconds are evidence, not a threshold —
this mode is a diagnostic and stays out of `--json` / `--compare`. Capture the process exit
status, not a filtered pipeline's.

### Decision rule

| result | do |
|---|---|
| BLAS clearly ahead of `Blocked` at both M=128 and M=512 | file the seam issue. Scope is §7. One commit records the probe table on [#458](https://github.com/khurram-uworx/Nivara/issues/458) first, so the issue exists before the design |
| BLAS ≈ `Blocked`, or the win is M=128 only and `Blocked` already has it | no native dependency. The small-M fix is a managed-kernel item under [#458](https://github.com/khurram-uworx/Nivara/issues/458) |
| BLAS ≈ in-tree at both | no GEMM change. [#458](https://github.com/khurram-uworx/Nivara/issues/458)'s non-GEMM lead stands |

"Clearly ahead" means ahead of the noise, not ahead of zero. The roll-up moves ~2% run to
run for the in-tree legs and ~9% for `Blocked`. A 10% win is not a dependency.

## 7. If it clears — the seam, not built here

Recorded so the probe's successor does not re-derive it. Not authorized by this page.

- **Core.** A zero-dependency provider slot that `MultiplyCore` consults for dense float and
  double only. Absent a provider, today's path runs, bit-identical. Mirrors
  `IExecutionStrategy`: interface in core, implementation elsewhere. The shape of the
  interface is [#133](https://github.com/khurram-uworx/Nivara/issues/133)'s decision, shared
  with [#444](https://github.com/khurram-uworx/Nivara/issues/444) — do not fork a BLAS-only
  contract if that issue has resolved by then.
- **Extensions.** The cblas backend. `[LibraryImport]`, `TryLoad`, RID-specific native
  asset. Default off. Thread count is a knob, defaulting to 1 inside a call that may
  already be running under `Parallel.For` or the query engine — oversubscription is a
  behaviour bug, not a tuning preference.
  **Binding note.** `LibraryImportAttribute` has no `CallingConvention` property:
  source-generated P/Invoke always uses the platform default. That is correct on x64
  Windows (one convention) and wrong on x86, so this binding is x64-only. If an x86 target
  is ever wanted, use `[DllImport]` with `CallingConvention.Cdecl`. Also, with no
  `DefaultDllImportSearchPaths` flag the runtime searches the assembly directory and then
  falls back to the OS, which is what makes the "copy the DLL next to the executable"
  acquisition work without a resolver; specifying `DllImportSearchPath.AssemblyDirectory`
  would both drop that fallback (.NET 10+) and trip CA5393.
- **Stays on the managed path.** Anything that is not dense float or double: transpose,
  null mask, Half, BFloat16, and the integer types `MultiplyCore` still accepts. Falling
  through to managed for those is the correct result, not a gap.
- **Integer width.** LP64 (`MKL_INT` = 32-bit `int`) is the Windows CBLAS default. ILP64
  (`long long`) is a different DLL and a different entry point. Pin one, assert it, do not
  discover it from a wrong answer.
- **Gate.** Managed-vs-native parity on `s_shapes` at the same `maxAbs <= 1e-3` bound, plus
  the GMAC/s table. A tolerance here is a judgement call; the probe's double-precision
  truth is the reference, and both legs are judged against it rather than against each
  other.

## 8. Constraints that apply either way

- **Threading.** OpenBLAS and MKL own a pool. Nivara already uses `Parallel.For` inside
  `MultiplyCore` and in the query engine. Two pools on one machine oversubscribe. Any
  native call sets its own thread count; it does not inherit "whatever the library likes".
- **Layout.** Nivara is row-major. Column-major BLAS (`CblasColMajor`, or a Fortran `sgemm`
  without the CBLAS wrapper) silently transposes the problem. Use the CBLAS layout argument.
  Do not transpose on the way in to satisfy a column-major entry point — that copy is a
  large part of what [#458](https://github.com/khurram-uworx/Nivara/issues/458) measured
  between `AutoDiff` and `BTranspose` (2414 vs 2791 ms projected, 1.16×).
- **Numerics.** A tuned GEMM accumulates in a different order. Bit-identical to
  `TensorPrimitives.Dot` is not the contract. The contract is the double-precision truth
  gate the probe already runs. Do not tighten it to byte-identity against the managed
  kernel, and do not loosen it because "native is the reference".
- **Packaging.** A native asset lives with the component that calls it. Core never gains a
  `PackageReference` on MKL or OpenBLAS. A win-x64 package that restore pulls on every OS
  is a broken CI change, not a probe.
- **Licensing.** Both are permissive enough to redistribute. oneMKL's redistributable is
  large enough that "opt-in, Extensions or the probe, never core" is the policy even after
  a win. Record the licence in the package notes if a shipped package ever carries it.
- **Acquisition (measured 2026-11-10).** No package reference anywhere. OpenBLAS 0.3.34 is a
  single 49 MB DLL whose imports are `KERNEL32.dll` + `msvcrt.dll` only; MKL is 481 MB on
  disk (82.8 rt+core, 245 kernel variants, 84 VML, 69.2 thread layers) and ships no official
  NuGet. Driver-shipped Intel graphics stacks carry OpenCL/Level Zero/oneVPL and no MKL.
  Torch's MKL is statically linked and unexported. The binding is seven hand-written
  `LibraryImport` declarations (not a generator pass over `cblas.h`), LP64 only, with an
  ILP64 guard in `TryInitialize`.
- **Threading (measured).** OpenBLAS's own pool is the hazard, not the kernel: its `min` and
  `median` disagree by up to 30x at M=128 because waking the pool costs more than the GEMM.
  The integration shape is the hybrid — Nivara's `Parallel.For` over row blocks, one
  OpenBLAS thread per call — which matched or beat the 8-thread leg in 3 of 4 runs and never
  had a bad one. Default the provider's thread count to 1.
- **AVX2 (measured).** Haswell is 78 GMAC/s against SkylakeX's 82-87 on the hybrid. The
  AVX2 fallback is usable; the AVX-512-transference worry largely dissolves for OpenBLAS.

## References

- [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) — options A/B/C and the M3 gate this page executes.
- [ACCELERATION.md](ACCELERATION.md) §7 — the GPU side of the same gap; open work names the native BLAS bridge as the CPU follow-up to [#458](https://github.com/khurram-uworx/Nivara/issues/458).
- [`tests/Nivara.PerformanceTests/README.md`](../tests/Nivara.PerformanceTests/README.md) — `--cpu-gemm`: four in-tree legs plus four gated OpenBLAS legs, the projected Laya table (both the #458 box and the 2026-11-10 laptop), and the run-to-run caveat.
- `src/Nivara/Tensors/TensorsHelper.cs` — `MultiplyCore`, and the note that `Tensor.MatrixMultiply` has not shipped.
- Issues: [#458](https://github.com/khurram-uworx/Nivara/issues/458) (the CPU gap; GEMM cleared at S=512, not at S=128) · [#435](https://github.com/khurram-uworx/Nivara/issues/435) (the GEMM harness this probe extends) · [#133](https://github.com/khurram-uworx/Nivara/issues/133) (provider-seam decision, open) · [#444](https://github.com/khurram-uworx/Nivara/issues/444) (GPU consumer of that seam) · [#457](https://github.com/khurram-uworx/Nivara/issues/457) (vision conv — not this bridge).
- [LibraryImport source generation](https://learn.microsoft.com/dotnet/standard/native-interop/pinvoke-source-generation) · [native library loading](https://learn.microsoft.com/dotnet/standard/native-interop/native-library-loading) · [native interop best practices](https://learn.microsoft.com/dotnet/standard/native-interop/best-practices).
