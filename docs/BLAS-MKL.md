# OpenBLAS / oneMKL — options and the measurement-first plan

> **Status: recorded 2026-10-15. Not started.** No `src/` change, no package reference,
> no issue filed. This captures the options, the decision, and the probe that has to run
> before either library becomes a dependency. It graduates into issues and a proper plan
> only if that probe clears the gate in §6.

This page is the CPU-BLAS counterpart of the GPU story in
[SILK.md](SILK.md). The earlier capture of the same three options (managed
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

## 5. Decision (2026-10-15)

Measure first. Do not add a dependency to core or to `Nivara.Extensions`.

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

## 6. The probe

Home: `tests/Nivara.PerformanceTests`, mode `--cpu-gemm`. Document the new leg in that
project's `README.md` in the same change. Run:

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

## References

- [ROADMAP-SUGGESTION.md](ROADMAP-SUGGESTION.md) — options A/B/C and the M3 gate this page executes.
- [SILK.md](SILK.md) — the GPU side of the same gap; §"Not done" names the native BLAS bridge as the CPU follow-up to [#458](https://github.com/khurram-uworx/Nivara/issues/458).
- [`tests/Nivara.PerformanceTests/README.md`](../tests/Nivara.PerformanceTests/README.md) — `--cpu-gemm` as it stands (three legs, the projected Laya table, the run-to-run caveat).
- `src/Nivara/Tensors/TensorsHelper.cs` — `MultiplyCore`, and the note that `Tensor.MatrixMultiply` has not shipped.
- Issues: [#458](https://github.com/khurram-uworx/Nivara/issues/458) (the CPU gap; GEMM cleared at S=512, not at S=128) · [#435](https://github.com/khurram-uworx/Nivara/issues/435) (the GEMM harness this probe extends) · [#133](https://github.com/khurram-uworx/Nivara/issues/133) (provider-seam decision, open) · [#444](https://github.com/khurram-uworx/Nivara/issues/444) (GPU consumer of that seam) · [#457](https://github.com/khurram-uworx/Nivara/issues/457) (vision conv — not this bridge).
- [LibraryImport source generation](https://learn.microsoft.com/dotnet/standard/native-interop/pinvoke-source-generation) · [native library loading](https://learn.microsoft.com/dotnet/standard/native-interop/native-library-loading) · [native interop best practices](https://learn.microsoft.com/dotnet/standard/native-interop/best-practices).
