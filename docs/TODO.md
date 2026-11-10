# TODO — OpenBLAS probe leg (`docs/BLAS-MKL.md` §6)

Plan for branch `khurram/blas-probe`. Probe-only: no `src/` change, no package reference
added to any shipping project.

## Problem

`docs/BLAS-MKL.md` §5–§6 requires one measurement before any native BLAS can become a
dependency: **native BLAS vs the dependency-free managed `Blocked` kernel**, at both the
M=128 regime (DistilBERT / MiniLM, every published transformer number) and the M=512 regime
(Laya), gated on `maxAbs <= 1e-3` against the same double-precision truth the other legs use.

That measurement has never been taken on this machine. The published 66–80 GMAC/s for the
in-tree kernel is from issue #458 on different hardware, so no BLAS-vs-managed verdict exists
for this box, and none transfers from it.

The doc's §5.2 assumes the probe can `PackageReference` a win-x64 MKL runtime. Reconnaissance
on this branch's predecessor established that this is the wrong vehicle for the leanest
option:

- **Intel MKL** is official (`IntelMKL.Redist.win-x64`, latest 2026.2.0.882) but is 170 MB
  compressed / 481 MB on disk, and the package cannot be subset at restore.
- **OpenBLAS has no official NuGet.** Every OpenBLAS package on nuget.org is stale
  (`openblas 0.2.14.1`, 2018) or third-party.
- The **official OpenBLAS 0.3.34 release** (`OpenBLAS-0.3.34-x64.zip`, 39 MB) is a single
  49 MB DLL whose import table is `KERNEL32.dll` + `msvcrt.dll` only — no gcc / gfortran /
  quadmath / winpthread. Exports 9,548 symbols including all seven we need, and
  `openblas_get_config()` reports `DYNAMIC_ARCH ... MAX_THREADS=64` with no `USE64BITINT`,
  i.e. the LP64 int32 ABI.

So the probe must be able to run against a **DLL on disk**, not a package. That is also
leaner: zero restore cost, zero bytes committed.

## Decisions taken (flag for review)

1. **OpenBLAS only; no MKL leg.** MKL works (verified: sequential threading layer, no OpenMP
   needed) but costs 481 MB to answer a question already answered — that a tuned native BLAS
   is fast. The recommendation is about leanness, and the lean candidate is OpenBLAS. The
   consequence is that the write-up cannot claim "MKL measured on this box with this harness";
   it will say so explicitly.
2. **No package reference at all.** The DLL arrives through a build property
   (`-p:OpenBlasDll=<path>`, falling back to the `OPENBLAS_DLL` environment variable, which
   MSBuild imports as a property automatically). `-p:BlasProbe=true` gates both the
   `BLAS_PROBE` constant and the copy-to-output. Default builds and CI restore nothing.
3. **Three BLAS legs, not one.** `Blas 1t` (B as `[N x K]`, the `MatMulTransposedB` layout
   that every `Linear<T>` uses), `Blas Nt` (same layout, `Environment.ProcessorCount`
   threads), and `Blas Nt(NoTrans)` (B as `[K x N]`). The third exists because OpenBLAS can
   penalise a transposed B; if `NoTrans` is materially faster, the seam design changes and we
   need to know before writing the recommendation, not after.
4. **AVX2 control via `OPENBLAS_CORETYPE`**, not a new CLI flag. Verified behaviour: the
   variable is read as a true process environment variable — setting it through the CRT
   (`putenv`) is silently ignored, while a real process env var flips
   `openblas_get_corename()` from `SkylakeX` to `Haswell`. The probe therefore reads it from
   the environment and **prints `openblas_get_config()` + `openblas_get_corename()` on every
   run**, so each measurement self-documents which kernel produced it. No new argument, no
   usage-string churn.
5. **Thread count is set per call.** Legs are interleaved, so the count is set inside each
   BLAS leg's action. Cost is a sub-microsecond call per GEMM and it will be noted in the
   README rather than hidden.

## Proposed changes

### `tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj`

```xml
<PropertyGroup Condition="'$(BlasProbe)' == 'true'">
  <DefineConstants>$(DefineConstants);BLAS_PROBE</DefineConstants>
  <OpenBlasDll Condition="'$(OpenBlasDll)' == ''">$(OPENBLAS_DLL)</OpenBlasDll>
</PropertyGroup>

<ItemGroup Condition="'$(BlasProbe)' == 'true'">
  <None Include="$(OpenBlasDll)" Link="libopenblas.dll" CopyToOutputDirectory="PreserveNewest" Visible="false" />
</ItemGroup>

<Target Name="ValidateBlasProbe" BeforeTargets="PrepareForBuild"
        Condition="'$(BlasProbe)' == 'true' and '$(OpenBlasDll)' == ''">
  <Error Text="BlasProbe=true needs -p:OpenBlasDll=&lt;path to libopenblas.dll&gt; or the OPENBLAS_DLL environment variable." />
</Target>
```

### `tests/Nivara.PerformanceTests/OpenBlasInterop.cs` (new, `#if BLAS_PROBE`)

Seven `[LibraryImport]` declarations, hand-written (per `docs/BLAS-MKL.md` §3B: source
generated, AOT/trim-safe). No codegen: generating from `cblas.h` produces thousands of lines
of dead surface, which is the opposite of the goal. `ref float` rather than `float*` keeps
`AllowUnsafeBlocks` off.

| declaration | why |
|---|---|
| `cblas_sgemm` | the one call the probe needs |
| `openblas_get_config` | assert LP64 (`USE64BITINT` absent) and record the build |
| `openblas_get_corename` | prove which kernel the AVX2 control actually selected |
| `openblas_set_num_threads` / `openblas_get_num_threads` | own the thread count, never inherit the library's |

Plus a `BlasProbe.TryInitialize()` that returns a reason string instead of throwing, so a
missing DLL prints `not available` and skips — never a silent pass, never a failed build.

### `tests/Nivara.PerformanceTests/CpuGemmProbe.cs`

Refactor the fixed four-element `runs` / `results` / `order` arrays into lists built from the
leg set, so a leg can be appended conditionally without touching the timing protocol. The
protocol itself is unchanged: one correctness pass against DP truth, one untimed warmup, 20
interleaved rounds with the first-measured leg alternating, median reported.

BLAS call mapping (no transpose staged for the native call — that would measure the copy):

| leg | CBLAS |
|---|---|
| `Blas 1t` / `Blas Nt` | `cblas_sgemm(RowMajor, NoTrans, Trans, M, N, K, 1, A, lda=K, bt, ldb=K, 0, C, ldc=N)` |
| `Blas Nt(NoTrans)` | `cblas_sgemm(RowMajor, NoTrans, NoTrans, M, N, K, 1, A, lda=K, b, ldb=N, 0, C, ldc=N)` |

`ProjectLayaForward` gains the BLAS legs so the roll-up — the number the decision actually
turns on — includes them.

### `tests/Nivara.PerformanceTests/README.md`

Document the new legs under `--cpu-gemm`: the build-property gate, the `OPENBLAS_DLL`
environment variable, the AVX2 control, the per-call thread-count note, and the LP64
assertion. Also correct the existing stale protocol sentence ("5 untimed warmups") to match
the code (`Warmups = 1`, 20 interleaved rounds) — same class of drift as the "best-of-5"
claim `docs/BLAS-MKL.md` already flags as stale.

### `docs/BLAS-MKL.md`

Update with what is now known rather than assumed: the size breakdown (481 MB MKL = 82.8 rt+core
+ 245 kernel variants + 84 VML + 69.2 thread layers), the driver-BLAS disproof (Intel graphics
drivers ship OpenCL / Level Zero / oneVPL and **no** MKL — the "already on every machine like
OpenCL" property does not transfer to CPU BLAS), torch's MKL being statically linked and
unexported, the official OpenBLAS import-table result, and the measured probe table. Replace
§5.2's package assumption with the build-property acquisition. §6's decision rule is then
applied with real numbers.

## Verification

1. `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release` —
   default (no `BlasProbe`) must succeed and restore nothing extra.
2. Same build with `-p:BlasProbe=true -p:OpenBlasDll=<path>` must succeed and copy
   `libopenblas.dll` next to the exe.
3. `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --cpu-gemm` in both
   configurations; capture the process exit status, not a filtered pipeline's. Default build
   prints the BLAS legs as not available and still exits 0.
4. BLAS run: gate PASS on every row, `openblas_get_config()` shows no `USE64BITINT`,
   corename reported per run.
5. AVX2 control: rerun with `OPENBLAS_CORETYPE=Haswell` as a real process env var; confirm
   corename reports `Haswell` before consuming the numbers.
6. Docs: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
   after the doc edit and after `docs/TODO.md` removal.
7. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits

1. `docs: plan the OpenBLAS probe leg in TODO.md`
2. `perf(probe): add gated OpenBLAS legs to --cpu-gemm`
3. `docs(perf): document the BLAS probe legs and correct the stale protocol note`
4. `docs: record the OpenBLAS probe result and the acquisition analysis`
5. `docs: remove TODO.md — plan executed`

## Blast radius

| file | change | depends on it |
|---|---|---|
| `tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj` | conditional `DefineConstants` + native copy + validation target | nothing; default build path unchanged |
| `tests/Nivara.PerformanceTests/OpenBlasInterop.cs` (new) | 7 P/Invokes, compiled only under `BLAS_PROBE` | `CpuGemmProbe` only |
| `tests/Nivara.PerformanceTests/CpuGemmProbe.cs` | list-based leg set, 3 optional legs, roll-up extension | `--cpu-gemm` mode only; not in `--json` / `--compare` |
| `tests/Nivara.PerformanceTests/README.md` | `--cpu-gemm` section | readers |
| `docs/BLAS-MKL.md` | analysis + measured result | readers; `DocsCoveragePins` counts |
| `tests/Nivara.Tests/Docs/DocsCoveragePins.cs` | citation counts **only if** citations are added to `docs/BLAS-MKL.md` (it carries none today, so any `.cs::` citation also adds a cited document) | `DocsCoverageTests` |

`src/` is untouched. No shipping project gains a reference. The only runtime surface is a
diagnostic mode that already exists and already stays out of the regression gates.

## GitHub issues log

- [ ] none yet — capture deferred work here as it appears
