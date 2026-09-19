# Plan: #435 — promote the deleted tiled-GEMM harness into a lasting regression gate

## Problem

Issue #435: the f32 tiled-GEMM validation harness the DistilBERT GPU work used lived in a
temp dir (`C:\Users\khurram\AppData\Local\Temp\opencode\gemm-measure\`) and was deleted
after use. It validated the committed ILGPU GEMM kernels against a double-precision truth
and measured GMAC/s per shape. Without a lasting gate, a driver/IGC bump or kernel edit can
silently regress correctness or throughput with no signal — the keystone bug this gate must
catch is the Row4 `colBase` indexing bug (maxAbs ~40–77 vs the honest f32-vs-DP floor of
4.4e-5–1.6e-4 at K=768–3072).

Nothing of the harness is in the repo today (`git log --all -- "*gemm*"` is empty; the temp
dir is gone). The committed pieces it measured are `samples/Nivara.Samples/Gpu/`
(`GemmKernels.cs` kernels + `IlgpuRuntime.cs` + public `TiledGemm`/`IlgpuGemmVariant`,
added in 349b481 "samples: add ILGPU runtime + tiled GEMM kernel"). We reconstruct only the
fixture/truth/gate/GMAC-s-reporting layer.

## Scope

- **Tests only** plus one test-friendliness line in the samples csproj. **No `src/Nivara`
  changes**, no samples behavior change.
- Location: `tests/Nivara.PerformanceTests` as a new `--gemm` on-demand mode — its README
  already declares the intent ("reach for them … instead of building a throwaway harness").
  The GpuProbe keeps its own `kernels` gate untouched.
- Kernels gated: **all six committed kernels** — `OneToOne` + `Row4` (the issue's scope)
  **plus** the M2 fused siblings `Row4Bias`, `Row4Gelu`, `Row4Relu`, `Row4Qkv` (user decision,
  2026-09-19).
- No OpenCL GPU → **UNBUILT + documented nonzero exit** (probe convention; driver-bump
  revalidation scripts must not silently pass).
- AC power required for meaningful perf numbers; correctness gate is battery-safe. A loud
  warning is printed on battery (P/Invoke `GetSystemPowerStatus`).

## Proposed changes

### 1. `samples/Nivara.Samples/Nivara.Samples.csproj` — InternalsVisibleTo

Add to the existing `AssemblyAttribute` ItemGroup (currently only `Nivara.Tests`):

```xml
<AssemblyAttribute Include="System.Runtime.CompilerServices.InternalsVisibleTo">
  <_Parameter1>Nivara.PerformanceTests</_Parameter1>
</AssemblyAttribute>
```

Needed so the harness can load the `internal` fused kernel methods and read
`GemmKernels.TileSize`/`BlockCols` (grid math). `TiledGemm`/`IlgpuRuntime`/`IlgpuGemmVariant`
stay public and unchanged.

### 2. `tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj` — ILGPU pins

```xml
<PackageReference Include="ILGPU" Version="1.5.3" />
<PackageReference Include="ILGPU.Algorithms" Version="1.5.3" />
```

Same snapshot pin as samples/probe. The harness touches `ArrayView`/`MemoryBuffer1D` and
`Accelerator.LoadKernel` directly.

### 3. New `tests/Nivara.PerformanceTests/GemmBenchmark.cs`

`internal static class GemmBenchmark` with `public static int Run(string[] args)`. Mirrors
the `SafeTensorsLoadBenchmark.cs` shape (standalone mode, JIT-warm, best-of-N timing).

- **Shapes** (seqLen 128; configs from `DistilBertConfig` 768/3072 and `BertConfig` 384/1536):

  | name | [aRows, K, bCols] | purpose |
  |---|---|---|
  | distilbert qkv/o | [128, 768, 768] | K=768 (floor 4.4e-5); Qkv QkvDest layout |
  | distilbert fc1 | [128, 768, 3072] | K=768, GELU epilogue |
  | distilbert fc2 | [128, 3072, 768] | K=3072 (floor 1.6e-4) |
  | distilbert head | [128, 768, 2] | bCols=2 < 64-col tile (partial block) |
  | minilm qkv/o | [128, 384, 384] | Qkv QkvDest layout |
  | minilm fc1 | [128, 384, 1536] | GELU epilogue |
  | minilm fc2 | [128, 1536, 384] | K=1536 |
  | minilm head | [128, 384, 2] | ReLU epilogue |
  | edge padded rows | [100, 770, 70] | non-multiple-of-16 rows/K/bCols → padded-grid bounds |
  | edge padded K | [64, 1032, 130] | partial K tiles + partial 64-col block |

- **Fixtures**: xorshift32 (`state = 0x9E3779B9` seed pattern from kernel fixtures), uniform
  `[-1, 1]` f32 for A[aRows·K], raw W[bCols·K], bias[bCols]. `Bt = transpose(W)` built once
  (the scenario's upload convention: `C = A·Bt`).
- **DP truth** (host, double): plain `C[i,j] = Σₖ A[i,k]·Bt[k,j]`; `Bias` adds `bias[j]`;
  `Gelu` applies the same A–S 7.1.26 polynomial (coeffs as doubles) to `Σ + bias[j]`;
  `Relu` clamps at 0; `Qkv` computes per block (`blockWidth = bCols/3`, bCols % 3 == 0 only)
  and stores into the block-separated layout via the `QkvDest(j, outRow, aRows, blockWidth)`
  mapping (`block·aRows·blockWidth + outRow·blockWidth + colInBlock`).
- **Launch** mirrors `TiledGemm.Launch` exactly (no stream rework — the scenario passes its
  gates with this pattern): grid `(ceil(aRows/16) × ceil(bCols/blockCols))`, `blockCols = 64`
  for the Row4 family, 16 for `OneToOne`; group `16×16`; delegate
  `kernel(runtime.Stream, (numGroups, groupSize), aView, bView, cView, aRows, K, bCols)`
  (+ `biasView` when fused; + `blockWidth` for Qkv), then `runtime.Synchronize()`.
- **Timing**: kernel LoadKernel'd once per variant (first launch JIT-compiles), then per
  (variant, shape): upload Bt/A/bias once, 1 warmup + best-of-25 synchronized launches,
  best µs wins; **GMAC/s = aRows·K·bCols / (µs·1e3)**.
- **Gate**: per cell `maxAbs(|gpu − truth|) ≤ 1e-3`. Rationale (documented in README): the
  honest f32-vs-DP floor is 4.4e-5–1.6e-4 at K=768–3072 for this value class (~10× margin);
  real bugs land ~40–77 (2+ orders above the bound).
- **Variant × shape matrix**: `OneToOne`, `Row4`, `Row4Bias` → all 10 shapes; `Row4Gelu` →
  the two fc1 shapes; `Row4Relu` → the two head shapes; `Row4Qkv` → the two qkv shapes.
  36 timed cells, a few seconds.
- **AC power**: `[DllImport("kernel32.dll")] GetSystemPowerStatus(out SYSTEM_POWER_STATUS)`;
  battery → prominent warning (perf leg invalid; correctness still gated).
- **Exit codes**: `0` all pass; `N` = failed cells; `10` = UNBUILT (no OpenCL GPU /
  `IlgpuRuntime` ctor threw).

### 4. `tests/Nivara.PerformanceTests/Program.cs`

- `ParseArgs`: add `bool gemm` to the tuple + a `case "--gemm":` near the other mode flags.
- `Main`: dispatch `if (gemm) return GemmBenchmark.Run(args);` before the `runs > 1` block.
- Usage line gains `[--gemm]`. Default scenario table and `--json`/`--compare` untouched.

### 5. `tests/Nivara.PerformanceTests/README.md`

- "On-demand helper modes" section: add `--gemm` — what it gates (six ILGPU GEMM variants ×
  model + padded-edge shapes vs double-precision truth ≤ 1e-3), GMAC/s reporting, AC-power
  caveat, driver-bump revalidation note, issue #435.
- Add `GemmBenchmark.cs` to the Files listing.

## Verification

1. `dotnet build Nivara.slnx` — compiles (samples + PerformanceTests).
2. `dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm` on AC power:
   - all 36 cells PASS (exit 0);
   - measured floor reproduces ~4.4e-5–1.6e-4 at K=768/3072;
   - Row4 GMAC/s in the documented 300+ band on model shapes; OneToOne ~150 flat;
   - padded-edge shapes behave (no bounds-check regressions).
3. Record the per-(shape, variant) maxAbs + GMAC/s baseline table in the README.
4. Negative sanity (review-time only): temporarily break `colBase` in a scratch copy →
   target rows fail ~40-class; revert (proves the gate trips; historically already proven).

## Planned commits

1. `docs: plan #435 tiled-GEMM gate in TODO.md`
2. `samples: grant Nivara.PerformanceTests IVT for fused GEMM kernels`
3. `perf: add --gemm tiled-GEMM regression gate (GemmBenchmark)`
4. `docs: document --gemm mode in Nivara.PerformanceTests README`
5. `docs: remove TODO.md — plan executed` (after G2 clears)

## GitHub issues log

- (none created so far; #435 is this PR's issue, #440 stays open as the GMAC/s follow-up —
  this gate's numbers feed it.)

## Blast radius

- `samples/Nivara.Samples/Nivara.Samples.csproj`: one IVT attribute; samples API/behavior
  unchanged; `Nivara.Tests` IVT already exists alongside.
- `tests/Nivara.PerformanceTests/`: new `GemmBenchmark.cs` + `--gemm` flag + package refs +
  README. Default scenario table and the `--json`/`--compare` regression gate are untouched
  (GPU-dependent, per-machine — must never run by default).
- No `src/Nivara` changes; `tests/Nivara.GpuProbe` untouched.
- Downstream users of `TiledGemm`/`IlgpuRuntime` (`BertEncoderGpuRunner`, NivaraGpuRunner
  samples) see no API change.