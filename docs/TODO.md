# TODO: Silk prefill GEMM GB/s and GFLOP/s reporting (#1482)

## Problem

Follow-up to #1473. The `--silk-kernels` perf gate (`tests/Nivara.PerformanceTests/SilkKernelsGate.cs`) times the prefill GEMM family (`gemm_row4_bias`, `gemm_row4_relu`, `gemm_row4_gelu`, `gemm_row4_qkv`, `gemm_row4`) across MiniLM/DistilBERT/ModernBERT shapes and reports byte-identity, `%bound`, `cpu us`, `silk us`, and timing ratio, but no throughput. The Leg table cannot answer whether a cell is bandwidth- or compute-bound.

Issue: #1482 (OPEN, `performance`).

## Proposed changes

### 1. Extend `Leg` with throughput inputs

File: `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`

Add additive fields (default 0 = not applicable):

```csharp
/// <summary>Logical bytes moved (A + B + C [+ bias]) for throughput reporting; 0 when not applicable.</summary>
public long LogicalBytes { get; init; }
/// <summary>Logical FLOPs (2*M*N*K for GEMM) for throughput reporting; 0 when not applicable.</summary>
public long Flops { get; init; }
```

### 2. Plumb through `MeasureCpuReference`

Add defaulted parameters to both overloads (`:1295-1327`): `long logicalBytes = 0, long flops = 0`, set on returned `Leg`. Existing call sites unchanged.

### 3. Compute and pass for prefill GEMM family

- `MeasureGemms` (`:130-261`, MiniLM/DistilBERT): for each GEMM cell, `M=aRows, K=aCols, N=bCols`.
  - `flops = 2L * M * N * K`
  - bias/relu/gelu/qkv: `logicalBytes = ((long)M*K + (long)K*N + (long)M*N + N) * sizeof(float)` (A + B_transposed + C + bias)
  - pass to each `MeasureCpuReference`.
- `MeasurePlainGemm` (`:642-699`, ModernBERT, `gemm_row4`, no bias):
  - same `flops`
  - `logicalBytes = ((long)M*K + (long)K*N + (long)M*N) * sizeof(float)` (no bias)

Scope is only the 5 prefill GEMM kernels named by #1482. Decode-shape GB/s sections already exist and are not changed.

### 4. Report Leg table throughput columns

File: `tests/Nivara.PerformanceTests/SilkKernelsGate.cs`, `Report` (`:1840-1889`).

Add additive `GB/s` and `GF/s` columns. For `LogicalBytes > 0`:
- `gbs = LogicalBytes / SilkUs / 1000.0` (`bytes / (us*1e-6) / 1e9`)
- `gflops = Flops / SilkUs / 1000.0`

For non-applicable legs print `-`. Widen header/separator. Keep byte-identity status, `%bound`, cpu/silk us, ratio, pass/fail criteria, and exit code unchanged.

### 5. Documentation

- `docs/SILK.md`: add prefill GEMM GB/s/GFLOP/s reporting subsection near existing decode GB/s measurement sections; document logical-bytes formula, GFLOP/s formula, comparability rules (same device, Release, median-of-rounds note/timing basis), and that GB/s is logical bytes moved (not measured DRAM). Reference #1482.
- Check `docs/DISTELBERT.md`, `docs/ACCELERATION.md` for any text asserting old `--silk-kernels` column set; update only if needed.

## Decision / red flag to resolve (G1)

- **Timing basis for throughput:** `Leg.SilkUs` is currently `GemmBenchmark.TimeBest` (best-of-10 after 3 warmups). Issue #1482 comparability says "median of rounds". Decode GB/s sections use `Median`. Using `SilkUs` (TimeBest) keeps the table internally consistent and preserves timing ratio semantics; using median would require capturing median separately. **Plan default: compute GB/s/GFLOP/s from existing `SilkUs` (TimeBest) for table consistency, and document the timing basis explicitly.** Confirm with human during G1 before implementation if a different basis is preferred.

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — Leg, MeasureCpuReference, MeasureGemms, MeasurePlainGemm, Report. Additive fields/columns; correctness gate unchanged.
- `docs/SILK.md` — new documentation subsection.
- Possibly `docs/DISTELBERT.md`, `docs/ACCELERATION.md` — only if they assert old column format.
- No changes to `src/Nivara.Extensions/Silk/*` kernels, no runtime behavior changes, no public API changes.
- Downstream: none (perf gate is a console harness, not a library API). Tests: no NUnit tests cover SilkKernelsGate output; build/format/snippet gate are the relevant checks.

## Verification

- [ ] `dotnet build Nivara.slnx -c Release` (or at least `tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`)
- [ ] `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- [ ] Targeted docs snippet gate if docs change: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (ask before running)
- [ ] **Human-run measurement (battery/AC):** `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` — confirm GB/s/GFLOP/s appear for gemm_row4* prefill cells and gate PASS/fail criteria unchanged. **Prompt before taking measurement; we are on battery and I will plug AC power.**
- [ ] Review branch changes and revise `DocsCoveragePins` if required (doc/citation counts may change).

## Planned commits

1. `docs: plan Silk prefill GEMM GB/s and GFLOP/s reporting (#1482) in TODO.md`
2. `perf: report GB/s and GFLOP/s for prefill GEMM cells in silk-kernels gate (#1482)`
3. `docs: document prefill GEMM GB/s and GFLOP/s reporting (#1482)` (if separate from code/doc coupling; may fold into commit 2 if small)
4. `docs: remove TODO.md — plan executed (#1482)` (only after G2 clears)

## GitHub issues log

- [ ] #1482 — Silk: prefill GEMM GB/s and GFLOP/s reporting (this plan)

As each task executes, if you find deferred work or a concern outside this plan, create a tracked issue immediately via `gh issue create --repo khurram-uworx/Nivara` and record its number here. Don't rely on memory or wait until the plan finishes.
