# Plan: Issue #799 — remove ILGPU incumbent from SilkKernelsGate (Silk-only transition, gate half)

## Problem

`--silk-kernels` (`SilkKernelsGate`) currently requires an ILGPU device and compares
every Silk kernel output against the ILGPU kernel of the same shape. That makes the
gate unusable on machines without ILGPU, even though the Silk path itself has no ILGPU
runtime dependency. The ILGPU leg is historical reference only; correctness should be
validated against a CPU reference that performs the same contraction in the same
accumulation order.

## Scope

In scope: `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` (+ doc wording).
Out of scope (tracked separately): removing the ILGPU package and `--ilgpu` backend
from `samples/Nivara.Samples` / `NivaraInference`, and ILGPU-dependent perf probes.

## Proposed changes

Convert each ILGPU cell to the existing `MeasureCpuReference(...)` path:

- GEMM (plain/bias/relu/gelu/qkv): ascending-K `mul+add` loops + epilogue.
- Attention: port of `AttentionKernels.BatchedAttention` to arrays.
- LayerNorm residual / LayerNorm: port of `LayerNormResidual1D` / `LayerNorm1D`.
- EmbedSum, Gather, Add, AddBias, Split, Rotary, GeGlu: direct ports of
  `ElementwiseKernels`.
- ModernBERT ports: same conversions.
- SmolLM + decode-shape A/B: already CPU-referenced; unchanged.

`Run()` drops ILGPU runtime creation; UNBUILT only when Silk/OpenCL unavailable.
Remove `using ILGPU`, `using ILGPU.Runtime`, `using Nivara.Samples.Gpu`, and all
`GpuBuffers.*` usage. Rename `Leg.IlgpuUs` → `ReferenceUs` for clarity.

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` (full rewrite of measurement methods).
- `docs/ACCELERATION.md` §6, `docs/SILK.md` gate section, `tests/Nivara.PerformanceTests/README.md`.
- No production `src/` changes. No test-suite (NUnit) changes.

## Planned commits

1. `docs: plan #799 in TODO.md`
2. `perf(gate): convert GEMM cells to CPU reference`
3. `perf(gate): convert attention/elementwise cells to CPU reference`
4. `perf(gate): convert ModernBERT ports and drop ILGPU leg`
5. `docs: update gate wording for CPU reference`
6. `docs: remove TODO.md — plan executed`

## Verification

- `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release`
- `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels` (human-run)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Handover notes

- Build: `dotnet build tests/Nivara.PerformanceTests/Nivara.PerformanceTests.csproj -c Release` PASSED (0 warnings, 0 errors).
- Gate run `--silk-kernels` started on this machine but is very slow / timed out; complete it on the other machine.
  Command: `dotnet run -c Release --project tests/Nivara.PerformanceTests -- --silk-kernels`
  Expected: all cells PASS (byte-identity where required, otherwise within 1e-3 bound); UNBUILT only if Silk/OpenCL unavailable.
- Whitespace check still to run: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Docs wording updates (ACCELERATION.md §6, SILK.md gate section, PerformanceTests README) still pending.

## GitHub issues log

- (none yet)
