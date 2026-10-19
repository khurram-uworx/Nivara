# Plan: Issue #658 Milestone 1 — tiled, cache-blocked, multithreaded core GEMM

Branch: `khurram/658-milestone-1`
Base: `main` (c274dcac)

## Problem

Nivara CPU SmolLM decode runs ~7.5 tok/s vs PyTorch ~15.8 tok/s (roughly a 2x gap).
The dominant per-token cost is matrix multiply through `TensorsHelper.MultiplyCore*`, which
currently dispatches float/double through `TensorPrimitives.Dot` row-by-row and generic
types through `WidenPrimitives.Dot`, with row-level `Parallel.For` for large matrices.

This milestone targets the GEMM core only, without changing accumulation order or K-loop
semantics, so parity remains bit-exact where the tests demand it.

## Scope for this branch

In scope:
- `src/Nivara/Tensors/TensorsHelper.cs`: row-tile grouping, column/output locality,
  pool hygiene, parallel granularity.
- Targeted parity/allocation tests.

Out of scope (deferred):
- Online-softmax decode attention (issue #658 Milestone 2).
- KV-cache lifecycle and fused decode hygiene (issue #658 Milestone 3).
- Public API, data-flow, or docs changes.

## Proposed changes

1. Preserve the single-row BLAS2 GEMV fast path in `MultiplyCore<T>` for
   `bTransposed && aRows == 1`.
2. Preserve per-type dispatch at entry (`float`, `double`, then generic) and all inner
   accumulation via `TensorPrimitives.Dot` / `WidenPrimitives.Dot`.
3. Replace direct row-index `Parallel.For` with row-tile grouping (`Parallel.For` over
   row-tile indices) while leaving the inner row/output loops sequential.
4. Keep `ShouldParallelize(aRows, aCols, bCols)` keyed on total volume.
5. Keep one `bT` rent and transpose/copy; rent `aCopy` only in parallel mode.
6. Ensure every pooled buffer is returned with `clearArray: true`.
7. Keep null-aware `Multiply` wrapper and mask propagation untouched.

## Blast radius

Primary:
- `src/Nivara/Tensors/TensorsHelper.cs`
- `MultiplyCore`, `MultiplyCoreFloat`, `MultiplyCoreDouble`, `MultiplyCoreGeneric`,
  `MultiplyRowFloat`, `MultiplyRowDouble`, `MultiplyRowScalar`, `ShouldParallelize`,
  `RentCopy`.

Downstream (call sites remain unchanged):
- `src/Nivara/AutoDiff/Operations/GradKernels.cs` (`MatMul`, `MatMulTransposedB`)
- `src/Nivara/AutoDiff/Nn/LlamaFusedKernels.cs` (delegates to GradKernels)
- Attention and decoder-block code that consumes GEMM.

Tests:
- `tests/Nivara.Tests/Tensors/TensorsHelperTests.cs`
- `tests/Nivara.Tests/Primitives/WidenPrimitivesPhase1Tests.cs`
- `tests/Nivara.Tests/AutoDiff/LlamaCausalAttentionTests.cs`
- `tests/Nivara.Tests/AutoDiff/LlamaDecoderBlockTests.cs`
- `tests/Nivara.Tests/AutoDiff/LlamaCausalKVCacheTests.cs`

## Verification steps

Build:
- `dotnet build Nivara.slnx`

Targeted Release tests:
- `dotnet test -c Release --filter "FullyQualifiedName~TensorsHelperTests"`
- `dotnet test -c Release --filter "FullyQualifiedName~WidenPrimitivesPhase1Tests"`
- `dotnet test -c Release --filter "FullyQualifiedName~LlamaCausalAttentionTests|LlamaDecoderBlockTests|LlamaCausalKVCacheTests"`

Formatting / snippet gate:
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test -c Release --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"`

Human-run performance (not to be polled by the agent unless directed):
- `dotnet run -c Release --project samples/NivaraInference -- smollm benchmark`

## Planned commits

1. `docs: plan #658 milestone 1 GEMM work in TODO.md`
2. `refactor(gemm): group MultiplyCore parallel work into row tiles`
3. `test(gemm): verify row-tiled implementation keeps parity gates green`

## GitHub issues log

- #658 — CPU: tiled multithreaded GEMM and cache-friendly SmolLM decode (parent)
- #458 — CPU-side performance deficit (context tracker)