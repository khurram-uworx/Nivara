# Plan: Silk int4/int8 block-quantized decode weights with in-kernel dequant (#1469)

## Problem

Silk decode currently uses BF16 storage / F32 compute (2 bytes/param). Profiling shows decode is not yet
bandwidth-bound on Intel Xe (~6.3 tok/s Qwen 0.5B), but narrower quantized weights are the major lever once
kernel efficiency approaches the bandwidth ceiling. This plan implements the first milestone: **INT8
symmetric per-block (block size 32) quantization with in-kernel dequant fused into the decode GEMV/GEMM
paths**, keeping F32 activations and accumulators. INT4 is explicitly deferred to a follow-up.

## Format decision (recorded)

- Quantization: symmetric, no zero point.
- Weight format: INT8 (`sbyte`).
- Block size: 32 (aligns with OpenCL subgroup size `GEMVLANES=32` and one cache line; benchmarkable).
- Scale: F32 per block (simplicity + matches F32 compute; revisit FP16 for memory later).
- Activations: F32. Accumulation: F32.
- Layout: weights stay in the existing device layout `[inCols, outCols]` (transposed from checkpoint
  `[out, in]`); scales stored as `[outCols, numBlocks]` so a fixed output column has contiguous scales.
- Block orientation: along the reduction (input) dimension, one scale per 32 input rows per output column.
- Versioned as `SilkInt8Block32 v1`; separate from AutoDiff/training integer tensor support (quantization
  lives only in the Silk inference path: conversion utilities + device buffers + kernels).

## Scope

In scope:
- Host-side conversion from float/BF16 weight tensors to INT8 blocks + F32 scales.
- New INT8 decode kernels: `rms_norm_gemv_int8`, `rms_norm_gemv_subgroup_int8`,
  `rms_norm_gemv_swiglu_int8`, `rms_norm_gemv_swiglu_subgroup_int8`, `gemm_gemv_add_int8`,
  `gemm_gemv_add_subgroup_int8`, `rms_norm_lm_head_int8`.
- Kernel registration + launcher dispatch + runtime buffer helpers.
- Runner integration (SmolLM + Qwen): opt-in `SilkWeightQuantization.Int8Block32`, upload paths,
  dispatch in the cached-decode path (no-LoRA and LoRA).
- Parity tests vs F32 reference.
- Benchmark/profile reporting hooks.

Out of scope (deferred):
- INT4 (follow-up issue).
- Quantizing embeddings / KV cache / activations.
- LoRA-aware quantized base weights beyond reusing the existing decomposed LoRA path on dequantized F32
  scratch (base weights stored INT8; LoRA path dequants base to a temporary F32 buffer — tracked as a
  known limitation, see issues log).

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs` — new kernel source + name constants + registration arrays.
- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — new typed dispatch methods.
- `src/Nivara.Extensions/Silk/SilkRuntime.cs` — byte/sbyte upload helper (raw bytes already supported via
  `AllocateBytes`/`WriteRaw`; add a typed `Upload(nint, ReadOnlySpan<sbyte>)`).
- `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs`, `SilkSmolLMGpuRunner.cs` — new field, constructor
  param, `LayerWeights` scale buffers, upload methods, dispatch branches.
- `src/Nivara.Extensions/Silk/SilkQwenSession.cs`, `SilkSmolLMSession.cs` — constructor param passthrough.
- New: `src/Nivara.Extensions/Silk/SilkInt8Quantizer.cs` (host conversion).
- Tests: new parity tests under `tests/Nivara.Tests/Gpu/`.
- Docs: `docs/SILK.md` format note; `docs/adr/` optional ADR entry.

Downstream: samples that open Silk sessions keep working (default F32/BF16 unchanged). Docs snippet gate
and DocsCoveragePins unaffected unless docs files added.

## Planned commits

1. docs: plan Silk int8 block-quant decode weights in TODO.md
2. feat(silk): add INT8 block-32 quantizer (host conversion) + unit tests
3. feat(silk): add INT8 decode kernel source + registration
4. feat(silk): add typed launcher dispatch for INT8 decode kernels
5. feat(silk): wire INT8 weights into SmolLM runner (upload + dispatch)
6. feat(silk): wire INT8 weights into Qwen runner (upload + dispatch)
7. feat(silk): session constructors expose quantization mode
8. test(silk): INT8 decode parity vs F32 reference
9. docs(silk): record INT8 block-32 format decision
10. docs: remove TODO.md — plan executed

## Verification

- Build: `dotnet build Nivara.slnx` (Extensions project).
- Targeted tests: Silk parity tests + kernel gate.
- Snippet gate: `EveryGatedBlock_CompilesWithoutErrors` if docs changed.
- Formatting: `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## GitHub issues log

- (none yet — deferred work captured during execution will be recorded here)
