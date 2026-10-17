# Plan: issue #657 ΓÇö Silk: fuse multi-op layer sequences beyond row4 GEMM epilogues

Branch: khurram/657
Issue: https://github.com/khurram-uworx/Nivara/issues/657

## Problem

Silk decoders issue a large number of small kernels per layer (RMSNorm, RoPE, attention, Q/K/V projections, MLP, cache copy, residuals). `--silk` is now fully pipelined (#652), but enqueue/set-arg overhead and intermediate buffer traffic remain. Fuse the common multi-op sequences at the kernel-source level on top of the existing row4 GEMM epilogues, keeping the per-kernel status checking in fused paths.

Where geometry does not warrant fusion (prefill batches), keep the existing individual kernels.

## Proposed changes

### Slice 0 - plan and grounding
Write this plan, ground it in code-memory, then implement slices below.

### Slice 1 - fused kernel primitives

New OpenCL kernels + launcher entry points:

- `gemm_row4_add` (`gemm_row4` accumulation, final store adds a residual row):
  - signature: `a, b, c, residual, aRows, aCols, bCols`
  - launcher: `GemmAdd(runtime, a, b, c, residual, aRows, aCols, bCols)`
- `rms_norm_gemm_bias` (clone of `rms_norm_lm_head`, store adds bias):
  - signature: `x, gamma, b, bias, c, aRows, aCols, bCols, eps`
  - launcher: `RmsNormGemmBias(runtime, x, gamma, b, bias, c, aRows, aCols, bCols, eps)`

Reuse existing `rms_norm_lm_head` (launcher `RmsNormLmHead`) for bias-free projections.

Reuse row4 accumulation macro and `#pragma OPENCL FP_CONTRACT OFF`; `c` and `residual` may alias (per-row ownership).

### Slice 2 - Qwen cached decode

In `SilkQwenGpuRunner.ForwardCached` and `ForwardCachedArgmax` only:
- Q/K/V: `RmsNormGemmBias` (three launches; eliminates standalone norm + separate `AddBias`)
- O-projection + residual: `GemmAdd`
- gate/up: `RmsNormLmHead` (two launches)
- down + residual: `SwiGlu` then `GemmAdd`

### Slice 3 - SmolLM cached decode

Same as Slice 2, using `RmsNormLmHead` for Q/K/V (no biases).

### Slice 4 - verification and docs

- keep prefill paths individual
- parity gate and targeted tests
- update SILK.md §11 kernel table

### Planned commits

1. docs: plan #657 in TODO.md
2. feat(silk): add `gemm_row4_add` and `rms_norm_gemm_bias` kernels + launchers
3. feat(silk): fuse Qwen cached-decode layer residuals through `gemm_row4_add`
4. feat(silk): absorb input/post RMSNorm into Qwen cached-decode GEMMs
5. feat(silk): apply the same fusion to SmolLM cached decode
6. docs: update Silk kernel catalog and Qwen/SmolLM notes

## Blast radius

- `src/Nivara.Extensions/Silk/SilkKernels.cs`, `SilkLauncher.cs` - new kernel names/source/launchers
- `src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs`, `SilkSmolLMGpuRunner.cs` - cached decode loops only
- `docs/SILK.md` - kernel table/scope
- Tests: `SilkQwenParityTests`, `SilkQwenKvCacheTests`, SmolLM parity if present, `InferenceCliArgumentTests`

## Verification steps

- `dotnet build Nivara.slnx`
- targeted Silk Qwen/SmolLM parity and KV-cache tests
- targeted `InferenceCliArgumentTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- device synthetic benchmark before/after where an OpenCL device is available

## GitHub issues log

(none yet)
