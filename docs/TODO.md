# Plan: Silk narrow tokEmb / tied LM head for Qwen and SmolLM (#1042)

Follow-up to #1019. Qwen and SmolLM keep linear weights in bf16 but widen
`tokEmb` / `tokEmbTransposed` to F32 before upload. Store them as bf16 (ushort)
using `gather_rows_bf16` and the bf16 LM-head path, measure parity, document
memory reduction.

## Problem

- `SilkQwenGpuRunner.Bf16.cs` and `SilkSmolLMGpuRunner.Bf16.cs` widen
  `model.embed_tokens.weight` to `float[]` before upload, doubling embedding
  memory vs. the bf16 linear weights.
- Shared forward paths call `SilkLauncher.Gather` (F32 kernel) and the F32
  LM-head kernels even when `bf16Weights == true`, because the buffers are F32.
- Qwen cached/argmax paths call `SilkLauncher.RmsNormLmHead` (F32) instead of
  `RmsNormLmHeadBf16`.
- SmolLM LM-head paths call `SilkLauncher.Gemm` (F32) instead of the bf16-aware
  `Gemm` helper.

## Proposed changes

### SilkQwenGpuRunner.Bf16.cs
- `tokEmb` / `tokEmbTransposed` allocated via `AllocateBytes(ushort)` and uploaded as `ushort[]`.
- Keep `finalNormW` widened to F32.

### SilkQwenGpuRunner.cs (shared)
- Add `GatherTokEmb(...)` and `RmsNormLmHead(...)` helpers that dispatch on
  `bf16Weights`.
- Replace direct `SilkLauncher.Gather(runtime, tokEmb, ...)` and
  `SilkLauncher.RmsNormLmHead(runtime, ...)` call sites.
- `Gemm(..., tokEmbTransposed, ...)` already dispatches via the `Gemm` helper.

### SilkSmolLMGpuRunner.Bf16.cs
- Same embedding storage change as Qwen.

### SilkSmolLMGpuRunner.cs (shared)
- Add `GatherTokEmb(...)` helper.
- Replace direct gather calls and direct `SilkLauncher.Gemm(..., tokEmbTransposed, ...)`
  calls with the `Gemm` helper.

## Blast radius

Files:
- src/Nivara.Extensions/Silk/SilkQwenGpuRunner.Bf16.cs
- src/Nivara.Extensions/Silk/SilkQwenGpuRunner.cs
- src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.Bf16.cs
- src/Nivara.Extensions/Silk/SilkSmolLMGpuRunner.cs

Symbols touched: constructors (bf16 route), `Gather` call sites, `RmsNormLmHead`
call sites, helper methods. F32 route untouched (buffers stay float, helpers
dispatch to F32 kernels). Parity tests: `SilkQwenBf16ParityTests`, Qwen KV cache
tests; SmolLM tests if present. Sessions unchanged (constructors keep same
signatures).

## GitHub issues log

- (none yet)

## Verification

- `dotnet build Nivara.slnx`
- Targeted Qwen bf16 parity + KV cache tests (Gpu category; requires OpenCL).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`