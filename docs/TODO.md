# Plan: Silk/OpenCL narrow-precision (bf16 + fp16/Half) translation unit (#1019)

## Problem
Silk/OpenCL runners widen BF16 weights to F32 before dispatch for ModernBERT, Laya, MiniLM, DistilBERT, and DistilBERT-SST, and only Qwen/SmolLM partially keep BF16 linear weights. `--silk --precision fp16` is rejected entirely on every model. Weights remain F32 in memory, so the GPU path does not get the memory-movement benefit the issue asks for.

## Goal
Support both BF16 and FP16 (Half) narrow weight storage on Silk for: Qwen, SmolLM, ModernBERT, Laya, MiniLM, DistilBERT, DistilBERT-SST. Embedding tables (`tokEmb`) and tied LM heads included in the same pass, not widened. MobileNetV2/ResNet18 stay F32-only (no BF16 weight path). fp16 supported (not rejected) on the listed models via OpenCL `cl_khr_fp16`.

## Constraints
- Activations remain F32. OpenCL on this target converts narrow -> float and multiplies in F32 (parity with existing bf16 kernels).
- Norm weights/biases and LayerNorm epsilons stay F32 (numerical stability; small footprint).
- CPU fp16/BF16 paths unchanged.
- No new third-party dependencies.
- CI can't run GPU tests; parity on hardware is human-run.

## Planned changes
### Phase 1 — kernel stack
- `SilkKernels.cs`: add BF16 + F16 epilogue variants: `gather_rows_bf16/f16`, `gemm_row4_bf16` (exists)+`_f16`, `gemm_row4_bias_bf16/f16`, `gemm_row4_qkv_bf16/f16`, `gemm_row4_gelu_bf16/f16`, `gemm_row4_relu_bf16/f16`, `rms_norm_lm_head_bf16/f16`. New `gemm_row4_f16`. Register all names in `KernelNames`.
- `SilkLauncher.cs`: add dispatch helpers for each variant, following the `GemmBf16`/`RmsNormGemvBf16` pattern.
- BF16 widen: `as_float(((uint)x) << 16)`; F16 widen: `((float)as_half(x))` with `#pragma OPENCL EXTENSION cl_khr_fp16`. Add both pragmas only in F16 translation section (the source is one translation unit; guard with `#ifdef`? Use separate `#pragma` blocks at top).

### Phase 2 — runners
- `SilkModernBertGpuRunner`: add `useNarrow` + precision enum; constructor accepting either float (F32), BFloat16, or Half tensors. Store QKV/Wo/Wi/WoMlp + `tokEmb` as `ushort`. Branch `LaunchGemm`, `LaunchGather`.
- `.Bf16.cs` / new fp16 partial: stop widening in `ToFloat`; instead defer to the narrow constructor path.
- `SilkLayaHeadGpuRunner`: same.
- `SilkBertEncoderRunner` + `SilkBertWeights`: add BFloat16/Half paths; linear weights + head become narrow; norms/biases/embeddings LayerNorm stay F32. `Export()` widens narrow buffers to F32 for trainer compatibility.
- Qwen/SmolLM: upgrade `tokEmb`/`tokEmbTransposed` to narrow and use new gather + `rms_norm_lm_head_*` kernels instead of widening embedding (otherwise weights-not-half regression remains here too).

### Phase 3 — sample, validation, tests
- `Program.cs`: dispatch narrow tensors directly instead of `WidenBFloat16Tensors`; fp16 allowed (parse Half, reinterpret to ushort) on supported models; MobileNetV2/ResNet18 keep widen + reject bf16/fp16.
- `InferenceCliArgumentTests`: update pins to reflect bf16/fp16 now consumed.
- Add parity tests per family (bf16 and fp16): ModernBERT at least, mirroring Qwen; SmolLM if time.

### Phase 4 — docs
- `docs/SILK.md`: update precision section (bf16 + fp16 supported on listed models; F32 elsewhere).
- Sample README / ACCELERATION.md precision lines.

### Phase 5 — deferred follow-ups
- If embedding LM-head bf16/fp16 parity slips, open a GitHub issue and record it; do not leave widen paths on supported models.

## Blast radius
- Source: `SilkKernels.cs`, `SilkLauncher.cs`, `SilkModernBertGpuRunner.cs` (+Bf16 partial + new fp16 partial), `SilkLayaHeadGpuRunner.cs` + partials, `SilkBertEncoderRunner.cs`, `SilkBertWeights.cs`, `SilkQwenGpuRunner.cs` + partial, `SilkSmolLMGpuRunner.cs` + partial.
- Sample: `Program.cs` (routing + `WidenBFloat16Tensors` for unsupported models only).
- Tests: `InferenceCliArgumentTests.cs`, new parity fixtures.
- Docs: `SILK.md`, sample README, `ACCELERATION.md`.

## GitHub issues log
- Pending follow-up (fill in as discovered).

## Planned commits
1. docs: plan 1019 in TODO.md
2. feat(silk): bf16+f16 kernel stack and launcher dispatch
3. feat(silk): ModernBERT narrow (bf16+fp16) storage
4. feat(silk): Laya narrow
5. feat(silk): BertEncoder/MiniLM/DistilBERT narrow
6. feat(silk): Qwen/SmolLM narrow embeddings/LM-head
7. feat(sample): route narrow tensors directly; remove widen on supported models
8. test: parity + CLI pins
9. docs: SILK precision update
10. docs: remove TODO.md
