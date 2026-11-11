# Plan: perf(silk): Phase B - apply BFloat16 helpers across Silk runners (issue #1141)

Follow-up to #1138. Apply the BFloat16 helpers landed in Phase A consistently across all Silk runners.
No numeric change; build + targeted Silk tests pass.

## Problem
Several Silk runners still widen BF16 via manual `MemoryMarshal.Cast<BFloat16, ushort>` + `BitConverter.UInt32BitsToSingle((uint)u << 16)`, and device-storage casts are undocumented, inconsistent with the helper-based pattern in SilkBertWeights.

## Proposed changes
- MobileNetV2/ResNet18 `.Bf16.cs` `ToFloat`: `dst[i] = (float)src[i];`
- Qwen `.Bf16.cs`/`.cs`: Widen and UploadFloat use direct cast; document device-storage casts.
- SmolLM `.Bf16.cs`/`.cs`: same.
- Laya head `.cs`: widen delegate uses helper; activation conversions use BitConverter helpers or direct casts.
- ModernBERT/Bert encoder: widen delegates use helper; ToUshort casts documented.

## Blast radius
Files: SilkMobileNetV2GpuRunner.Bf16.cs, SilkResNet18GpuRunner.Bf16.cs, SilkQwenGpuRunner.Bf16.cs/.cs, SilkSmolLMGpuRunner.Bf16.cs/.cs, SilkLayaHeadGpuRunner.cs/.Bf16.cs, SilkModernBertGpuRunner.cs/.Bf16.cs, SilkBertEncoderRunner.Bf16.cs. Tests: Silk targeted.

## Planned commits
1. docs: plan issue #1141 in TODO.md
2. refactor(silk): direct BF16 casts in MobileNetV2/ResNet18 ToFloat
3. refactor(silk): direct BF16 casts in Qwen runner
4. refactor(silk): direct BF16 casts in SmolLM runner
5. refactor(silk): helper-based widen in Laya head
6. refactor(silk): helper-based widen in ModernBERT/Bert encoder
7. docs: document device-storage casts
8. docs: remove TODO.md

## GitHub issues log
(none yet)
