# Plan: Activation-level BF16 kernels for Silk Laya head (#1102)

## Problem
Silk Laya head today is narrow-precision only: BF16/Half weights are uploaded as `ushort` and
widened in-shader, but every activation buffer and every elementwise kernel (layer_norm, add,
add_bias, gather_rows, attention) is still F32. Activation traffic is not halved.

## Proposed changes
1. Add BF16 elementwise kernels for the head block: layer_norm_bf16, add_bf16, add_bias_bf16,
   gather_rows_bf16_act.
2. Add BF16 GEMM/attention kernels with BF16 activations: gemm_row4_*_bf16act, batched_attention_bf16act.
3. Add byte-aware runtime helpers for ushort buffers.
4. Add `activationBf16`/`activationF16` opt-in to SilkLayaHeadGpuRunner and SilkLayaSession.
5. Add parity gate vs F32 head.

## Verification
- dotnet build Nivara.slnx
- targeted SilkLaya tests on GPU

## Planned commits
- docs: plan issue #1102
- feat(silk): BF16 elementwise kernels for Laya head
- feat(silk): BF16 GEMM/attention kernels with BF16 activations
- feat(silk): byte-aware runtime helpers for BF16 activations
- feat(silk): opt-in activation-level BF16 in Laya head runner
- feat(silk): thread activation BF16 through session + seam
- test(gpu): parity gate for activation-level BF16 Laya head
- docs: update SILK.md

## GitHub issues log
- (none yet)
