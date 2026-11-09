# Plan: issue #1013 - opt-in BF16 inference lane (BERT/Laya) via Silk

## Problem

Silk ships BF16 GEMM variants and narrow-weight BERT/ModernBERT runners that already execute
end-to-end, but the Laya decision head widens BF16/Half weights to F32 silently, the
GpuSeam factory delegates accept only F32 tensors so callers cannot opt into narrow precision
through the seam abstraction, and there is no documented tolerance/parity gate for the ModernBERT
and Laya narrow CPU vs Silk paths. The issue asks for a defined parity/quality story (opt-in,
not default), starting with BERT-family inference then Laya.

Scope for this branch (Phase 1, inference only):
- ModernBERT BERT-family BF16/Half inference via Silk, documented tolerance, parity gates.
- Laya BF16/Half inference via Silk without silent weight widening (head weights stay narrow;
  activations remain F32, since activation-level BF16 kernels for the head are not yet shipped).
- Seam-level opt-in (delegates that surface the narrow sessions).

Out of scope (deferred, tracked issues):
- Activation-traffic halving in the Laya/Meta head (new BF16 Elementwise/LayerNorm kernels).
- BF16 training lane and the AdamW moment behavior (Silk trainer is F32; partial/frozen training only
  recently shipped under #1012).
- ModernBERT/Laya Silk trainers (do not exist yet).

## Proposed changes

1. src/Nivara.Extensions/AI/GpuSeam.cs - add seam overloads/delegates for narrow tensors.
2. src/Nivara.Extensions/Silk/SilkLayaHeadGpuRunner.Bf16.cs and Half.cs - stop widening in the
   constructor; instead store narrow weights through a narrowBf16/narrowF16 flag mirroring the
   encoder, allocating activation buffers in F32 (existing kernels), dispatching weights via GemmBf16/F16.
3. Tests - extend SilkLayaNarrowParityTests with Laya BF16/F16 head-path parity.
4. Docs - docs/SILK.md narrows inference story and tolerance.

## Verification steps

- dotnet build Nivara.slnx
- targeted Gpu tests for narrow parity
- Docs snippet gate after doc edit
- formatting verify

## Planned commits

1. docs: plan #1013 in TODO.md
2. feat(silk): seam opt-in for BF16/F16 ModernBERT and Laya sessions
3. fix(silk): keep Laya head weights narrow instead of widening at load
4. test(silk): parity gates for Laya BF16/F16 head path
5. docs(silk): document BF16 inference lane and tolerance

## GitHub issues log

- [ ] (to be created) #? - defer activation-level BF16 Laya head kernels
- [ ] (to be created) #? - defer BF16 Silk training lane (optimizer moments in FP32)

## Blast radius

- GpuSeam delegates new; non-breaking, appended only.
- Laya head runner constructor change: sample uses narrow constructor; sample updated if needed.
- Silk ModernBERT session signature unaffected.