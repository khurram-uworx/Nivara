# Plan: #691 — prototype single-work-group fused O-projection behind capability check

## Problem

`DecodeGqaAttentionOproj` currently runs `decode_gqa_attention` into a `context`
scratch buffer and then `gemm_gemv_add`. True single-buffer fusion was deferred
because the target iGPU reports `CL_DEVICE_MAX_WORK_GROUP_SIZE = 256` while a
single work-group fusion needs `numHeads * AttentionGroupSize` (896 Qwen / 576
SmolLM) threads. This plan prototypes that fused kernel behind a runtime
capability check so capable devices can use it, while the target keeps the
existing two-kernel path.

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkKernels.cs`
   - Add constant `DecodeGqaAttentionOprojFused = "decode_gqa_attention_oproj_fused"`.
   - Add kernel source: single work-group. Loop over heads (each head's 64 lanes
     compute online-softmax attention into a `__local ctx[numHeads*headDim]`
     segment; all threads hit the barrier each iteration). Then a flat strided
     pass over `hiddenDim` output columns doing the GEMV (`sum_k ctx[k]*Wo[k*hiddenDim+col]`)
     plus residual. No `context` global buffer.
   - Add to `KernelNames`.
   - Add `EnsureFusedDecodeOprojLocalMemory`-relevant constant? Local memory
     check lives in runtime.
2. `src/Nivara.Extensions/Silk/SilkRuntime.cs`
   - Add `CanFuseDecodeOproj(int numHeads, int headDim)` (max WG + local memory).
   - Add `EnsureFusedDecodeOprojLocalMemory(int numHeads, int headDim)` mirroring
     `EnsureAttentionLocalMemory`.
3. `src/Nivara.Extensions/Silk/SilkLauncher.cs`
   - Add `DecodeGqaAttentionOprojFused(...)` launcher (single work-group of
     `numHeads * AttentionGroupSize`).
   - Make `DecodeGqaAttentionOproj` check capability and dispatch to fused or
     fall back to existing two-kernel path.
4. `tests/Nivara.Tests/Gpu/SilkDecodeOprojFusedTests.cs`
   - Bit-for-bit parity: fused vs. two-kernel path across kvLen and tail widths,
     skipping on incapable devices.
5. `docs/SILK.md`
   - Kernel table + KernelNames list entry.

## Verification

- Build Extensions + Tests (Release).
- Run targeted: `SilkDecodeOprojFusedTests` (will skip on target iGPU).
- `dotnet format ... --verify-no-changes`.

## Planned commits

1. docs: plan #691 in TODO.md
2. perf(silk): add fused O-proj kernel + runtime capability check
3. perf(silk): wire capability-gated fused O-proj into DecodeGqaAttentionOproj
4. test(gpu): pin fused O-proj against two-kernel path bit-for-bit
5. docs(silk): document fused O-proj kernel
6. docs: remove TODO.md — plan executed

## Blast radius

- SilkKernels.cs: new kernel source + KernelNames (affects build of program; all kernels still created).
- SilkRuntime.cs: new helpers (additive).
- SilkLauncher.cs: DecodeGqaAttentionOproj behavior change (transparent; fallback identical).
- Runners unchanged (call site same).
- Docs citation counts: KernelNames line numbers shift; update citations if needed.

## GitHub issues log

- [ ] #691 — prototype single-work-group fused O-proj behind capability check (this plan)
EOF
git add docs/TODO.md && git commit -m "docs: plan #691 in TODO.md"