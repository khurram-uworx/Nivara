# Plan: CPU tiled GEMM + cache-friendly SmolLM decode (#658)

## Problem
Nivara CPU SmolLM decode is competitive for F32 but BF16 remains ~2.4x slower than F32 due to narrow-float GEMV/GEMM and attention overhead. Issue asks for tiled multithreaded GEMM and fused online-softmax attention.

Baseline (Release):
- SmolLM F32: 15.80 tok/s
- SmolLM BF16: 6.55 tok/s
- CpuGemmProbe maxAbs gate PASS all legs

## Blast radius
- `src/Nivara/Tensors/TensorsHelper.cs` (`MultiplyCore*`): all AutoDiff linear/projection matmuls.
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` (`DecodeAttention`, `BatchedAttention`): Llama decode/prefill.
- `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`, `LlamaDecoderBlock.cs`: cached inference paths.
- `samples/Nivara.Samples/LlamaForCausalLM.cs`: orchestration and KV cache growth.
- `NarrowFloatKernels` only called indirectly through `WidenPrimitives`.

## Proposed changes
1. Add a blocked, vectorized CPU GEMM path in TensorsHelper, starting behind the existing `MultiplyCore` entry point, preserving `bTransposed` semantics and the M==1 fast path.
2. Replace attention's score-then-softmax-then-V passes with an online-softmax single pass for decode (and prefill where parity-safe).
3. Pre-size KV cache and reduce per-token setup overhead.
4. Add numerical and regression coverage.

## Verification steps
- `dotnet build Nivara.slnx`
- Targeted tests (ask before running): Llama KV cache, attention, decoder block, TensorsHelper, GradKernels.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Doc snippet gate EveryGatedBlock_CompilesWithoutErrors
- SmolLM F32/BF16 benchmarks and CpuGemmProbe.

## Planned commits
1. docs: plan #658 in TODO.md
2. perf(gemm): blocked tiled float/double CPU GEMM (if beneficial)
3. perf(attention): online-softmax decode attention
4. perf(smollm): pre-size KV cache and cache spans
5. tests: numerical equivalence coverage
6. docs: remove TODO.md

## GitHub issues log
(none yet)
