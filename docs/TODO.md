# Plan: #771 Silk/ILGPU Laya head parity and performance

## Problem
Both GPU backends (Silk and ILGPU) fail `laya compare` on the Laya decision head:
logits beyond 1e-3 bound. The CPU head matches the Python wheel, and the encoder
passes on both backends. Phase timings (#708) show the head is ~4.8s of a 4.9s
assessment. Root cause likely in the shared head attention/GEMM kernels at seqLen 512.

## Investigation findings (2026-10-21)
- Forced the attention kernel's recompute path (disable scoreCache): still FAIL.
- Reduced seqLen to 128 and 32: still FAIL.
- Reduced head layer count to 0 (type add + scorer only): still FAIL.
- Ran GPU head on a trivial all-ones input (seqLen=4, 0 layers): CPU logits -0.084776 (identical
  for both markers); Silk -0.237384/-0.237392; ILGPU -0.136088 (identical). Both backends
  produce symmetric results but differ from CPU and from each other.
- Conclusion: failure is not seqLen- or layer-dependent; it is a fundamental operator or weight-
  loading difference in the GPU head (type-add or scorer), common to both backends. The CPU head
  matches the Python wheel.
- Next: isolate each operator (type-add, LayerNorm, GEMM, GELU) on a controlled input.

## Verification
- `laya --silk compare` PASS
- `laya --ilgpu compare` PASS
- `typesafe nivara verify` all AGREE
- `typesafe nivara benchmark` median materially better

## GitHub issues log
- #771 (this issue)
