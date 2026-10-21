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
- Ran GPU head on a trivial all-ones input (seqLen=1, 0 layers): CPU -0.084776, ILGPU -0.041759.
- Zeroed type_emb + set scorer LayerNorm to identity (weight=1, bias=0): diff dropped from
  0.088 to 0.00018. This isolates the LayerNorm as the primary culprit.
- Conclusion: the GPU LayerNorm kernel produces different output than CPU LayerNorm. The remaining
  0.00018 is GELU approximation error (GPU uses A&S erf polynomial, CPU uses Math.Erf).
- Next: fix the GPU LayerNorm to match CPU (accumulation order or numerical formula).

## Verification
- `laya --silk compare` PASS
- `laya --ilgpu compare` PASS
- `typesafe nivara verify` all AGREE
- `typesafe nivara benchmark` median materially better

## GitHub issues log
- #771 (this issue)
