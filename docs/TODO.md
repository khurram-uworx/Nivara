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
  0.088 to 0.00018. This rules out LayerNorm as the primary culprit (identity LayerNorm
  reduces the diff to near-zero).
- With identity LayerNorm, the remaining 0.00018 diff is in the GEMM accumulation or bias
  loading (scorerHidden GELU + scorerOut GEMM).
- Conclusion: the GPU GEMM kernel or weight loading differs from CPU. The LayerNorm was a
  red herring (it amplified a small GEMM difference through the weight matrix).
- GEMM kernel verified correct on a known input (2x3 @ 3x2). CPU and GPU agree exactly.
- With identity LayerNorm, the remaining 0.00018 diff is in the GELU or bias loading.
- The GPU GELU uses the A&S erf polynomial (same as Silk). The CPU uses Math.Erf (exact).
  The difference between exact erf and A&S is ~1e-7, not 0.00018.
- The bias loading: GPU uses UploadPlain (direct copy). CPU uses StateDictLoader.LoadLinear.
  Both read from the same tensors dictionary. Should be identical.
- Next: check if the issue is in how the bias is applied (GEMM epilogue) or in the
  type embedding loading.

## Verification
- `laya --silk compare` PASS
- `laya --ilgpu compare` PASS
- `typesafe nivara verify` all AGREE
- `typesafe nivara benchmark` median materially better

## GitHub issues log
- #771 (this issue)
