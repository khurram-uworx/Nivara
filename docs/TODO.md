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
- Root cause found: in-place GEMM in the Laya head. The scorerHidden GEMM used markerScored
  as both input and output buffer, which corrupted results because the OpenCL/ILGPU kernels
  read from and wrote to the same buffer (aliasing). Out-of-place GEMM maxDiff=9e-7,
  in-place GEMM maxDiff=1.6.
- Fix: allocate a separate markerScoredOut buffer for the GEMM output.
- Verified: laya --silk compare PASS (worst 4.23e-6), laya --ilgpu compare PASS (worst 4.23e-6),
  typesafe nivara verify all AGREE.

## Verification
- `laya --silk compare` PASS
- `laya --ilgpu compare` PASS
- `typesafe nivara verify` all AGREE
- `typesafe nivara benchmark` median materially better

## GitHub issues log
- #771 (this issue)
