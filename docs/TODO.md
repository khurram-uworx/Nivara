# Plan: #771 Silk/ILGPU Laya head parity and performance

## Problem
Both GPU backends (Silk and ILGPU) fail `laya compare` on the Laya decision head:
logits beyond 1e-3 bound. The CPU head matches the Python wheel, and the encoder
passes on both backends. Phase timings (#708) show the head is ~4.8s of a 4.9s
assessment. Root cause likely in the shared head attention/GEMM kernels at seqLen 512.

## Proposed changes
1. Add a head diagnostic mode (`laya --<backend> diag`) that runs CPU vs GPU head at
   configurable seqLen and reports maxdiff, to find the failing threshold.
2. If small seqLens pass and 512 fails, fix the attention/GEMM tiling in the kernel.
3. Re-run `laya --silk compare`, `laya --ilgpu compare`, verify, benchmark.

## Verification
- `laya --silk compare` PASS
- `laya --ilgpu compare` PASS
- `typesafe nivara verify` all AGREE
- `typesafe nivara benchmark` median materially better

## GitHub issues log
- #771 (this issue)
