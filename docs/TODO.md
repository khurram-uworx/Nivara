# Plan: #476 ModernBERT CPU banded attention

## Problem
CPU attention path capped at 2048 by ModernBertMasks.MaxDenseLength because mask is [L,L].
Need carry band + validLength as metadata bounds, no [L,L] mask allocation, so 8192 runnable.

## Proposed changes
- ReverseGradOperations.MultiHeadAttention + BatchedMultiHeadAttention: add mask=null, band=-1, validLength=-1.
- AttentionKernels.ApplyBounds helper.
- ModernBertModel/layer/attention: thread band+validLength, stop building masks.
- LayaHeadModel + BertSelfAttention: validLength path.
- Inference ModernBert.cs: stop building masks.

## Verification
- Build Nivara + Extensions.
- Targeted tests: ModernBertMaskTests, attention tests, Laya tests.
- CPU forward at 8192 (no mask alloc).

## Planned commits
1. docs: plan
2. API + kernel bounds
3. ModernBERT call sites
4. Laya call sites
5. Inference call sites
6. tests (if needed)

## GitHub issues log
(none yet)

## Blast radius
ReverseGradOperations public API (additive optional params), ModernBertAttention/layer signatures (additive overloads), LayaHeadLayer.Forward (additive overload).
