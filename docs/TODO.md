# Plan: Issue #450 — Fused pre-norm block for ModernBERT

## Problem

A 28-layer ModernBERT forward is 28 separate module forwards. The masks were the
dominant cost and are already gone (#447/#476); what remains is per-layer module
overhead and per-op tensor allocation. Land a fused pre-norm block (per-layer band +
theta + GeGLU epilogue) on the CPU inference path, mirroring the Llama
`ForwardCachedFused` idiom, so `modernbert benchmark` exercises it.

## Proposed changes

- `ModernBertAttention<T>`: expose `NumHeads`, `HeadDim`.
- `ModernBertLayer<T>`: store `hiddenSize`; add `ForwardFused(input, output, band, validLength)`.
- `ModernBertEncoder<T>.Forward(int[], int)`: use fused path when `!GradEnabled`, else per-op chain.
- Tests: parity vs per-op chain (1e-5), layer-0 missing norm, NaN-row guarantee, alloc guard.
- Docs: update `docs/MODERNBERT.md` performance rows.

## GitHub issues log

- (none yet)

## Verification

- build, targeted tests, format check; human-run benchmark + compare.

## Planned commits

1. docs: plan #450 in TODO.md
2. feat: expose ModernBertAttention.NumHeads/HeadDim and store hiddenSize
3. feat: add ModernBertLayer.ForwardFused
4. feat: route ModernBertEncoder inference through fused layers
5. test: ModernBert fused block parity/NaN/alloc
6. docs: record #450 perf update

## Blast radius

- `ModernBertLayer`, `ModernBertAttention`, `ModernBertEncoder.Forward(int[],int)`.
- Downstream: sample `modernbert` benchmark/compare, Laya (uses ModernBertEncoder), GPU runners unaffected (separate code paths).
- Tests: new + existing ModernBert/banded attention tests.