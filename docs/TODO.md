# Plan: issue #664 — Perf: fused final RMSNorm + tied LM head and larger-kvLen decode for Qwen

Branch: `khurram/664`
Issue: https://github.com/khurram-uworx/Nivara/issues/664

## Problem

Silk Qwen decode still runs final RMSNorm (`rms_norm`) and the tied LM head (`gemm_row4`) as separate kernels, even though the normed output feeds immediately into the head GEMV. This adds one kernel launch and an intermediate `normed` (or `lastRowView`) buffer on the hottest per-token path. Qwen has the largest decoder head in the sample (151936×896), so this boundary is worth fusing at the kernel level, consistent with #657's multi-op fusion direction but isolated to the final stage.

Cached decode also needs to be verified at larger `kvLen`; the underlying `decode_gqa_attention` kernel is already a fused per-head online softmax, so no algorithmic change is expected unless benchmarks show otherwise.

## Proposed changes

### Slice 1 — Add `rms_norm_lm_head` kernel

New kernel in `SilkKernels.cs` source, name `rms_norm_lm_head`, added to `KernelNames`.

Arguments:
- `x` (hidden-sized input row), `gamma`, `b` (toKEmbTransposed [hidden, vocab]), `c` (logits [vocab]), `aRows`, `aCols` (hidden), `bCols` (vocab), `eps`.

Semantics:
- Compute `sumSq` ascending over `hidden`, `inv = 1 / sqrt(sumSq/hidden + eps)` in the same order as `rms_norm`.
- Accumulate the Row4 GEMV with the normalized activation `x[k] * inv * gamma[k]`.
- Keep aliasing rules and bounds handling identical to `gemm_row4`.

Launcher:
- `SilkLauncher.RmsNormLmHead(...)` with the same grid and tile geometry as `Gemm`.

### Slice 2 — Replace final stage in single-row Qwen paths

Use the fused kernel in:
- `ForwardCached`
- `ForwardCachedArgmax`
- `ForwardPrefill`
- `ForwardPrefillArgmax`

No change to `Forward` (full-row path) in this issue.

### Slice 3 — Larger-`kvLen` decode verification

Run the existing Qwen decode attention probe at 512/1024/4096 after the change.
No kernel change planned unless a measured bottleneck appears.

### Slice 4 — Tests and docs

- Extend `SilkQwenParityTests` to verify fused output matches the norm + GEMM baseline within the existing numeric bound.
- Leave BF16 GPU weight storage deferred pending measurement.

## Planned commits

1. docs: plan #664 in TODO.md
2. feat(silk): add rms_norm_lm_head kernel + launcher
3. feat(silk): use fused final RMSNorm + tied LM head in Qwen paths
4. test: parity for fused final stage
5. docs: note BF16 deferral and kernel name

## Blast radius

- `SilkKernels.cs` public catalog
- `SilkLauncher.cs`
- `SilkQwenGpuRunner.cs` final stages
- Wired tests only; no sample API change.

## GitHub issues log

- [ ] #664 — fused final RMSNorm + tied LM head and larger-kvLen decode for Qwen