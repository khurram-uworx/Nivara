# Plan: optimize batched_attention scoreCache (#787)

## Problem

The OpenCL `batched_attention` kernel in `src/Nivara.Extensions/Silk/SilkKernels.cs`
keeps a per-thread `float scoreCache[512]`, even though sliding-window layers attend only
`~2*band+1` keys (≈257 for `band == 128`). Reducing the reserved per-thread
storage should lower private/local memory pressure and improve occupancy. Apply
identically in spirit to both backends: Silk uses `scoreCache`; the ILGPU kernel already
runs the recompute path and allocates none.

## Proposed changes

1. Constants in `SilkKernels`:
   - `SlidingAttentionBandMax = 128`
   - `SlidingAttentionCacheSize = 2 * 128 + 1 = 257`
   - `AttentionSliding = "batched_attention_sliding"`
2. Add `AttentionSliding` to `KernelNames` so `SilkRuntime` builds the second program entry.
3. Keep `batched_attention` unchanged for `band < 0` (global attention): 512-float cache,
   `sl <= 512` guard.
4. Add `batched_attention_sliding`, identical body, with `float scoreCache[257]` and
   `cacheReady = range <= 257`. Used for sliding layers (`band >= 0` with span fits).
5. `SilkLauncher.Attention`: choose `AttentionSliding` when `band >= 0 && 2*band+1 <= 257`,
   otherwise `Attention`. Same buffer/arg setup and launch geometry.

## Verification

- Build `Nivara.Extensions`.
- Run targeted Silk parity tests (SilkAttentionTorchParityTests).
- Run `SilkKernelsGate` (human-run) – attention rows must remain bit-identical and inside
  the timing bound.
- Profile `--silk-encoder-profile` before/after; adopt only if attention improves.

## Planned commits

1. `docs: plan #787 in TODO.md`
2. `feat(silk): add batched_attention_sliding with 257-float score cache`
3. `feat(silk): route sliding bands through the sliding kernel`
4. (if needed) targeted test updates

## Blast radius

- `SilkKernels.cs`: new constant, new kernel text, `KernelNames` addition.
- `SilkLauncher.cs`: Attention launcher branch.
- `SilkRuntime`: builds new kernel name; runtime failure if name missing.
- No test/serializer/schema changes expected.

## GitHub issues log

- [ ] #787 — this issue.
