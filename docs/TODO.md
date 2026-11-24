# Plan: Silk decode attention long-kvLen shape (#1467)

## Problem

Issue #1467 (follow-up to #1453 / PR #1465). After MLP decomposition, attention+O-proj is
~17-21 ms/token in profiled synthetic decode (SmolLM 17.024, Qwen 21.459 ms/token). Not the
current bottleneck, but shape/kvLen sensitivity matters.

Current decode path uses `SilkLauncher.DecodeGqaAttentionOproj` which selects
`decode_gqa_attention_oproj_fused` (single workgroup, `numHeads*ATTNGROUP` threads) when device
limits allow; fallback is split decode attention (`decode_gqa_attention`) + `gemm_gemv_add`.

The fused kernel is single-WG, which limits occupancy and may degrade at larger kvLen.
Long-context decode may benefit from flash-decoding style kv-splitting/reduction.

## Acceptance criteria

- [x] Measured kvLen sweep reported with steady-state tok/s and attn+oproj phase ms/token.
- [x] Clear selection rule or heuristic documented (or justified default).
- [x] Parity tests for selected path(s), including GQA and long kvLen.

## Code-memory symbol navigation

Key symbols:
- `Nivara.Silk.SilkLauncher.DecodeGqaAttentionOproj` (SilkLauncher.cs:507-521) — dispatch
- `Nivara.Silk.SilkRuntime.CanFuseDecodeOproj` (SilkRuntime.cs:197-210) — capability gate
- `Nivara.Silk.SilkKernels.DecodeGqaAttentionOprojFused` (SilkKernels.cs:151) — fused kernel
- `Nivara.Silk.SilkKernels.DecodeGqaAttention` (SilkKernels.cs:3102) — split attention
- `Nivara.Silk.SilkKernels.GemmGemvAdd` (SilkKernels.cs:146) — flat GEMV+residual
- `Nivara.Tests.Gpu.SilkDecodeAttentionParityTests` — parity tests (split + fused)

## Recent relevant history

- #1486 (K-split decode GEMV subgroup reduction): template for kv/k-split.
- #1483 (fused MLP block GB/s): established logical-bytes-moved reporting pattern.
- #1482 (prefill GEMM GB/s/GFLOP/s): throughput reporting pattern.
- #1453/#1465 (MLP decomposition): exposed attention+O-proj as next profiling target.

## Assessment (based on fresh numbers)

### Device

- Intel(R) Iris(R) Xe Graphics, driver 31.0.101.5333, max WG 512, local 65536 B, subgroups yes.
- Fused path NOT available on this device (maxWG 512 < 576 for SmolLM, < 896 for Qwen).
  `canFuse=False` for both models.

### Decode attention + O-proj kvLen sweep (from release/vCurrent/silk-kernels.txt)

| model  | kvLen | split ms | tok/s  |
|--------|-------|----------|--------|
| smollm | 128   | 0.580    | 1723.7 |
| smollm | 512   | 0.881    | 1134.9 |
| smollm | 2048  | 1.901    | 526.1  |
| smollm | 8192  | 5.600    | 178.6  |
| qwen   | 128   | 0.792    | 1263.2 |
| qwen   | 512   | 1.188    | 841.4  |
| qwen   | 2048  | 2.117    | 472.3  |
| qwen   | 8192  | 6.383    | 156.7  |

Scaling is sub-linear: 64x kvLen (128->8192) gives ~9.7x time (SmolLM) / ~8.1x (Qwen).
The attention compute grows linearly with kvLen but the O-projection (GEMV) is kvLen-independent
and dominates at small kvLen. At kvLen=8192, attention is ~5.6ms (SmolLM) which is ~33% of a
17ms token — reasonable, not a bottleneck.

### Decision

- **No flash-decoding needed.** The split path scales sub-linearly and does not degrade
  disproportionately at large kvLen. The O-projection dominates at small kvLen; attention
  grows linearly but stays within budget.
- **Keep capability-gated selection.** Fused where device supports it (maxWG >= numHeads*64),
  split otherwise. No kvLen-based switching.
- **Selection rule documented** in docs/SILK.md section 13d.

## Planned commits (executed)

1. `docs: plan Silk decode attention long-kvLen shape (#1467) in TODO.md`
2. `perf(silk): add decode attention + O-proj kvLen sweep measurement (#1467)`
3. `test(gpu): add fused decode attention O-proj parity across kvLens (#1467)`
4. `docs(silk): document decode attention selection rule (#1467)`
5. `perf: capture silk-kernels gate output for #1467 (vCurrent)`

## Blast radius

- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — dispatch logic (unchanged)
- `src/Nivara.Extensions/Silk/SilkRuntime.cs` — capability gate (unchanged)
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — new measurement section
- `tests/Nivara.Tests/Gpu/SilkDecodeAttentionParityTests.cs` — new fused parity test
- `docs/SILK.md` — new section 13d
- `release/vCurrent/silk-kernels.txt` — benchmark artifact

## GitHub issues log

(none — no deferred work found)