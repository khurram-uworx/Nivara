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

## Proposed direction

- Benchmark fused vs split decode attention+O-proj by kvLen (128/512/2k/8k synthetic positions) on target iGPUs.
- Evaluate flash-decoding split across kv sequence with reduction for long kvLen.
- Keep capability-gated selection; prefer the simplest shape that wins on measured targets.
- Preserve GQA semantics and parity.

## Acceptance criteria

- [ ] Measured kvLen sweep reported with steady-state tok/s and attn+oproj phase ms/token.
- [ ] Clear selection rule or heuristic documented (or justified default).
- [ ] Parity tests for selected path(s), including GQA and long kvLen.

## Code-memory symbol navigation

Key symbols:
- `Nivara.Silk.SilkLauncher.DecodeGqaAttentionOproj` (SilkLauncher.cs:507-521) — dispatch
- `Nivara.Silk.SilkRuntime.CanFuseDecodeOproj` (SilkRuntime.cs:197-210) — capability gate
- `Nivara.Silk.SilkKernels.DecodeGqaAttentionOprojFused` (SilkKernels.cs:151) — fused kernel
- `Nivara.Silk.SilkKernels.DecodeGqaAttention` (SilkKernels.cs:3102) — split attention
- `Nivara.Silk.SilkKernels.GemmGemvAdd` (SilkKernels.cs:146) — flat GEMV+residual
- `Nivara.Tests.Gpu.SilkDecodeAttentionParityTests` — existing parity tests (split only)

## Recent relevant history

- #1486 (K-split decode GEMV subgroup reduction): template for kv/k-split — split contraction
  across subgroup lanes, `sub_group_reduce_add`, tolerance parity, capability-gated subgroup
  variant, GB/s reporting.
- #1483 (fused MLP block GB/s): established logical-bytes-moved reporting pattern.
- #1482 (prefill GEMM GB/s/GFLOP/s): throughput reporting pattern.
- #1453/#1465 (MLP decomposition): exposed attention+O-proj as next profiling target.

## Plan

### Phase 1 — Capture fresh baseline

1. Run existing `--silk-kernels` gate, save to `release/vCurrent/silk-kernels.txt`.
   Command: `dotnet run -c Release --project tests\Nivara.PerformanceTests -- --silk-kernels > release\vCurrent\silk-kernels.txt`
   (run backgrounded, no polling; wait for completion).

### Phase 2 — Add decode-attention kvLen sweep measurement

2. Add a dedicated measurement section to `SilkKernelsGate` (or a new `--silk-decode-attn` flag)
   that benchmarks fused vs split `DecodeGqaAttentionOproj` at kvLen = 128, 512, 2048, 8192
   for both SmolLM (hidden 576, 9/3 heads, headDim 64) and Qwen (hidden 896, 14/2, 64) shapes.
   Report: ms/op (best-of-N after warmups), logical bytes moved, GB/s, tok/s-equiv (1000/ms).
   Follow the interleaved A/B ordering pattern from existing gate cells.

3. Run the new measurement, save to `release/vCurrent/silk-decode-attn-kvlen.txt`.

### Phase 3 — Assessment & selection rule

4. Analyze the captured numbers. Identify crossover kvLen where fused loses to split.
   Decision: keep fused (if wins across all kvLens), switch to kvLen-gated selection (if crossover
   exists), or implement flash-decoding kv-split (if split still degrades at 2k/8k).

### Phase 4 — Implementation (based on assessment)

5. Implement the selected change in `SilkLauncher.DecodeGqaAttentionOproj` / `SilkRuntime`.
   - If kvLen-gated: add `kvLen` parameter awareness or a new `CanFuseDecodeOproj(kvLen)` overload.
   - If flash-decoding: add split-kv partials kernel + reduction (patterned after #1486).
   Capability gate always first.

6. Add parity tests:
   - Fused `DecodeGqaAttentionOprojFused` vs CPU reference (attention + Wo + residual) for
     GQA shapes and long kvLen (including 4096/8192 where feasible).
   - Split-vs-fused comparison where both eligible.
   In `SilkDecodeAttentionParityTests.cs` (or new test file), `[Category("Gpu")]`.

### Phase 5 — Docs & verification

7. Update `docs/SILK.md` kernel table and selection notes with measured rule + new benchmark command.
8. Run `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
9. Run targeted tests: `SilkDecodeAttentionParityTests`, `SilkKernelsGate` (if feasible).
10. Review DocsCoveragePins if docs changed.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkLauncher.cs` — dispatch logic
- `src/Nivara.Extensions/Silk/SilkRuntime.cs` — capability gate
- `src/Nivara.Extensions/Silk/SilkKernels.cs` — potential new kernel (flash-decoding only)
- `tests/Nivara.Tests/Gpu/SilkDecodeAttentionParityTests.cs` — new parity tests
- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — new measurement section
- `docs/SILK.md` — documentation update
- `release/vCurrent/` — benchmark artifacts

Callers of `DecodeGqaAttentionOproj`: `SilkQwenGpuRunner.cs:849`, `SilkSmolLMGpuRunner.cs:815`,
`SilkTinyShakespeareGpuRunner.cs:219`. Signature change would require updating all three.

## GitHub issues log

(none yet)

## Notes on machine comparability

Performance numbers captured on different machines are not directly comparable. The decision
should be based on relative trends (fused vs split ratio) rather than absolute ms values.
The captured device info (max WG, local memory, subgroups) must be recorded alongside numbers.
