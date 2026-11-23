# Plan: Silk gemm_gemv_add decode O-projection GB/s measurement (#1480)

## Problem

The cached-decode O-projection path uses `gemm_gemv_add` / `SilkLauncher.GemmGemvAdd`
(#672, #681) — a Silk-only decode GEMV with residual add. It is correctness-gated in
`SilkKernelsGate.MeasureSmolLMKernels` (SmolLM 576x576 only) but is not part of the
GB/s report added in #1473. The BF16 variant (`gemm_gemv_add_bf16`) already has GB/s
coverage from #1479 (`MeasureDecodeShapeAbBf16` `gemm_gemv_add_bf16` subsection), but
the F32 path has no GB/s numbers, and the Qwen `(896,896)` O/down shape has no
correctness leg at all.

## Proposed changes

### Step 1 — Add F32 `gemm_gemv_add` GB/s measurement

Add a new method `MeasureDecodeShapeGemvAddF32` to
`tests/Nivara.PerformanceTests/SilkKernelsGate.cs`, mirroring the #1479 pattern
(`MeasureDecodeShapeAbBf16` `gemm_gemv_add_bf16` subsection) but using base
`SilkLauncher.GemmGemvAdd` (F32 weights). Wire it into `Run()` next to the existing
decode-shape GB/s calls (`MeasureDecodeShapeAb`, `MeasureDecodeShapeAbBf16`).

Shapes: reuse the 10-shape decode table (LM-head/vocab, QKV, gate/up, O/down, small,
tiny) for comparability, matching the BF16 subsection. Label rows as `aCols/bCols`.

Bytes-moved model (as specified in the issue):
```
bytes = (aCols + aCols*bCols + bCols + bCols) * sizeof(float)
      = x_read + weight_read + residual_read + output_write
GB/s  = bytes / elapsed_seconds / 1e9
```

Warmup/rounds: `warmups=5`, `rounds=10`, `Time`/`Median`, `Finish()` per launch,
matching the BF16 subsection. Base (non-subgroup) variant per gate convention.

### Step 2 — Documentation

Add a short F32 `gemm_gemv_add` bytes-model note to `docs/SILK.md` §14a, alongside
the existing BF16 `gemm_gemv_add_bf16` bytes model, so methodology stays comparable.

### Step 3 — Verification

- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet build Nivara.slnx -c Release` (perf project builds)
- Targeted perf-project build check.
- No snippet-gate impact (no new gated C# blocks).

## Planned commit list

1. `docs: plan Silk gemm_gemv_add decode GB/s measurement in TODO.md`
2. `perf: report F32 gemm_gemv_add decode GB/s in Silk kernels gate`
3. `docs: document F32 gemm_gemv_add decode GB/s measurement in SILK.md §14a`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- `tests/Nivara.PerformanceTests/SilkKernelsGate.cs` — new method + call site in `Run()`.
  No correctness-leg changes; measurement only. No unit tests depend on this file.
- `docs/SILK.md` — §14a documentation addition only.
- No production code changes; no public API changes.

## GitHub issues log

- (none created yet)
