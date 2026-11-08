# Plan: Phase B — Silk rolling-mean kernel (#1026)

## Problem

Only `grouped_hash_reduce_partial` has Silk acceleration. The degradation analysis
uses rolling mean on StatusCode. A rolling-mean kernel is needed to expand backend
coverage.

## Proposed changes

### 1. Kernel registration (`SilkKernels.cs`)
- Add `RollingMean = "rolling_mean"` constant.
- Add to `KernelNames`.
- Add OpenCL C kernel to `Source`.

### 2. Host driver (`SilkAnalytics.cs`)
- Add `RollingMean(SilkRuntime, input, windowSize, minPeriods?)`.
- Return float[] means + bool[] valid.
- Preserve no-null, full-window v1 scope; document partial behavior.

### 3. Parity tests (`RollingMeanSilkParityTests.cs`, Gpu category)
- Compare Silk output against `NivaraColumn<int>.RollingMean`.
- Cover edge sizes, tolerances.

### 4. Benchmark
- Record CPU vs Silk timings (warmup + interleaved A/B).
- Keep unwired unless profitable.

## Blast radius

- Additive only in Silk analytics + tests.

## Verification

- Build and run Gpu parity tests.
- Format check.

## GitHub issues log

- #1026 — Phase B
- #1027 — Phase C
- #1028 — Phase D