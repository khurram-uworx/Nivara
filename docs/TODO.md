# Plan: Phase C (issue #1027, slice 1) — Silk rolling-max kernel + parity gate

## Problem

Silk analytics currently covers grouped hash reduce and rolling mean. The saturation
analysis needs `RollingMax`, so add a rolling-max kernel with the same parity gate.

## Proposed changes

1. Add `RollingMax = "rolling_max"` kernel constant and source to SilkKernels.
2. Add `SilkAnalytics.RollingMax(SilkRuntime, values, windowSize, minPeriods)`.
3. Add `RollingMaxSilkParityTests` in `tests/Nivara.Tests/Incident` with null-free
   int inputs, CPU vs Silk output comparison, window-count assertion.
4. Leave unwired until profitability is evaluated (benchmark evidence recorded in issue).

## Blast radius
- Additive Extension + test only; no sample CLI changes.

## Verification
- Build Extensions + Tests (Release)
- Run Gpu-category parity tests on OpenCL machine
- format whitespace --verify-no-changes

## GitHub issues log
- #1025 Phase A complete
- #1026 Phase B complete
- #1027 Phase C (this rolling-max slice)
- #1028 Phase D