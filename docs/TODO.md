# Plan: rolling-mean cost gate (#1027 stacked PR)

## Problem

Parity exists for the Silk rolling-mean kernel, but there is no recorded CPU-vs-Silk
timing ratio. Issue #1026 asks for a benchmark gate before wiring.

## Proposed changes

1. Add `RollingMeanSilkCostGateTests` in `tests/Nivara.Tests/Incident`.
2. Measure CPU `NivaraColumn<int>.RollingMean` vs `SilkAnalytics.RollingMean`.
3. Use warmup + interleaved A/B pairs and report median ratio.
4. OpenCL-unavailable hosts use `Assert.Ignore`.
5. No sample or production changes; kernel remains unwired.

## Verification
- Build tests.
- Run Gpu-category test slice.
- Format check.

## GitHub issues log
- #1026 Phase B parity+benchmark
- #1027 Cost gate (this branch, stacked on #1027)