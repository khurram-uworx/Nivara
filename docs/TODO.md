# Plan: rolling-max CPU-vs-Silk cost gate

## Problem
Rolling-max parity is proven, but no timing evidence exists.

## Changes
1. Add `RollingMaxSilkCostGateTests` (Gpu + Performance categories).
2. Measure CPU `NivaraColumn<float>.RollingMax` vs `SilkAnalytics.RollingMax`.
3. Warmup + interleaved A/B median ratio.
4. OpenCL-unavailable hosts `Assert.Ignore`.

## Verification
- Build + run incident tests
- format check

## Issues
- #1027 (stacked)