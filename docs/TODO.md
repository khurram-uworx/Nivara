# Plan: Phase C slice 2 — Silk rolling-min kernel (#1027)

## Problem
Rolling-min is needed for symmetry with rolling-max and possible future windowing.

## Changes
1. Add `RollingMin = "rolling_min"` constant and OpenCL kernel (mirror rolling-max with `<`).
2. Add `SilkAnalytics.RollingMin` host driver.
3. Add `RollingMinSilkParityTests` and `RollingMinSilkCostGateTests`.
4. Leave unwired (gate recorded).

## Blast radius
- Additive Extension and Incident tests only.

## Verification
- Build + incident tests
- format check

## Issues
- #1027 slice 2