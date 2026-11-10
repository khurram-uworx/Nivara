# Plan: issue #1109 — wire seam-level BF16/F16 session factories into samples

## Problem
The GpuSeam opt-in BF16/F16 delegates exist, but nothing consumes them: samples still hard-code `SilkModernBertSession.Open` / `SilkLayaSession.Open`.

## Proposed changes
- Add `Nivara.AI.GpuSeamFactories` static class with static readonly delegate fields for F32/BF16/FP16 ModernBERT and Laya.
- Migrate sample entry points to use these factories.
- Keep silk CLI F32-only policy.

## Verification
- dotnet build
- targeted tests

## GitHub issues log
- (none)