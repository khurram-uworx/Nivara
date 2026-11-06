# Plan: Fill Silk `compare_diag` mode parity gap

## Problem

`compare_diag` exists on the CPU path for `modernbert`, `mobilenet_v2`, and `resnet18`,
but it is explicitly rejected when `--silk` is passed. The diagnostics kernels are CPU/AutoDiff
step dumps; the Silk runners expose only final logits/hidden state, so the gap is not a new
GPU diagnostic algorithm but the CLI guard. Filling it means `--silk compare_diag` should run
the existing CPU diagnostic path instead of being rejected, while retaining honest `--silk`
parity (Silk-backed inference still runs where the guard is the only blocker).

## Proposed changes

- Add `compare_diag` to `SilkModes` for `modernbert`, `mobilenet_v2`, `resnet18`.
- In `samples/NivaraInference/Program.cs` dispatch, when `useSilk && compareDiag`, run the existing
  `ModernBert.CompareDiag` / `RunCompareDiag` CPU diagnostic paths rather than a Silk runner.
- Update CLI argument tests that used `modernbert --silk compare_diag` as the rejected vehicle.
- Update docs/SILK.md and docs/ACCELERATION.md mode tables/commentary.

## Blast radius

- samples/NivaraInference/Program.cs dispatch, SilkModes.
- ModernBert.CompareDiag and RunCompareDiag (unchanged signatures, now reachable under --silk).
- tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs rejection vehicles/tests.
- docs/SILK.md, docs/ACCELERATION.md.

## Planned commits

1. docs: plan Silk compare_diag parity in TODO.md
2. feat: allow --silk compare_diag by routing to CPU diagnostics
3. test: update InferenceCliArgumentTests vehicles for Silk compare_diag
4. docs: update Silk compare_diag parity wording

## GitHub issues log

- none yet
