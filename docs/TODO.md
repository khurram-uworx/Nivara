# Plan: Issue #1052 — test-owned Qwen synthetic weight helper

## Problem

Tests should not depend on `NivaraInference.Qwen.SynthesizeTensors` (sample internals).
Beneficiaries are the Silk Qwen parity tests. Keep sample/CLI behavior intact.

## Proposed changes

- Introduce canonical test helper `QwenTestWeights` under `tests/Nivara.Tests/Qwen`.
- Preserve smoke config: 2 layers, 256 vocab, deterministic seed 1234, same scale.
- Add precision conversions: generic `Synthesize<TWeight>`, `ToBFloat16`, `ToF32`.
- Keep `SilkQwenTestWeights` as compatibility wrapper.
- Update parity tests to use `QwenTestWeights`.
- Leave `SynthesizeTensorsBiasGatingTests` marked as CLI end-to-end coverage.

## Verification

- targeted build
- targeted Silk tests
- whitespace format check

## Planned commits

1. docs: plan #1052
2. refactor/tests: add `QwenTestWeights` helper
3. test: update parity tests
4. docs: remove TODO.md (after review)

## GitHub issues log

None yet.
