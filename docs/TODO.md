# Plan: Wire `--synthetic-weights` for NivaraInference (issue #1233)

## Problem

`--synthetic-weights` is referenced in `samples/NivaraInference/Program.cs` (error message) and in
`docs/QWEN.md`, and `ConsumesNarrow(..., synthetic:)` already accepts a `synthetic` parameter, but
the flag is never parsed: `ConsumesNarrow(..., synthetic: false)` is always used and
`Qwen.RunSyntheticBenchmarkSilk` is never invoked from `Program.cs`. The documented no-checkpoint
synthetic benchmark escape hatch does not exist.

## Proposed changes

### `samples/NivaraInference/Program.cs`

1. Parse `--synthetic-weights` as a boolean flag.
2. Add `bool syntheticWeights = false;` declaration.
3. Model-scope guard: qwen only.
4. Mode-scope guard: default run or `benchmark` only.
5. Pass `synthetic: syntheticWeights` to `ConsumesNarrow`.
6. Dispatch branch before model-file lookup:
   - Silk: `Qwen.RunSyntheticBenchmarkSilk(plain)`
   - CPU native: `Qwen.RunSyntheticBenchmark<float, BFloat16>(plain)`
   - CPU bf16: `Qwen.RunSyntheticBenchmark<BFloat16, BFloat16>(plain)`
   - CPU f32: `Qwen.RunSyntheticBenchmark<float, float>(plain)`
7. Add one help-text line under Qwen options.

### Tests

Extend `tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs` with synthetic-weights scope and
accepted-path coverage, mirroring the existing `--plain` pins.

## Verification

- `dotnet build Nivara.slnx`
- Targeted: `InferenceCliArgumentTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. docs: plan #1233 in TODO.md
2. feat(samples): wire --synthetic-weights flag
3. test(inference): pin --synthetic-weights CLI scope
4. docs: remove TODO.md — plan executed

## GitHub issues log

- (none yet)
