# Plan: test(silk) — loss-decreasing smoke test for SilkDistilBertTrainer (#962)

## Problem

#956 added `SilkDistilBertTrainer` with forward parity and gradient parity tests against reverse-mode AutoDiff, but there is no end-to-end training test verifying that the AdamW step actually reduces loss. The gradient test proves the math; a loss test proves the optimizer wiring. Existing classifier tests (`GpuTextLinearClassifierTests.Train_LossDecreases_OnSeparableData`, `GpuTextConvClassifierTests`) follow this pattern and it has caught wiring bugs before.

## Proposed changes

Add `SilkDistilBertTrainerTests.Train_LossDecreases` in `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerTests.cs`:

- Reuse `Make()` (2-layer / hidden-32 / 4-head / vocab-64 / seqLen-8 config, `maxBatch: 4`, `maxSeqLen: 16`).
- Build a separable toy dataset: label `i % 2`, first token `2` for class 0 and `5` for class 1, remaining positions random, all attention masks `1`.
- Run `trainer.Train(...)` for 4 epochs, batch 4, LR 0.05, weight decay 0.01, seed 41.
- Assert epoch count, finite losses, and `epochs[^1].Loss < 0.5f * epochs[0].Loss`.

No changes to trainer or kernel code.

## Blast radius

- Only test file modified. No production code, no docs pins, no public API.
- Test is `[Category("Gpu")]` via the class; skips when no OpenCL device via `Device()`.

## Verification steps

1. `dotnet build Nivara.slnx -c Release`
2. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkDistilBertTrainerTests.Train_LossDecreases"`
3. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #962 in TODO.md`
2. `test(silk): add loss-decreasing smoke test for SilkDistilBertTrainer`

## GitHub issues log

- (none yet)
