# Plan: canonicalize partial/frozen training API surface (#1106)

Follow-up to #1012 / PR #1101 and #1105.

## Problem

`SilkDistilBertTrainer.TrainableLayerCount` is the canonical state for partial/frozen training,
but the same state is reachable through several aliases and helper methods:

- Properties: `NumTrainableLayers`, `TrainableLayers`
- Methods: `FreezeEncoder`, `SetTrainableLayers`, `FreezeLayers`, `Freeze`
- Profiling alias: `BackwardPhaseMs => LastPhaseMs`

This sprawls the public surface and invites inconsistent usage. Nothing is released yet, so the
redundant surface can be removed cleanly.

## Proposed changes

1. Remove the redundant aliases/methods from `SilkDistilBertTrainer`:
   - `NumTrainableLayers`, `TrainableLayers`
   - `FreezeEncoder`, `SetTrainableLayers`, `FreezeLayers`, `Freeze`
   - `BackwardPhaseMs`
2. Keep `TrainableLayerCount` (clamped to `[0, NumHiddenLayers]`, negative -> 0) and `LastPhaseMs`.
3. Keep constructors and the `trainableLayerCount` parameter (including the `freezeEncoder` overload).
4. Update `docs/SILK.md` partial/frozen section to name `TrainableLayerCount` as canonical and
   document the removed members for migration.
5. Update #1105 tests to drop alias/method coverage; keep clamping + head-trainable coverage.

## Blast radius

- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs` only (partial class; `.Bf16`/`.Fp16`
  only forward constructors, no aliases).
- `docs/SILK.md`
- `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerTests.cs`
- No other in-repo callers (grep clean).

## Verification

- `dotnet build src/Nivara.Extensions/Nivara.Extensions.csproj -c Release`
- Targeted GPU tests (requires OpenCL device)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- Docs snippet gate

## Planned commits

1. docs: plan #1106 in TODO.md
2. refactor(silk): remove redundant partial/frozen training aliases
3. docs(silk): name TrainableLayerCount as canonical entry point
4. test(silk): drop coverage of removed aliases/methods
5. docs: remove TODO.md — plan executed

## GitHub issues log

(none)