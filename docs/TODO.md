# Plan: Silk Laya head trainer (issue #1148)

## Problem

Nivara has no Silk trainer for the Laya decision head. The head trains on CPU via AutoDiff
(`LayaHeadModel` / `LayaHeadLayer`) on frozen ModernBERT features. This ticket delivers a
device-side trainer mirroring `SilkDistilBertTrainer`, with a BF16 narrow lane and a parity
gate against the CPU AutoDiff head on a seeded fixture. Encoder stays frozen.

## Proposed changes

1. `src/Nivara.Extensions/Silk/SilkLayaHeadWeights.cs` (internal) — tracked trainable weight
   set for the head's key space, with F32 and narrow (BF16/FP16) constructors, AdamW buffers,
   `Trainable` flags, `Export`/`ExportGradients`.
2. `src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.cs` (sealed partial, IDisposable) —
   per-question forward/backward, AdamW update, export.
3. `src/Nivara.Extensions/Silk/SilkLayaHeadTrainer.Bf16.cs` (partial) — `CreateTrueBf16`
   factory and narrow constructor wiring.
4. Tests + seeded fixture helper:
   - `tests/Nivara.Tests/Gpu/SilkLayaHeadTrainerTests.cs`
   - `tests/Nivara.Tests/Gpu/SilkLayaHeadTrainerBf16ParityTests.cs`
   - `SilkLayaHeadBf16Fixtures` helper
5. Docs: update `docs/plan/LAYA-HEAD-TRAINING.md` status.

## Blast radius

- New files only; no changes to existing public APIs. Exports compatible with
  `LayaDecisionHead.LoadWeights(tensors, headLayerCount, actClassCount)`.
- Reuses existing kernels; no new kernels unless activation-level BF16 lands first.
- Tests under `Category("Gpu")`.

## Verification steps

- Build `Nivara.Extensions`.
- Targeted tests for new fixtures.
- Docs snippet gate if docs changed.

## Planned commits

1. docs: plan #1148 in TODO.md
2. feat(silk): add SilkLayaHeadWeights
3. feat(silk): add SilkLayaHeadTrainer (F32)
4. feat(silk): add BF16 narrow lane for Laya head trainer
5. test(gpu): add Silk Laya head trainer parity gate
6. docs: mark Laya head training shipped

## GitHub issues log

- (none yet)
