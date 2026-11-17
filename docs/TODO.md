# Plan: issue #1278 — NivaraInference: wire SilkLayaHeadTrainer into a user-facing path

## Problem
`SilkLayaHeadTrainer` (F32 + true BF16 via `CreateTrueBf16`) ships in `src/Nivara.Extensions/Silk/` and is exercised by the smoke tests, but is not wired into any sample or CLI. The only Laya head training path is CPU AutoDiff in `samples/NivaraChat` (`AdaptiveTraining.cs`). Goal: expose Laya head training on the Silk backend from a user-facing entry point, mirroring `NivaraFineTuning --silk`, with a parity gate vs the CPU AutoDiff head and docs.

## Proposed changes
1. Phase 1 — Parity tests (F32 + BF16) under `tests/Nivara.Tests/Gpu/`.
2. Phase 2 — CLI entry point: a `train` mode for `laya` in `samples/NivaraInference` running the Silk trainer on a seeded fixture, exporting + reloading checkpoint.
3. Phase 3 — Docs update (`docs/LAYA.md`, sample README).

## Verification
- `dotnet build Nivara.slnx`
- Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkLayaHead"`
- Docs snippet gate if docs touched.

## Planned commits
1. docs: plan #1278 in TODO.md
2. test: add SilkLayaHeadTrainerTests (F32 parity vs CPU AutoDiff head)
3. test: add SilkLayaHeadTrainerBf16ParityTests (BF16 vs F32)
4. feat: add `laya train` Silk entry point
5. docs: document Laya head training command
6. docs: remove TODO.md

## Blast radius
- tests/Nivara.Tests/Gpu/* (new)
- samples/NivaraInference/Laya.cs, Program.cs (CLI dispatch)
- tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs (surface pin)
- docs/LAYA.md, samples/NivaraInference/README.md

## GitHub issues log
(none yet)
