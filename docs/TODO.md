# Plan: issue #1135 — BF16 DistilBERT trainer parity gates

## Problem

Issue #1135 is the follow-up to #1103/#1133/#1134. The true BF16 weight path and BF16 backward GEMM kernels now exist, but there is no parity gate proving the BF16 trainer's gradients match the F32 trainer. BERT-family/Laya BF16 trainers must stay gated behind that proof, and `docs/SILK.md` must state BF16 training status. A decision is also needed on whether `GeluBackward`, `LayerNormBackward`, `ColSum` need BF16 variants.

## Proposed changes

1. Add `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerBf16ParityTests.cs` — forward + gradient parity of the true-BF16 trainer vs the F32 trainer on the tiny DistilBERT shape, tolerance `maxAbs <= 1e-2 * (1+|ref|)`.
2. Verify the existing smoke gate `SilkDistilBertTrainerTrueBf16SmokeTests` (already green).
3. Update `docs/SILK.md` narrow-precision section: BF16 training status, gating, remaining scope (BERT/Laya not yet implemented).
4. Document the decision: `GeluBackward`/`LayerNormBackward`/`ColSum` stay F32-only for narrow (F32-activation) training; spin off if activation-level BF16 training is pursued. No code change.
5. Revise `DocsCoveragePins` if any docs citation/count changes.

## Verification steps

- `dotnet build Nivara.slnx -c Release`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkDistilBertTrainerBf16ParityTests"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkDistilBertTrainerTrueBf16SmokeTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- If docs changed: `dotnet test ... --filter "FullyQualifiedName~Nivara.Tests.Docs"`

## Planned commits

1. `docs: plan issue #1135 in TODO.md`
2. `test(silk): BF16 DistilBERT trainer forward+gradient parity gate`
3. `docs(silk): state BF16 training status and remaining scope`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- New test file only touches test assembly; internal members (`Backward`, `ExportGradients`) already exposed via InternalsVisibleTo.
- Docs change affects snippet/citation gates; revise pins if needed.
- No production code changes unless parity test surfaces a defect (then fix in trainer/weights).

## GitHub issues log

- (none yet; create if deferred work arises)
