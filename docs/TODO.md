# Plan: NivaraFineTuning BF16 training exposure (#1279)

## Problem

`NivaraFineTuning` trains DistilBERT on SST-2. CPU path (`RunTrain`) always loads F32 weights
(`SafeTensorsLoader.Read`); Silk path (`RunTrainSilk`) also loads F32 and constructs
`SilkDistilBertTrainer` in F32 mode. There is no `--precision` flag. However
`SilkDistilBertTrainer` already ships true BF16 via `CreateTrueBf16`, and the CPU AutoDiff path
has no BF16 fine-tune sample. Capability exists but unreachable from the sample.

## Proposed changes

1. Promote reusable DistilBERT tensor-loading to `Nivara.Samples`:
   - Add `DistilBertLoader.LoadModel<T>(modelDir, numClasses=2)` reading safetensors as `T`.
   - Refactor `NivaraInference.DistilBertSst.Load<T>` to delegate.
2. `samples/NivaraFineTuning/Program.cs`:
   - Add `Precision` (default `f32`) to `CliArgs`, parse `--precision`, validate `f32|bf16`.
   - Dispatcher `RunTrain` -> `RunTrainCore<float>` / `RunTrainCore<BFloat16>`.
   - Generic `RunTrainCore<T>` mirroring float body (weights via promoted helper, int-overload
     `Forward`, `CrossEntropyLoss<T>`, `AdamW<T>`, generic `Evaluate<T>`/`ArgMax<T>`).
   - Silk: F32 CPU model + `ToSilkTensors()` for F32 trainer; for `bf16` load
     `SafeTensorsLoader.Read<BFloat16>` and use `SilkDistilBertTrainer.CreateTrueBf16`.
     Export still produces F32 checkpoint.
3. Update `samples/NivaraFineTuning/README.md`.

## Verification

- `dotnet build Nivara.slnx` (sample builds).
- CPU smoke: `--mode train --precision bf16 --max-examples 25 --epochs 1 --batch-size 2`.
- Silk smoke (if device available).
- Existing `SilkDistilBertTrainerTrueBf16SmokeTests`.

## Planned commits

1. docs: plan #1279 in TODO.md
2. feat(samples): promote DistilBertLoader.LoadModel<T>
3. feat(NivaraFineTuning): add --precision flag and CPU BF16 path
4. feat(NivaraFineTuning): Silk BF16 via CreateTrueBf16
5. docs: README precision usage
6. docs: remove TODO.md

## Blast radius

- `samples/Nivara.Samples/DistilBertModel.cs` (DistilBertLoader) - new helper; low risk.
- `samples/NivaraInference/DistilBertSst.cs` - refactor to delegate; behavior unchanged.
- `samples/NivaraFineTuning/Program.cs` - large; CPU generic + Silk precision.
- Hidden CLI tests (if any) expect `--precision`.

## GitHub issues log

(none yet)
