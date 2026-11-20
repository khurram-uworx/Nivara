# Plan: NivaraInference Laya CPU training path (issue #1343)

## Problem

`laya train` requires `--silk` (Program.cs guards print "laya train requires --silk."). There is no CPU
AutoDiff training path for the Laya decision head in NivaraInference, unlike NivaraChat's
`AdaptiveTraining`. Goal: add a CPU path so `laya train` works without `--silk`, keeping Silk as
default when `--silk` is set, with a parity gate CPU vs Silk.

## Blast radius

- samples/NivaraInference/Program.cs (two guards)
- samples/NivaraInference/Laya.Train.cs (add RunTrain)
- tests: SilkLayaHeadTrainerTests (parity gate extension, follow-up)
- No core/src changes; no docs/*.md changes except this TODO.

## Proposed changes

### Laya.Train.cs — add RunTrain (CPU, F32)

Mirror RunTrainGpu but use LayaDecisionHead<float,float> + AutoDiff. Manual upstream gradients
(sg, ag) fed as gradient seeds via Multiply/Sum/Add so parameter gradients match Silk's
Backward(sg, ag). Export checkpoint with fused in_proj keys matching LoadWeights.

### Program.cs

- Synthetic path: dispatch to RunTrain when !useSilk.
- Real path: replace guard with `if (mode == "train") return Laya.RunTrain(F32(), modelDir);`.

## Verification

- dotnet build Nivara.slnx
- targeted tests: SilkLayaHeadTrainerTests, InferenceCliArgumentTests
- dotnet format whitespace --verify-no-changes

## Planned commits

1. docs: plan #1343 in TODO.md
2. feat: add Laya.RunTrain CPU path
3. fix: wire laya train (non-silk) to RunTrain in Program.cs
4. (optional) test: parity gate CPU vs Silk

## GitHub issues log

- (none yet)
