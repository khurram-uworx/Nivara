# Plan: issue #952 — read-only Silk/OpenCL inference for NivaraFineTuning (eval/predict)

## Problem
`NivaraFineTuning` has no GPU path. Training stays CPU, but eval/predict should be able to use the already-validated Silk/OpenCL DistilBERT kernels for read-only inference, with OpenCL-unavailable fallback to CPU, matching `NivaraClassifier`.

## Proposed changes
- `samples/Nivara.Samples/DistilBertModel.cs`: add `ToSilkTensors()` mapping model params (registration order) to Silk HF-shaped keys.
- `samples/NivaraFineTuning/Program.cs`:
  - `CliArgs.Silk` + `--silk` parsing; reject for `train`.
  - `PrintHelp()` GPU section.
  - `RunEval(--silk)` and `RunPredict(--silk)` using `SilkBertEncoderRunner`.

## Verification
- Build sample.
- Targeted tests if any.
- Run `dotnet format ... whitespace --verify-no-changes`.

## Planned commits
1. docs: plan #952 in TODO.md
2. feat(samples): tensor-extraction helper
3. feat(samples): --silk flag + eval/predict paths
4. docs: remove TODO.md — plan executed

## GitHub issues log
- (none yet)