# Plan: drop ILGPU DistilBERT-SST GPU path; keep Silk (#825)

Base: `khurram/824` (`72d688d0`)
Branch: `khurram/825`

## Problem

`distilbert_sst` still reaches a GPU path through the ILGPU-backed `OpenIlgpuEncoder` /
`BertEncoderGpuRunner` (`RunDistilBertSstGpuInference`, `BenchmarkDistilBertSstGpu`,
`RunDistilBertSstGpuCompare`), plus its dispatch arm and `GpuModes` entry. Silk
(`OpenSilkEncoder` / `SilkBertEncoderRunner`) is already wired behind `--silk`, so the ILGPU
remnants are dead weight ahead of #814 (remove ILGPU entirely).

## Proposed changes

`samples/NivaraInference/Program.cs`:
- Remove `["distilbert_sst"]` from `GpuModes`; update the guard message's model list.
- Delete the `if (useGpu)` block in `case "distilbert_sst"`.
- Delete helpers `RunDistilBertSstGpuInference`, `BenchmarkDistilBertSstGpu`,
  `RunDistilBertSstGpuCompare`.
- Leave `OpenIlgpuEncoder` and other models untouched (scope is #814's job).
- Leave narrow-precision constants (`ConsumesNarrow`, `ConsumesWiden`, `NarrowFloatModels`)
  unchanged — they cover CPU/Silk paths.

`tests/Nivara.Tests/Inference/InferenceCliArgumentTests.cs`:
- Replace the `distilbert_sst --ilgpu predict` entry with a model-scope rejection:
  `(["distilbert_sst", "--ilgpu"], "--ilgpu is not supported for 'distilbert_sst'")`.

## Blast radius

- Dispatch only: `distilbert_sst --ilgpu` now exits 1 via the generic model-scope guard.
- CPU and `--silk` paths (default/benchmark/compare/predict) unchanged.
- No source changes under `src/`; only sample + test.

## Verification

1. `dotnet build` samples/NivaraInference.
2. Targeted `InferenceCliArgumentTests`.
3. Smoke `distilbert_sst --silk` default/benchmark/compare/predict where weights exist.

## Planned commits

1. `docs: plan #825 in TODO.md`
2. `refactor(inference): drop ILGPU distilbert_sst entry points`
3. `test(inference): pin distilbert_sst ilgpu rejection`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

- #825 — this issue.
- #814 — broader ILGPU removal (still open; not in scope here).
