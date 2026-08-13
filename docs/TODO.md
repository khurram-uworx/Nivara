# Plan: #209 — Private fields use `_` prefix in AutoDiff training/nn files

## Problem

AGENTS.md mandates private fields be `camelCase` without a `_` prefix. 13 AutoDiff
files violate this (the 4 listed in issue #209 plus 9 more discovered during
planning). The `.editorconfig` rule `private_fields_should_be_camel_case` uses an
empty `required_prefix`, confirming camelCase is the enforced style.

All affected members are `private`; no public API changes. `Module<T>` uses
explicit `RegisterParameters`/`RegisterModules` registries (no field reflection),
so `StateDict`/`LoadStateDict`/serialization are unaffected.

## Scope

Rename `_`-prefixed private fields to camelCase in all 13 AutoDiff files. Per
issue #209 plus the approved expanded scope.

### Rename map

| File | New field names |
|------|-----------------|
| `Training/TrainingLoop.cs` | `model, loader, lossFn, optimizer, maxEpoch` |
| `Training/DataParallelTrainer.cs` | `model, loader, lossFn, optimizer, epochs, maxDegreeOfParallelism` |
| `Training/DataLoader.cs` | `dataset, batchSize, shuffle, seed` |
| `Training/TensorDataset.cs` | `frame, featureColumns, labelColumns` |
| `Nn/LayerNorm.cs` | `normalizedShape, eps, affine, weight, bias` |
| `Nn/MultiheadAttention.cs` | `embedDim, numHeads, headDim, attnScale, causal, qProj, kProj, vProj, oProj, attnDropout` |
| `Nn/BatchNorm.cs` (BatchNorm1d + BatchNorm2d) | `numFeatures, eps, momentum, affine, trackRunningStats, weight, bias, runningMean, runningVar, numBatchesTracked` |
| `Nn/Conv1d.cs` | `inChannels, outChannels, kernelSize, stride, padding, useBias, weight, bias` |
| `Nn/Conv2d.cs` (Conv2d) | `inChannels, outChannels, kernelSize, stride, paddingTop, paddingBottom, paddingLeft, paddingRight, groups, useBias, weight, bias` |
| `Nn/Conv2d.cs` (ConvTranspose2d) | `inChannels, outChannels, kernelSize, stride, padding, useBias, weight, bias` |
| `Nn/AdaptiveAvgPool2d.cs` | `outputSize` |
| `Nn/MaxPool2d.cs` | `kernelSize, stride, padding` |
| `Nn/DepthwiseSeparableConv2d.cs` | `depthwise, pointwise` |
| `Nn/ConvVAE.cs` | `encoderConvs, muConv, logVarConv, decoderConvs, reconConv, latentChannels, spatialSize` |

### Shadowing hazards (must compile)

1. **Ctor param/field collisions** → `this.field = param;` (existing convention in
   `Adam.cs`/`SGD.cs`/`Dropout.cs`):
   - `TrainingLoop`: `model, loader, lossFn, optimizer`
   - `DataParallelTrainer`: all 6
   - `DataLoader`: all 4; `TensorDataset`: all 3
   - `LayerNorm`: `normalizedShape, eps, affine`
   - `MultiheadAttention`: `embedDim, numHeads, causal`
   - `BatchNorm` (both): `numFeatures, eps, momentum, affine, trackRunningStats`
   - `Conv1d`: `inChannels, outChannels, kernelSize, stride, padding`
   - `Conv2d` main ctor: `inChannels, outChannels, kernelSize, stride, paddingTop/Bottom/Left/Right, groups`; delegating ctor body unchanged
   - `ConvTranspose2d`: `inChannels, outChannels, kernelSize, stride, padding`
   - `AdaptiveAvgPool2d`: `outputSize`; `MaxPool2d`: all 3; `ConvVAE`: `latentChannels, spatialSize`
2. **`bias` param shadows renamed `bias` field** in `Conv1d`, `Conv2d`, `ConvTranspose2d`
   ctors → `this.bias` for the field writes/`RegisterParameters` inside the `if (bias)` block.
3. **Local `bool affine` collides with renamed `affine` field** (CS0844) → rename the
   *local* to `useAffine` (matches `useCausal` in `MultiheadAttention.cs`):
   - `LayerNorm.cs:84` (closure uses lines 95, 99)
   - `BatchNorm.cs` BatchNorm1d line 149 and BatchNorm2d lines 330, 367

### Not renamed (out of scope)
- `disposed`, `enumerationCount` — already conform.
- Non-AutoDiff core `OptimizationRule.cs`, `QueryPlanVisitor.cs` — excluded per
  scope decision; tracked as #NNN below.

## Blast radius

- **Files changed:** 13 files under `src/Nivara/AutoDiff/` (see map). No public
  API, property, or method signatures change.
- **Downstream:** none at source level (private fields). No reflection over these
  fields exists (`WeightAccessConsistencyTests.cs` uses public `GetProperty`;
  `Module<T>` parameter registry is name-based on `Parameter.Name`, not field names).
- **Test coverage:** `tests/Nivara.Tests/AutoDiff/` — `TrainingTests`,
  `DataParallelTests`, `NnTests`, `InferenceFastPathTests`, `SerializationTests`,
  `WeightAccessConsistencyTests`, plus `NivaraTorch/NormalizationTests`.
  Samples (`Nivara.Samples`, `NivaraChat`, `NivaraInference`) consume these
  modules only via public members.

## Verification

1. `dotnet build Nivara.slnx` after each commit — must be warning-clean on touched files.
2. Ask human before `dotnet test`. Target: AutoDiff suites listed above.
3. Final grep: no `\b_\w+` field declarations remain under `src/Nivara/AutoDiff`.

## Planned commits

1. `docs: plan #209 in TODO.md`
2. `refactor: rename _-prefixed fields in AutoDiff training files` — TrainingLoop, DataParallelTrainer, DataLoader, TensorDataset
3. `refactor: rename _-prefixed fields in AutoDiff norm modules` — LayerNorm, BatchNorm
4. `refactor: rename _-prefixed fields in MultiheadAttention`
5. `refactor: rename _-prefixed fields in AutoDiff conv modules` — Conv1d, Conv2d, ConvTranspose2d, DepthwiseSeparableConv2d
6. `refactor: rename _-prefixed fields in pooling/vae modules` — AdaptiveAvgPool2d, MaxPool2d, ConvVAE
7. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [x] #226 — remaining `_`-prefixed private fields in non-AutoDiff core files
      (`OptimizationRule.cs`, `QueryPlanVisitor.cs`) left out of #209 scope
      (created while working on #209).
- Reminder: as each task executes, create a GitHub issue immediately via
  `gh issue create --repo khurram-uworx/Nivara` for any deferred work/concern and
  record its number here — do not rely on memory or wait until the plan finishes.
