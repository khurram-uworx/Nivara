# Plan: issue #1105 — test(silk): cover trainable-layer API aliases and clamping

## Problem

`SilkDistilBertTrainer` gained partial/frozen training (#1012) plus convenience
aliases (`NumTrainableLayers`, `TrainableLayers`, `FreezeEncoder`, `SetTrainableLayers`,
`FreezeLayers`, `Freeze`). Existing tests only exercise gradient parity for
head-only and top-one modes; the convenience API and clamping are untested.

Two concrete gaps:

1. `TrainableLayerCount` setter special-cases negative values into `numLayers`
   (`value < 0 ? numLayers : ...`), which contradicts the documented contract that
   values are clamped to `[0, NumHiddenLayers]`. A negative value should clamp to 0.
   (The constructor keeps the `-1` "all layers" sentinel separately.)

2. No tests pin alias equality, clamping, or "head stays trainable in every mode".

## Proposed changes

### Code fix
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs` — `TrainableLayerCount.set`:
  `int normalized = Math.Clamp(value, 0, numLayers);` (drop the `value < 0 ? numLayers` branch).
  Constructor normalization keeps `trainableLayerCount < 0 ? numLayers` sentinel.

### Tests
- `tests/Nivara.Tests/Gpu/SilkDistilBertTrainerTests.cs` (already `[Category("Gpu")]`):
  - `TrainableLayerCount_Aliases_ReturnSameValue`
  - `FreezeEncoder_SetsZeroTrainableLayers`
  - `SetTrainableLayers_ClampsAndUpdatesAliases`
  - `FreezeLayers_SemanticsAndClamping`
  - `TrainableLayerCount_ClampsNegativeAndOversized`
  - `Head_RemainsTrainableInAllModes` (gradient check via existing parity pattern)

## Verification

- `dotnet build Nivara.slnx`
- targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkDistilBertTrainer"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #1105 in TODO.md`
2. `fix(silk): clamp negative TrainableLayerCount to 0 in setter`
3. `test(silk): cover trainable-layer aliases and clamping`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- Only `SilkDistilBertTrainer.TrainableLayerCount` setter behavior changes for
  negative inputs; constructor path unchanged. No callers set the property to a
  negative value in-repo (grep confirmed). Tests in `SilkDistilBertTrainerTests`
  extended.

## GitHub issues log

- (none yet)
