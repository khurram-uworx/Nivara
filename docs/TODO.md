# Plan: AISTACK Phase 3 — frame-level dataset operations (#1291)

## Problem
`TensorDataset<T>` + `DataLoader<T>` exist at the training-API level, but split/shuffle/batch/stratified
operations are still hand-rolled in samples. Add them as frame operations.

## Proposed changes
- New `src/Nivara/Data/` directory.
- `TrainTestSplit` result type with `Train`/`Test` properties and `Deconstruct`.
- `FrameDatasetCore` internal implementation of `SplitTrainTest`, `Shuffle`, `Batch`/`Chunk`,
  `StratifiedSplit`.
- Public surfaces: `DatasetExtensions` (extension methods), static facades `Dataset`,
  `Datasets`, `NivaraDataset`, `FrameDataset`, and a `Nivara.Data` namespace mirror.
- Additive only; no changes to existing training APIs.

## Verification steps
- `dotnet build Nivara.slnx`
- targeted `FrameDatasetTests`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan #1291 in TODO.md
2. code: add frame-level dataset operations
3. tests: add FrameDatasetTests
4. docs: remove TODO.md — plan executed

## Blast radius
- New files under `src/Nivara/Data/` and `tests/Nivara.Tests/Data/`.
- No existing public API changed; new additive surface only.

## GitHub issues log
- (none yet)
