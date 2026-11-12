# Plan: Fix unsynchronised lazy caches in ColumnStorage<T> (#1160)

## Problem

`ColumnStorage<T>` memoises `tensorView` and `hashCodes` without publication barriers. Under a weak memory model (ARM64) readers can observe partially published arrays or two different `Tensor<T>` views for the same storage. `Dispose()` nulls `tensorView` with a plain write.

## Proposed changes

- `src/Nivara/Storage/ColumnStorage.cs`: publish `tensorView` and `hashCodes` via `LazyInitializer.EnsureInitialized` + `Volatile` reads/writes; remove `hashLock`; update remarks.
- `src/Nivara/Storage/ChunkedColumn.cs`: apply volatile barriers to `materializedData`/`materializedMask` (defensive).
- Tests: add concurrency stress tests under `Category("Performance")`.

## Verification steps

- `dotnet build Nivara.slnx`
- Targeted `ColumnStorageTests` + new concurrency tests (Release)
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #1160 in TODO.md`
2. `fix(storage): publish ColumnStorage lazy caches safely`
3. `test(storage): add concurrency stress tests for lazy caches`
4. `docs: remove TODO.md — plan executed`

## Blast radius

- `ColumnStorage<T>` is foundational (every column/slice/chunk). Change is internal synchronization only; public API unchanged.
- Callers: `NivaraColumn<T>.AsTensorView`, `NivaraSeries<T>.AsTensorView`, `GradTensor<T>.AsTensor`, string hash fast paths.
- Tests: `ColumnStorageTests`, `TypeSafetyTests`, AutoDiff/tensor tests.

## GitHub issues log

- (none yet)