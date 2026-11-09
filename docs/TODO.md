# Plan: Issue #1074 — pin mixed chunk-grid Arrow export

## Problem

`ArrowInterop.ToArrowTable` with `PreserveChunkStructure = true` has two branches that are
currently unpinned:

- `anyChunked == true` → `new Table(schema, columns)`, one `Column` per frame column from
  its own array list. This must survive differing per-column chunk counts, including a flat
  column beside chunked columns, and reconcile `RowCount`.
- `anyChunked == false` but the option is on → every column contributed a single array, which
  is appended back onto the batched `RecordBatch` route. This fallback must produce the same
  table as the default route.

`ChunkedArrowRoundTripTests` only covers equal 3-chunk grids and never exercises these paths.

## Blast radius

- Tests only: `tests/Nivara.Tests/IO/ChunkedArrowRoundTripTests.cs`.
- No production code change expected; `ArrowInterop.ToArrowTable` already implements both
  branches (verified by reading `src/Nivara.Extensions/IO/ArrowInterop.cs` lines 53-129).
- No docs pins affected (no `docs/*.md` added/removed, no citations changed).

## Proposed changes (test file only)

Add three private helpers and two tests to `ChunkedArrowRoundTripTests`:

- `NullChunk(int[] values, int nullAt)` → `ColumnStorage<int>` with a null mask only when
  `nullAt >= 0`.
- `ChunkedColumn(params (int[] Values, int NullAt)[] chunks)` → `NivaraColumn<int>` over a
  `ChunkedColumn<int>` (each chunk ≥ 2 storage, invariant preserved).
- `FlatColumn(int[] values, int nullAt = -1)` → flat `NivaraColumn<int>`.

Test A — `ToArrowTable_Preserve_MixedChunkGrid_KeepsPerColumnChunkCounts`:
- a: 2 chunks `(2,4)` rows, b: 3 chunks `(2,2,2)` rows, c: flat (6 rows), with nulls.
- Export with `Preserve`; assert `ArrayCount` (2,3,1), `RowCount == 6`, values and nulls.

Test B — `ToArrowTable_Preserve_AllFlatFallback_MatchesDefaultRoute`:
- all three columns flat, at least one null.
- Export with `Preserve` and with `null` (default route); assert `ArrayCount == 1` for all
  and identical null patterns / non-null values / row count.

## Verification steps

1. Build: `dotnet build Nivara.slnx -c Release`
2. Targeted tests: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release \
   --filter "FullyQualifiedName~ChunkedArrowRoundTripTests"` (ask before running per policy).
3. Full docs snippet gate only if docs touched — not applicable here.

## Planned commits

1. `docs: plan issue 1074 mixed chunk-grid Arrow export tests in TODO.md`
2. `test(io): pin mixed chunk-grid Arrow export` (test file only)

## GitHub issues log

- [#1074](https://github.com/khurram-uworx/Nivara/issues/1074) — pin mixed chunk-grid
  Arrow export (this issue).
- No additional issues captured.

## G1 grounding notes

- `ArrowInterop.ToArrowTable` branch logic at `src/Nivara.Extensions/IO/ArrowInterop.cs`
  lines 53-129 already implements both branches; only test coverage is missing.
- `ChunkedColumn<T>` requires ≥2 chunks (constructor throws otherwise) — callers satisfy.
- `ColumnStorage<T>(T[], ReadOnlyMemory<bool>)` mask constructor is internal but reachable
  under `InternalsVisibleTo`, as existing tests already do.