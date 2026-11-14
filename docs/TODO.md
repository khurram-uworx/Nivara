# Plan: issue #1175 — row-level access cost

## Problem
`NivaraRow` per-cell access does a case-insensitive dictionary lookup + runtime type probe; `ColumnNames` allocates per read; object indexer boxes. `Where(Func<NivaraRow,bool>)` is far slower than expression path with no signature hint.

## Proposed changes
- `NivaraRow`: add shared `columnNames` array; primary constructor takes it; legacy constructor fallback. `ColumnNames` returns shared array (zero alloc) on normal path. Add `GetColumnIndex`/`TryGetColumnIndex`/`GetColumn(int/string)`.
- `Where` and `EnumerateRows`: build & share `columnNames` array.
- New `NivaraRowReader` + `NivaraColumnReader<T>` with hoisted nullable probe via `NullableColumnReader.CreateReader<T>`.
- Aliases `NivaraRowAccessor`/`NivaraColumnAccessor<T>`.
- `NivaraFrame.RowReader()`/`GetRowReader()` extensions, `NivaraRow.AsReader()`.
- `docs/LINQ.md` cost note (prose only).

## Verification
- build, targeted NivaraRowTests, format check, docs namespace tests.

## Planned commits
1. docs: plan 1175 in TODO.md
2. feat: non-allocating NivaraRow.ColumnNames + index accessors
3. feat: NivaraRowReader + NivaraColumnReader + aliases
4. docs: row-access cost note in LINQ.md
5. test: (if needed) updates
6. docs: remove TODO.md

## GitHub issues log
- (none yet)
