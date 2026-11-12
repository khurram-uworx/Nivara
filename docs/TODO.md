# Plan: Issue #1158 — Empty materialisation changes a nullable column's CLR type

## Problem
`FilterByMask` (all-false), `Take(0)`, `Skip(RowCount)` and the query/streaming empty paths rebuild columns via
`ColumnFilterHelper.CreateEmptyColumn(originalColumn.ElementType)`. `ElementType` strips nullable-ness
(`Nullable.GetUnderlyingType`), so a `NivaraColumn<int?>` becomes `NivaraColumn<int>`. Runtime type changes while
`Schema`/`ElementType` stay `int`, so `GetColumn<int?>` throws `ColumnTypeMismatchException` on empty results.

## Proposed changes
1. Add `internal IColumn CreateEmpty()` to `IColumn` (Interfaces.cs).
2. Implement in `NivaraColumn<T>`: `NivaraColumn<T>.Create(Array.Empty<T>())`.
3. Replace empty-materialisation call sites with `column.CreateEmpty()`:
   - `NivaraFrame.cs` (4 sites)
   - `Operations/SliceOperation.cs`
   - `Execution/StreamingWindowProcessor.cs`
4. Keep `ColumnFilterHelper.CreateEmptyColumn(Type)` for type-only callers; optionally add nullable-aware overload.
5. Add regression + invariant tests.

## Blast radius
- `src/Nivara/Interfaces.cs` (IColumn)
- `src/Nivara/NivaraColumn.cs`
- `src/Nivara/NivaraFrame.cs`
- `src/Nivara/Operations/SliceOperation.cs`
- `src/Nivara/Execution/StreamingWindowProcessor.cs`
- Tests: `ColumnFilterHelperTests`, `NivaraFrameFilteringSlicingTests`, new invariant test.

## Verification
- Build `dotnet build Nivara.slnx`
- Targeted tests (ask before running)
- Repro from issue prints `EMPTY GET<int?>: OK len=0`

## Planned commits
1. docs: plan #1158 in TODO.md
2. feat: add IColumn.CreateEmpty preserving runtime type
3. fix: use CreateEmpty in frame empty-materialisation paths
4. fix: use CreateEmpty in SliceOperation and StreamingWindowProcessor
5. test: invariant and regression tests for empty materialisation
6. docs: remove TODO.md

## GitHub issues log
- (none yet)
