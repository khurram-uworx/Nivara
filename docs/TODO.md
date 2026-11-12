# Plan: Fix #1159 — frame column ownership on disposal

## Problem

`NivaraFrame.Dispose()` disposes every `IColumn` it holds. Column-set operations
(`WithColumn`, `SelectColumns`, `WithoutColumn`, `RenameColumn`/`SelectAndRename`,
`Where`/`FilterByMask`, `Skip(0)`, identity `ReorderByIndices`, carried columns in
`Normalize`) reuse the same `IColumn` instances as the source, so disposing a
derived frame invalidates the source (and disposing the source invalidates
borrowers that share instances). No ownership contract or test covers it.

## Proposed changes

Implement Option A from the issue: per-column ownership.

1. `Interfaces.cs`: add `bool IsDisposed { get; }` to `IColumn`.
2. `NivaraColumn.cs`: implement `IsDisposed`.
3. `NivaraFrame.cs`:
   - Add `readonly HashSet<IColumn> ownedColumns` (reference equality).
   - Constructor `NivaraFrame(namedColumns, bool ownsColumns = true)`.
   - Internal constructor `NivaraFrame(namedColumns, IEnumerable<IColumn> ownedColumns)`.
   - `Dispose()` disposes only owned columns; `Debug.Assert(!column.IsDisposed)` before
     disposing each owned column.
   - `WithColumn(name, column, ownsColumn = true)`.
   - `WithoutColumn`, `SelectColumns` construct with `ownsColumns: false`.
   - `Skip(0)` and identity `ReorderByIndices` construct with `ownsColumns: false`.
   - Update `Dispose` XML docs.
4. `NivaraFrameExtensions.cs`:
   - `SelectAndRename` constructs with `ownsColumns: false`.
   - `RenameColumn` calls `WithColumn(newName, column, ownsColumn: false)`.
   - `Normalize` owns only normalized (new) columns.
5. `ARCHITECTURE.md`: document ownership rule.
6. (Optional) ADR-011 + AGENTS.md entry.

## Test coverage

New fixture `NivaraFrameOwnershipTests` with both disposal directions for sharing
ops, fresh-column ops, mixed ownership, idempotency, and a reflection-based
allow-list guard over public `NivaraFrame` methods returning `NivaraFrame`.

## Blast radius

- `src/Nivara/Interfaces.cs`, `NivaraColumn.cs`, `NivaraFrame.cs`
- `src/Nivara/NivaraFrameExtensions.cs`
- `ARCHITECTURE.md`, possibly `docs/adr/011-...md`, `AGENTS.md`
- Tests: new + `DocsCoveragePins` if doc count changes

## Planned commits

1. docs: plan #1159 in TODO.md
2. feat: add IsDisposed to IColumn/NivaraColumn
3. feat: per-column ownership in NivaraFrame
4. feat: mark borrowing ops in NivaraFrame/NivaraFrameExtensions
5. docs: ownership rule in ARCHITECTURE.md (+ADR)
6. test: frame ownership disposal coverage
7. docs: remove TODO.md

## GitHub issues log

(none yet)
