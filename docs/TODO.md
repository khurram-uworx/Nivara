# TODO: Fix in-memory query source column aliasing (#279 + second aliasing hole)

## Problem

`NivaraFrame.AsQueryFrame()` builds a `QueryFrame` whose `IQuerySource` is a
`MemoryQuerySource` holding the **source frame's own column instances**. The
query pipeline transfers ownership of columns into result frames, so disposing
either side breaks the other.

### Hole 1 (issue #279, reported)

```csharp
using (var frame = NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { 1, 2, 3 }))))
{
    using var query = frame.AsQueryFrame();
    query.Collect();
} // <- disposing the QueryFrame disposed frame's "A" column
```

`MemoryQuerySource.Dispose()` disposes each stored column. Since those are the
source frame's columns, disposing the `QueryFrame` marks them disposed — the
frame throws `ObjectDisposedException` on later use.

### Hole 2 (aliasing via Collect results, folded into this plan)

```csharp
var frame = ...;
using var query = frame.AsQueryFrame();
using (var result = query.Collect()) { }   // no-op Collect: result wraps the frame's column instances
// frame.GetColumn<int>("A")[0] -> ObjectDisposedException
```

`MemoryQuerySource.Execute()` returns the shared dictionary, and the executor
builds `new NivaraFrame(columns)` over those instances, so a result frame owns
(and disposes) the source frame's columns. This is **broader than the no-op
case**: `FusedExpressionEvaluator` returns the *same column instance* for a bare
`ColumnReference` (`Expressions/FusedExpressionEvaluator.cs:221-227`), so
`frame.AsQueryFrame().Select("A").Collect()` also shares the frame's "A" column.

## Root cause

`MemoryQuerySource` is a *non-owning view* (it shares the caller's columns) but
behaves as an *owner*: `Dispose()` destroys the shared columns, and `Execute()`
hands the shared instances to result frames that then own them. The pipeline
assumes `Execute()` results are safe to transfer ownership of.

## Proposed fix (contained in `src/Nivara/Query/MemoryQuerySource.cs`)

Make `MemoryQuerySource` consistently non-owning:

1. **`Execute()` returns fresh column instances** via `Slice(0, Length)`
   (`NivaraColumn.Slice` -> `ColumnStorage.Slice` creates a *new* storage
   instance with its own `disposed` flag over the same backing array — zero
   data copy, independent disposal). Result frames therefore own disposal of
   the slices, never the source frame's storages. Repeated `Collect()` calls
   each get independent slices, so results never alias each other either.
   The default `IQuerySource.ExecuteAsync` delegates to `Execute()`
   (`Query/IQueryInterfaces.cs:28-33`), so the async path is covered too.

2. **`Dispose()` no longer disposes the stored columns** — it only marks the
   source disposed (subsequent `Execute()` throws `ObjectDisposedException`),
   leaving the source frame's columns untouched.

```csharp
public IReadOnlyDictionary<string, IColumn> Execute()
{
    ObjectDisposedException.ThrowIf(disposed, this);

    // Fresh instances over the same backing storage (zero-copy slices) so result
    // frames own independent disposal. Never share the caller's column instances.
    var fresh = new Dictionary<string, IColumn>(columns.Count, StringComparer.OrdinalIgnoreCase);
    foreach (var (name, column) in columns)
        fresh[name] = column.Slice(0, column.Length);
    return fresh;
}

public void Dispose()
{
    disposed = true;
}
```

Docs: update `MemoryQuerySource` XML doc (non-owning view, independent-disposal
slices) and `NivaraFrame.AsQueryFrame()` XML doc (QueryFrame is a non-owning
view; disposing it or a collected result never disposes the source frame's
columns). Add a `CHANGELOG.md` Fixed entry referencing #279.

## Blast radius

- `src/Nivara/Query/MemoryQuerySource.cs` — the only file whose behavior
  changes. Constructed exclusively by `NivaraFrame.AsQueryFrame()`.
- `src/Nivara/NivaraFrame.cs` — XML doc only.
- `CHANGELOG.md`, `docs/TODO.md` — docs.
- Downstream consumers: `QueryFrame.Collect()`, `AsStream()`, and all four
  execution strategies resolve memory sources through `Execute()`/`ExecuteAsync`
  (both covered). Values/`HasNulls`/nulls identical via shared backing arrays.
- Tests: existing query tests assert values (slices return identical values), so
  they stay green. New regression tests in `tests/Nivara.Tests/Query/QueryFrameTests.cs`.
- Perf: one tiny `ColumnStorage` + `NivaraColumn` allocation per column per
  execute; no data copy. No perf-harness scenario uses `AsQueryFrame()`.

## Verification

1. `dotnet build Nivara.slnx` (no confirmation needed).
2. `dotnet test` (ask human first): new tests in `QueryFrameTests.cs` plus the
   full query suite.

## Planned commits

1. `docs: plan AsQueryFrame ownership fix (#279) in TODO.md`
2. `fix: make MemoryQuerySource non-owning and return independent-disposal slices (#279)`
   — `src/Nivara/Query/MemoryQuerySource.cs`, `src/Nivara/NivaraFrame.cs` (XML
   doc), `CHANGELOG.md`
3. `test: pin QueryFrame/Collect disposal isolation from source frame (#279)`
   — `tests/Nivara.Tests/Query/QueryFrameTests.cs`
4. `docs: remove TODO.md — AsQueryFrame ownership fix delivered`
5. Offer push + PR (human confirms)

## GitHub issues log

- [ ] #279 — disposing a `QueryFrame` from `AsQueryFrame()` destroys the source frame's columns (created while working on phase-4 review); second aliasing hole (Collect result sharing source columns) folded into this plan per human request — no separate issue.
