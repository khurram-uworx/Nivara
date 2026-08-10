# Plan: remove misleading `QueryFrame.ToList()` (issue #176)

## Problem

`NivaraLinqExtensions.ToList(this QueryFrame)` returns a `NivaraFrame`, not a
`List<T>` of rows. In .NET, `ToList` universally means "materialize this sequence
into a List<T>", so the name is a trap: callers reasonably expect row iteration
(`Count`, `foreach`) but get a columnar `NivaraFrame` that is not
`IEnumerable<row>`. It also diverges from the typed layer's
`NivaraQuery<T>.ToList()` (real `List<T>` of rows) and from
`NivaraQuery<T>.ToObjects()` (row list).

Decision (per issue #176 discussion): **remove** `ToList` outright (no
`[Obsolete]` shim) and add a genuinely-named row-list materializer
`ToRowList()` so callers have a real replacement before the removal.

## Changes

### 1. Remove `ToList` + add `ToRowList()` — `src/Nivara/Linq/NivaraLinqExtensions.cs`

Delete the `ToList(this QueryFrame)` method and its XML doc (currently lines
135-143). Add a `ToRowList()` extension returning `IReadOnlyList<NivaraRow>`,
reusing the `NivaraRow` construction pattern from
`NivaraFrameExtensions.Where` (`NivaraFrameExtensions.cs:989-1004`):

```csharp
/// <summary>
/// Executes the query and materializes each result row as a typed <see cref="NivaraRow"/> view
/// </summary>
/// <param name="source">The source query frame</param>
/// <returns>A read-only list of row views over the materialized frame</returns>
public static IReadOnlyList<NivaraRow> ToRowList(this QueryFrame source)
{
    ArgumentNullException.ThrowIfNull(source);

    var frame = source.Collect();
    var columns = frame.ColumnNames.Select(name => frame.GetColumn(name)).ToArray();
    var map = new Dictionary<string, int>(frame.ColumnNames.Count, StringComparer.OrdinalIgnoreCase);
    for (int i = 0; i < frame.ColumnNames.Count; i++)
        map[frame.ColumnNames[i]] = i;

    var rows = new List<NivaraRow>(frame.RowCount);
    for (int i = 0; i < frame.RowCount; i++)
        rows.Add(new NivaraRow(columns, map, i));

    return rows;
}
```

`NivaraRow`'s ctor is `internal`, accessible within `src/Nivara`.

### 2. Update the sole caller — `tests/Nivara.Tests/LinqQueryTests.cs:118`

`ChainedLinqOperations_RunCorrectly` chains `.ToList()` on a `QueryFrame`
(returned by `AsQueryFrame().Where(...).OrderBy(...).Select(...)`). Change
`.ToList()` → `.ToNivaraFrame()`. Required — otherwise the build fails
(`ToList(QueryFrame)` no longer exists; `TreatWarningsAsErrors=true` in
`Directory.Build.props:4`).

Add a new `[Test]` for `ToRowList()`: assert one `NivaraRow` per row with correct
`GetValue<T>`/indexer values and a null cell via `IsNull`.

### 3. Update docs — `docs/REVIEW.md`

- Action item #4 (line 335): `~~Remove QueryFrame.ToList() (obsolete in favor of ToNivaraFrame()).~~ — Done 2026-08-10 (issue #176)` (reworded since we removed rather than marked obsolete).
- Finding #5 (lines 143-149): note resolved — `ToList` removed, `ToRowList()` added; match the `~~...~~ Done (issue #NNN)` style used for #173/#179.

## Blast radius

- **Removed symbol:** `Nivara.Linq.NivaraLinqExtensions.ToList(QueryFrame)` — public API in core.
- **Downstream callers:** only `tests/Nivara.Tests/LinqQueryTests.cs:118` in the whole repo (verified by grep). Samples don't use the QueryFrame LINQ layer.
- **Added symbol:** `Nivara.Linq.NivaraLinqExtensions.ToRowList(QueryFrame)` — public, returns `IReadOnlyList<NivaraRow>`.
- **Tests covering the area:** `tests/Nivara.Tests/LinqQueryTests.cs` (all LINQ-layer tests).
- No PublicAPI analyzer surface file in the repo; no CHANGELOG entry required for this change.

## Verification

1. `dotnet build Nivara.slnx` (fast check; confirmed with human before long test runs).
2. Re-grep for `.ToList()` on `QueryFrame`/`NivaraFrame` across `tests/` and `samples/` — expect zero matches on `QueryFrame`.
3. `dotnet test` for the `LinqQueryTests` fixture (ask human before running).

## Planned commits

1. `docs: plan issue #176 in TODO.md`
2. `chore: add opencode investigate command` (`.opencode/command/investigate.md` — user asked to check it in)
3. `refactor: remove QueryFrame.ToList() and add ToRowList()`
4. `test: update LinqQueryTests to ToNivaraFrame; cover ToRowList()`
5. `docs: mark REVIEW.md finding #5 resolved`
6. Final: `git rm docs/TODO.md` → `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] none so far — deferred work discovered during execution gets an issue here immediately.
