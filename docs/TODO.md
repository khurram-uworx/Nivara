# TODO — Remove `NivaraFrame.Where(Func<dynamic, bool>)` (issue #154)

**Branch:** `khurram/154` · **Tracker:** [Nivara #154](https://github.com/khurram-uworx/Nivara/issues/154) · **Direction:** breaking change is approved (zero in-repo callers; removal aligns with Phase 2 of the POLARS-ROADMAP and the 1.2.0 removal precedent).

## Problem

`NivaraFrame.Where(Func<dynamic, bool>)` (`src/Nivara/NivaraFrameExtensions.cs:989`) is the **last public `dynamic` surface** in the core library. It builds an `ExpandoObject` per row plus a reflection `Item` lookup per element (`CreateDynamicRow`, `:1212`). The Phase 2 purge removed `dynamic`/boxed fallbacks from `ExpressionEvaluator`, `NivaraColumn<T>` arithmetic, and `NivaraSeries` Sum/Average; this is the Phase 2 follow-up that closes the last hole.

Verified blast radius:
- **Zero** callers in `src/`, `tests/`, `samples/` (all `.Where(...)` uses route through `AsQueryFrame()` or `frame.Query<T>()`).
- Docs that reference it: `ARCHITECTURE.md:452` (`row.Age > 25` — member access), `ARCHITECTURE.md:983` ("Use `ExpandoObject` for `Where()` predicates"), `GETTING-STARTED.md:884` (`row.GetValue<int>("Age")` — currently a latent runtime bug on the dynamic path).
- No `PublicAPI.*` baseline files exist; `CHANGELOG.md` is the only contract artifact.

## Decisions (confirmed)

1. **Remove the dynamic overload outright** (no `[Obsolete]` bridge) — consistent with 1.2.0 removals; breaking change approved.
2. New public **`readonly struct NivaraRow`** — allocation-free per-row (holds `IColumn[]` + name→index map + `int RowIndex`).
3. **Predicate exceptions propagate** (no `InvalidOperationException` row-index wrapper).
4. Update `docs/plan/POLARS-ROADMAP.md` as part of this change.

## Changes

### 1. New `src/Nivara/NivaraRow.cs`

```csharp
public readonly struct NivaraRow
{
    readonly IColumn[] columns;
    readonly IReadOnlyDictionary<string, int>? map;
    readonly int rowIndex;

    public int RowIndex { get; }
    public object? this[string columnName] { get; }      // IColumn.GetValue; null for null cells
    public T GetValue<T>(string columnName);              // ColumnNotFoundException / ColumnTypeMismatchException / ArgumentException
    public bool TryGetValue<T>(string columnName, out T value);
    public bool IsNull(string columnName);
}
```

- `default(NivaraRow)` (all-zero state) is valid and throws a clear `InvalidOperationException` on access (MS struct-design guidance: zero state must be valid).
- `GetValue<T>` returns the stored value on null cells, matching the `NivaraColumn<T>` indexer contract; `IsNull` reports nullness.

### 2. `src/Nivara/NivaraFrameExtensions.cs` — swap `Where`

- Delete `Where(this NivaraFrame, Func<dynamic, bool>)` (`:989-1014`) and private `CreateDynamicRow` (`:1206-1236`).
- Add `public static NivaraFrame Where(this NivaraFrame frame, Func<NivaraRow, bool> predicate)`:
  - `ArgumentNullException.ThrowIfNull` on both args.
  - Build `IColumn[]` + case-insensitive `Dictionary<string,int>` **once** per call.
  - `mask[i] = predicate(new NivaraRow(columns, map, i))`; zero per-row allocation.
  - Apply via existing `frame.FilterByMask(NivaraColumn<bool>.Create(mask))`; exceptions propagate.

### 3. Tests — new `tests/Nivara.Tests/NivaraRowTests.cs`

- Filter via `GetValue<T>` and via `(int)row["Age"]` indexer cast.
- Null handling: `row.IsNull(...)` include/exclude; null cell indexer returns `null`.
- `ColumnNotFoundException` / `ColumnTypeMismatchException` / blank-name `ArgumentException`.
- Predicate exception propagates unwrapped.
- `default(NivaraRow)` access throws clearly; null frame/predicate → `ArgumentNullException`.

### 4. Docs & roadmap

- `ARCHITECTURE.md:452` → `frame.Where(row => row.GetValue<int>("Age") > 25)`.
- `ARCHITECTURE.md:983` → drop the "Use `ExpandoObject`" note; describe the typed `NivaraRow`.
- `GETTING-STARTED.md:884` → already `row.GetValue<int>("Age")`; now correct (verify no edit needed).
- `CHANGELOG.md` (Unreleased → Breaking changes) → removal + typed replacement entry.
- `docs/plan/POLARS-ROADMAP.md` → note that the last public `dynamic` surface is removed; the "no dynamic in the public API" Phase 2 goal is closed.

## Verification

- `dotnet build Nivara.slnx` after each code step.
- Ask before running `dotnet test`; then run `NivaraRowTests` + `LinqQueryTests`/`TypedLinqTests`/`ExpressionEvaluatorTests` as regression guards.
- Grep `src/Nivara` for `dynamic`/`ExpandoObject` → expect zero remaining.

## Planned commits

1. `docs: plan issue #154 (remove dynamic frame Where) in TODO.md`
2. `feat: add public NivaraRow readonly struct`
3. `feat: replace dynamic Where with typed Where(Func<NivaraRow, bool>)`
4. `test: cover NivaraRow and typed Where`
5. `docs: update ARCHITECTURE/GETTING-STARTED/CHANGELOG/POLARS-ROADMAP for #154`

## Follow-ups

- `docs/TODO.md` removed once every item is done; offer to push `khurram/154` and open a PR.
