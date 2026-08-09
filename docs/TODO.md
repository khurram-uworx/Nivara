# Plan: Remove reflection from frame Take/Skip/Slice (issue #173)

## Problem

`NivaraFrame.Take` / `Skip` / `Slice` slice every column via **reflection**
(`GetMethod("Slice", ...)` + `MethodInfo.Invoke`) on each call — a hot path in a
library that advertises high performance. The same duplicated pattern exists in
the query engine's `SliceOperation.SliceColumn`.

- `src/Nivara/NivaraFrame.cs:50-64` — `sliceColumn` helper (reflection + dead fallback)
- Call sites: `NivaraFrame.cs:1018` (Take), `NivaraFrame.cs:1063` (Skip), `NivaraFrame.cs:1106` (Slice)
- `src/Nivara/Operations/SliceOperation.cs:105-119` — `SliceColumn` duplicate (query-engine slice path)

`IColumn` already declares `IColumn Slice(int start, int length)` at
`src/Nivara/Interfaces.cs:47`; `NivaraColumn<T>` (the sole `IColumn`
implementation in the repo) implements it via explicit interface impl
(`NivaraColumn.cs:1617`) → public `Slice` (`NivaraColumn.cs:1596`) →
`ColumnStorage<T>.Slice` (`ColumnStorage.cs:167`). The reflection is obsolete
legacy predating the v1.2.0 single-`ColumnStorage<T>` consolidation.

## Changes

### 1. Fix `NivaraFrame.sliceColumn` (src/Nivara/NivaraFrame.cs)

Replace the reflection + fallback body with a direct virtual call:

```csharp
static IColumn sliceColumn(IColumn column, int start, int length)
    => column.Slice(start, length);
```

Delete the `GetMethod` / `MethodInfo.Invoke` and the unreachable
`ColumnFilterHelper.CreateFilteredColumn` fallback. Keep the XML doc comments.
No `using` changes needed (LINQ used elsewhere in the file).

### 2. Fix `SliceOperation.SliceColumn` (src/Nivara/Operations/SliceOperation.cs)

Same replacement for the identical duplicate:

```csharp
static IColumn SliceColumn(IColumn column, int start, int length)
    => column.Slice(start, length);
```

`using Nivara.Helpers;` stays — `CreateEmptyColumn` (`SliceOperation.cs:121`)
still uses `ColumnFilterHelper.CreateEmptyColumn`.

`ColumnFilterHelper.CreateFilteredColumn` is **kept** — still used by
`DistinctOperation` (55), `FilterOperation` (100), `SelectRowsOperation` (50),
and `NivaraFrame.FilterByMask` (981).

### 3. Add Frame Slice perf scenario (tests/Nivara.PerformanceTests/Program.cs)

Add a scenario to `RegisterScenarios()` that repeatedly slices a multi-column
frame, e.g. reusing `BuildScoreFrame(int rows, int cols)` (Program.cs:327):

```csharp
Run("Frame Slice 10k x 64 cols", 5, 100,
    () =>
    {
        var frame = BuildScoreFrame(10_000, 64);
        return () => frame.Slice(0, 5_000);
    });
```

The harness already reports `BytesPerOp` / `Gen0PerOp`, so this demonstrates the
reflection `object[]` allocation disappearing.

### 4. Update docs/REVIEW.md

Mark finding #2 resolved (matching the #1 / #3 pattern) with date and a note
that the query-engine `SliceOperation.SliceColumn` duplicate was fixed too, and
update the issue tracker table row for #173.

## Verification

- `dotnet build Nivara.slnx` (build check after each step)
- `dotnet test tests/Nivara.Tests` — run **after explicit human confirmation**
  (AGENTS.md); existing `NivaraFrameFilteringSlicingTests` and `QueryFrameTests`
  cover both fixed paths.
- `dotnet build tests/Nivara.PerformanceTests` — perf scenario compiles.

## Blast radius

- `src/Nivara/NivaraFrame.cs` — `Take`/`Skip`/`Slice` behavior unchanged; only
  the mechanism changes. Public API surface unchanged.
- `src/Nivara/Operations/SliceOperation.cs` — query-engine `Slice`/`Skip`/`Take`
  path. Covered by `tests/Nivara.Tests/Query/QueryFrameTests.cs` (Skip/Take/Slice,
  incl. parallel round-trip) and `TypedLinqTests.SkipTake_SlicesRows`.
- `ColumnFilterHelper.CreateFilteredColumn` — untouched; still called by
  Distinct/Filter/SelectRows/FilterByMask (NivaraFrame.cs:981).
- Custom external `IColumn` implementations must implement `Slice(int, int)` —
  it is part of the `IColumn` contract, so a direct call is always safe.
- Perf harness: additive scenario only; `Nivara.PerformanceTests/Program.cs`.

## Commits (one logical change per commit)

1. `docs: plan issue #173 in TODO.md` — this file.
2. `perf: slice columns directly in NivaraFrame Take/Skip/Slice` —
   `NivaraFrame.cs`.
3. `perf: slice columns directly in SliceOperation` — `SliceOperation.cs`.
4. `test: add Frame Slice perf scenario` — `Program.cs`.
5. `docs: resolve REVIEW finding #2 (reflection-based frame slicing)` —
   `REVIEW.md`.
6. `docs: remove TODO.md — #173 plan executed` (final review step, optional).

## GitHub issues log

- [ ] (none created yet) — if any deferred work or concern surfaces during
  execution, file it immediately via `gh issue create --repo khurram-uworx/Nivara`
  and record the number here.
