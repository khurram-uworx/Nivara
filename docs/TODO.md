# TODO — #208: `GetGraphInfo` returns `Dictionary<string, object>` (boxing values)

## Problem

`GradientUtils.GetGraphInfo` (`src/Nivara/AutoDiff/Utilities/GradientUtils.cs:291`)
and `ComputationGraph.GetGraphInfo` (`src/Nivara/AutoDiff/ComputationGraph.cs:163`)
both return `Dictionary<string, object>`, boxing all `int`/`bool` values into
`object` and forcing consumers to cast (`(int)info["TotalNodes"]`).

## Proposed change

Return a strongly-typed `readonly record struct GraphInfo`:

```csharp
public readonly record struct GraphInfo(
    int TotalNodes,
    bool IsLeaf,
    bool RequiresGrad,
    IReadOnlyDictionary<string, int> OperationCounts);
```

- New public type in `Nivara.AutoDiff.Utilities` (public because
  `GradientUtils.GetGraphInfo` is public; `ComputationGraph` is internal).
- `ComputationGraph.GetGraphInfo` → returns `GraphInfo`.
- `GradientUtils.GetGraphInfo` → returns `GraphInfo`.
- `GradientUtils.PrintGraphSummary` → typed access
  (`info.TotalNodes`, `info.IsLeaf`, `info.RequiresGrad`, `info.OperationCounts`).

## Blast radius

| File | Change |
|------|--------|
| `src/Nivara/AutoDiff/Utilities/GraphInfo.cs` | **new** — `GraphInfo` record struct |
| `src/Nivara/AutoDiff/ComputationGraph.cs` | return type `Dictionary<string, object>` → `GraphInfo` |
| `src/Nivara/AutoDiff/Utilities/GradientUtils.cs` | `GetGraphInfo` return type → `GraphInfo`; `PrintGraphSummary` typed access |
| `tests/Nivara.Tests/AutoDiff/GradientUtilsTests.cs` | 5 call sites (`info["TotalNodes"]`, `["IsLeaf"]`, `["RequiresGrad"]`, `["OperationCounts"]`) → typed |
| `tests/Nivara.Tests/AutoDiff/LossTests.cs` | 1 call site (`info["OperationCounts"]` cast) → typed |
| `samples/Nivara.SampleApp/AutoDiffExample.cs` | `graphInfo["TotalNodes"]` / `["IsLeaf"]` → typed |
| `docs/AUTODIFF.md` | update `GetGraphInfo` notes (lines 278, 1184, 1470-1472) |

All existing `using Nivara.AutoDiff.Utilities;` imports are already present in
tests/sample, so no import changes needed.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test` (ask human first) — `tests/Nivara.Tests/AutoDiff/` covers
  `GradientUtilsTests` and `LossTests` which exercise the changed API.

## Commits

1. `docs: plan #208 in TODO.md`
2. `refactor: return typed GraphInfo record from GetGraphInfo`
3. `test: update GetGraphInfo consumers to typed GraphInfo access`
4. `docs: document GraphInfo return in AUTODIFF.md` (and remove TODO.md if plan complete)

## GitHub issues log

- None yet. As work executes, any deferred work / concern discovered must be
  created immediately via `gh issue create --repo khurram-uworx/Nivara` and the
  number recorded here — don't rely on memory.
