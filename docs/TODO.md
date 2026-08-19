# Plan: Post-aggregation ranking in LINQ DSL (#306)

## Problem

`QueryFrame.DenseRank()` / `Rank()` / `RowNumber()` / `PercentRank()` exist only on `QueryFrame`, not on `NivaraQuery<T>`. After `.GroupBy().Select()` in the LINQ pipeline, users must `Collect()` and manually sort/rank with LINQ loops. This breaks the lazy pipeline and is verbose.

The `AnalyzeRegionalPartitioning` sample (`samples/Nivara.Samples/Incident/Analysis.cs`) has 8 lines of manual sorting/ranking code that should be expressible as a single `.DenseRank(...)` call.

## Root cause

`NivaraQuery<T>` (src/Nivara/Linq/NivaraQuery.cs) wraps a `QueryFrame` and exposes Where/Select/OrderBy/GroupBy/Skip/Take/Distinct, but **zero** window function methods. The execution pipeline already handles GroupBy → window ops in sequence (GroupByOperation + RankOperation work correctly when chained in a QueryPlan), but the LINQ-level API never exposes this path.

## Proposed changes

### Step 1: Add window function methods to `NivaraQuery<T>`

File: `src/Nivara/Linq/NivaraQuery.cs`

Add four methods (DenseRank, Rank, RowNumber, PercentRank) with the same overloads that QueryFrame offers:

```csharp
public NivaraQuery<T> DenseRank(string resultColumn, IReadOnlyList<SortKey> orderBy, string[]? partitionBy = null)
    => new(frame.DenseRank(resultColumn, orderBy, partitionBy ?? []));

public NivaraQuery<T> DenseRank(string resultColumn, WindowSpec spec)
    => new(frame.DenseRank(resultColumn, spec));

public NivaraQuery<T> Rank(string resultColumn, IReadOnlyList<SortKey> orderBy, string[]? partitionBy = null)
    => new(frame.Rank(resultColumn, orderBy, partitionBy ?? []));

public NivaraQuery<T> Rank(string resultColumn, WindowSpec spec)
    => new(frame.Rank(resultColumn, spec));

public NivaraQuery<T> RowNumber(string resultColumn, IReadOnlyList<SortKey> orderBy, string[]? partitionBy = null)
    => new(frame.RowNumber(resultColumn, orderBy, partitionBy));

public NivaraQuery<T> RowNumber(string resultColumn, WindowSpec spec)
    => new(frame.RowNumber(resultColumn, spec));

public NivaraQuery<T> PercentRank(string resultColumn, IReadOnlyList<SortKey> orderBy, string[]? partitionBy = null)
    => new(frame.PercentRank(resultColumn, orderBy, partitionBy ?? []));

public NivaraQuery<T> PercentRank(string resultColumn, WindowSpec spec)
    => new(frame.PercentRank(resultColumn, spec));
```

Each delegates to the existing `QueryFrame` method, preserving the lazy pipeline.

### Step 2: Add tests

File: `tests/Nivara.Tests/Query/PostGroupRankTests.cs` (new)

Tests covering:
- GroupBy → Select → DenseRank → Collect produces correct ranks
- GroupBy → Select → Rank → Collect (standard rank with ties)
- GroupBy → Select → PercentRank → Collect
- Post-group DenseRank with no partition (single partition over all aggregated rows)
- Post-group DenseRank with ties produces no gaps (dense ranking semantics)
- Pre-group DenseRank (on raw rows) still works (regression check)
- Verify via QueryFrame.AsQueryFrame() path produces same results

### Step 3: Update AnalyzeRegionalPartitioning sample

File: `samples/Nivara.Samples/Incident/Analysis.cs`

Replace the manual LINQ ranking block with:
```csharp
var result = grouped
    .DenseRank("ErrorRank", [new SortKey("ErrorRate", SortDirection.Descending)])
    .Collect();
```

Remove the 8-line manual sort/rank loop and the manual column assembly.

### Step 4: Update existing test that verifies the sample output

File: `tests/Nivara.Tests/Incident/AnalysisTests.cs`

`RegionalPartitioning_D_HasErrorRateRankAndPercentRank` should still pass — the ErrorRank values must be identical.

## Blast radius

| Change | Files affected | Downstream risk |
|--------|---------------|-----------------|
| Step 1: NivaraQuery<T> methods | `src/Nivara/Linq/NivaraQuery.cs` | Additive only — no existing behavior changes |
| Step 2: Tests | `tests/Nivara.Tests/Query/PostGroupRankTests.cs` | None — new file |
| Step 3: Sample simplification | `samples/Nivara.Samples/Incident/Analysis.cs` | Output must be identical |
| Step 4: Test verification | `tests/Nivara.Tests/Incident/AnalysisTests.cs` | Asserts identical results |

Existing window function tests in `tests/Nivara.Tests/Query/RankOperationTests.cs` and `tests/Nivara.Tests/Query/WindowOperationTests.cs` are unaffected.

## Verification

1. `dotnet build Nivara.slnx` — must compile cleanly
2. `dotnet test` — all existing tests pass, new tests pass
3. The `RegionalPartitioning_D_HasErrorRateRankAndPercentRank` test validates that the simplified sample produces the same ErrorRank values

## Commit plan

1. `docs: plan post-aggregation ranking in TODO.md`
2. `Add window function methods to NivaraQuery<T> for post-aggregation ranking`
3. `Add post-group ranking tests for NivaraQuery<T>`
4. `Simplify AnalyzeRegionalPartitioning to use DenseRank from LINQ DSL`

## GitHub issues log

- [ ] #306 — Post-aggregation ranking not expressible in QueryFrame DSL (this is the issue being resolved)
