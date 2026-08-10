# Plan: Resolve issue #175 — one primary query paradigm (typed NivaraQuery<T>)

Branch: `khurram/175` (off `main`). Issue: https://github.com/khurram-uworx/Nivara/issues/175

## Problem

Three parallel, overlapping query paradigms with no declared primary API:

1. `QueryFrame` fluent API (string + `ColumnExpression`) — `src/Nivara/Query/QueryFrame.cs`
2. `NivaraLinqExtensions` — `Func<RowExpressionBuilder, ColumnExpression>` DSL sugar over `QueryFrame`
   — `src/Nivara/Linq/NivaraLinqExtensions.cs`, `RowExpressionBuilder.cs`
3. `NivaraQuery<T>` typed expression-tree LINQ — `src/Nivara/Linq/NivaraQuery.cs`,
   `TypedLinqExtensions.cs`

Each has its own laziness, discoverability, and error model, tripling the maintenance tax on any
query-semantics change.

## Decision (confirmed with human)

- **Keep** the typed `NivaraQuery<T>` layer as the single public query API.
- **Delete** `NivaraLinqExtensions` + `RowExpressionBuilder` outright (pure delegation sugar).
- **Internalize** the `QueryFrame` engine (`Query/`, `Expressions/`, `Operations/`) so it is internal
  plumbing only. `InternalsVisibleTo(Nivara.Tests / Nivara.Extensions)` already exists, so the
  engine test suite stays compilable.
- **Port** four `QueryFrame`-only features into `NivaraQuery<T>`:
  - `Distinct()` / `DistinctBy(keySelector)`
  - `SelectRows(int[])`
  - typed multi-key sort (per-key `SortDirection` / `NullOrdering` overloads on `OrderBy`/`ThenBy`)
  - typed lazy file-source entries (`Json.ScanAsQuery<T>` / `ScanJsonAsQuery<T>`,
    `Csv.ScanAsQuery<T>` / `ScanCsvAsQuery<T>`)
- Window ops and joins live on `NivaraFrame` directly (`WindowFrameExtensions.cs`,
  `NivaraFrameExtensions.cs`) and are **unaffected**.
- `SortKey`, `SortDirection`, `NullOrdering` **stay public** (used by the public `NivaraFrame`
  `Rank`/`DenseRank`/`PercentRank` window API).

## Blast radius

- **Core `src/Nivara`**: `Query/`, `Expressions/`, `Operations/`, `Linq/`, `IO/JsonExtensions.cs`,
  `NivaraFrame.cs` (`AsQueryFrame()` → internal).
- **`src/Nivara.Extensions`**: `IO/CsvExtensions.cs` — `Scan`/`ScanCsv`/`ScanAsQueryFrame`/
  `ScanCsvAsQueryFrame` removed (return `IQuerySource`/`QueryFrame`); replaced by typed variants.
- **Tests (`tests/Nivara.Tests`)**: friend assembly via `InternalsVisibleTo`, so engine tests keep
  compiling. Files exercising the removed sugar must be rewritten to the typed layer:
  `LinqQueryTests.cs`, `Execution/ParallelExecutionStrategyTests.cs`, `MLNet/MLNetIntegrationTests.cs`.
- **Docs**: `docs/LINQ.md` (heavy rewrite), `README.md` (quick-start lines 57-71), `docs/REVIEW.md`
  (finding #4 → resolved), `AGENTS.md` (implementation map).
- **Package version**: bump `PackageVersion` to 2.0.0 (breaking public surface).

## Feature design notes

### Typed multi-key sort
`NivaraQuery<T>` gains optional ordering params on existing methods (keeps type-safety; column-name
`SortKey[]` is not exposed in the typed layer):
```csharp
public NivaraQuery<T> OrderBy(Expression<Func<T, object?>> keySelector,
    SortDirection direction = SortDirection.Ascending,
    NullOrdering nullOrdering = NullOrdering.NullsLast)
public NivaraQuery<T> ThenBy(Expression<Func<T, object?>> keySelector,
    SortDirection direction = SortDirection.Ascending,
    NullOrdering nullOrdering = NullOrdering.NullsLast)
```
Backing `QueryFrame.Sort` / `ThenBy` already accept `SortKey` (name-based) and expression keys via
`SortByExpression(ColumnExpression, SortDirection, NullOrdering)`. Implementation: translate the
typed key to a `ColumnExpression`, then call `SortByExpression`/`ThenBy` with direction + nullOrdering.

### Distinct
```csharp
public NivaraQuery<T> Distinct()                       // dedup all columns
public NivaraQuery<T> DistinctBy(Expression<Func<T, object?>> keySelector)  // dedup on key columns
```
Backing: `QueryFrame.Distinct()` / `QueryFrame.Distinct(params string[])` — translate the typed key
to a column-name reference.

### SelectRows
```csharp
public NivaraQuery<T> SelectRows(params int[] indices)
```
Backing: `QueryFrame.SelectRows(params int[])`.

### Typed lazy file-source entries
Internal factory on `NivaraTypedLinqExtensions` (friend-accessible):
```csharp
internal static NivaraQuery<T> FromFrame<T>(QueryFrame frame) where T : class, new()
    // validates row type against frame.Schema, returns new NivaraQuery<T>(frame)
```
Public entries (core + Extensions):
```csharp
// Nivara.IO.Json (core)
public static NivaraQuery<T> ScanAsQuery<T>(string filePath, JsonOptions? options = null) where T : class, new()
public static NivaraQuery<T> ScanJsonAsQuery<T>(string filePath, JsonOptions? options = null) where T : class, new()
// Nivara.Extensions.IO.Csv
public static NivaraQuery<T> ScanAsQuery<T>(string filePath, CsvOptions? options = null) where T : class, new()
public static NivaraQuery<T> ScanCsvAsQuery<T>(string filePath, CsvOptions? options = null) where T : class, new()
```
Removed public methods (now impossible — return internal types):
`Json.Scan`, `Json.ScanJson`, `Json.ScanAsQueryFrame`, `Json.ScanJsonAsQueryFrame`;
`Csv.Scan`, `Csv.ScanCsv`, `Csv.ScanAsQueryFrame`, `Csv.ScanCsvAsQueryFrame`.

### Internalized types (Phase 1)
- `Query/`: `QueryFrame`, `IQueryOperation`(+generic), `IQuerySource`, `OperationType`,
  `QueryDiagnostics`, `QueryDiagnosticMode`, `QueryExecutor` (already internal), `QueryNode` family +
  visitors, `QueryPlan`, `QueryPlanAnalyzer`, `QueryOptimizer`, plan visitors/interfaces.
- `Expressions/`: `ColumnExpression` + node types + enums, `ColumnExpressions`,
  `ExpressionTypeInferer`, `FusedExpressionEvaluator`, `FusedNodeTreeRunner`, `FusedKernel`.
- `Operations/`: all operation classes, `GroupKey`, `GroupedData`, `CompositeKey`, `JoinOperation` +
  enums, `AggregationFunction` family, `ProjectionOperation`, `SelectRowsOperation`,
  `SortByExpressionOperation`/`SortExpressionKey`. **Keep public:** `SortKey`, `SortDirection`,
  `NullOrdering`.
- `NivaraFrame.AsQueryFrame()` → internal; `NivaraQuery<T>.AsQueryFrame()` → internal (or removed).
- **Keep public:** `NivaraQuery<T>`, `NivaraGroupedQuery<TKey,T>`, `Grouping<TKey,T>`,
  `NivaraTypedLinqExtensions`.

## Planned commits

1. `docs: plan issue #175 in TODO.md` — this document.
2. `refactor: internalize query engine types (Query/Expressions/Operations)` — Phase 1.
3. `refactor: remove NivaraLinqExtensions and RowExpressionBuilder DSL` — Phase 2 + test rewrites.
4. `feat: add Distinct, SelectRows, typed multi-key sort to NivaraQuery<T>` — Phase 3a.
5. `feat: add typed lazy file-source queries (Json/Csv); remove QueryFrame-returning IO methods` — Phase 3b.
6. `test: cover new typed query features and Collect/ToObjects parity` — Phase 4.
7. `docs: make typed NivaraQuery<T> the single query API; bump to 2.0.0` — Phase 5 (docs + version + REVIEW.md resolved note + close #175).

## Verification

- `dotnet build Nivara.slnx` after each phase (fast, no test run).
- `dotnet test` — **ask the human first**; run once at the end of Phase 4 (full suite) and once
  after Phase 5 if docs/version touched anything compile-relevant.
- Confirm no public API outside the typed layer returns internal types (build catches CS0050/CS0051).
- Manual spot-check: README quick-start compiles (typed-only).

## GitHub issues log

- [ ] No new issues yet. If deferred work surfaces (e.g., custom `IQuerySource` public entry,
  `ToRowList` loss, Diagnostics surface removal), create an issue immediately via
  `gh issue create --repo khurram-uworx/Nivara` and record it here.
- [ ] #175 — the tracked issue this plan resolves (close at end of Phase 5).

---

Reminder during execution: as each task executes, if you find deferred work or a concern outside the
current plan, create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and
record its number in the GitHub issues log above — don't rely on memory or wait until the end.
