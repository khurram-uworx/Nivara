# Plan: Issue #264 — Public `AsStream` API + explicit non-streamable boundary + budget↔chunk-size docs

**Branch:** `khurram/264` · **Repo:** khurram-uworx/Nivara · **Date:** 2026-08-16

## Problem

`QueryFrame.AsStream(chunkSize:, ct:)` (src/Nivara/Query/QueryFrame.cs:433) is
`internal` even though docs/PHASE4.md:114 lists it as public. For plans containing
non-streamable operations (Sort, SortByExpression, GroupBy, Join, Distinct, Rolling,
Cumulative, Shift, Rank) or window expressions, `StreamingExecutionStrategy.StreamChunksAsync`
silently falls back to a single frame produced from the full source — undocumented
behavior. The memory-budget → chunk-size derivation (explicit `ChunkSize` wins; otherwise
`clamp(budget/10 ÷ 100 bytes/row, 1000, 100000)`) is also undocumented at the API level.

User scope decision: make `AsStream` externally reachable too — public `ScanAsQueryFrame`
factories, public `AsQueryFrame()` on `NivaraFrame`/`NivaraQuery<T>`, and an `AsStream`
passthrough on `NivaraQuery<T>`.

## Blast radius

- **Types promoted internal → public:** `QueryFrame` (QueryFrame.cs:14), `ColumnExpression`
  (ColumnExpression.cs:38), `ColumnExpressions` (ColumnExpression.cs:765). Required for C#:
  a `public` factory cannot return internal `QueryFrame`; a `public QueryFrame` cannot expose
  `Filter/Select/GroupBy/SortByExpression` whose signatures use internal `ColumnExpression`;
  users need the `ColumnExpressions` factory to build expressions.
  - `ColumnExpression` subclasses stay internal; all its members already use public types
    (Type, Schema, object, operators returning ColumnExpression). No further cascade expected.
- **New public methods:** `QueryFrame.AsStream`, `NivaraFrame.AsQueryFrame`,
  `NivaraQuery<T>.AsQueryFrame`, `NivaraQuery<T>.AsStream`,
  `Csv.ScanAsQueryFrame`, `Json.ScanAsQueryFrame`, `NivaraParquetReader.ScanAsQueryFrame`.
- **Downstream callers:** `Nivara.Extensions` (CSV/Parquet factories, Streamix bridge per
  PHASE45.md relies on InternalsVisibleTo to QueryFrame — unaffected), `Nivara.Tests`
  (InternalsVisibleTo). Existing tests call `AsStream` via internal access — unaffected by
  promotion.
- **No behavior change** to `StreamChunksAsync`/`ExecuteCore*` — only documentation of the
  existing fallback contract.

## Proposed changes

### 1. Core public API (src/Nivara)
- `QueryFrame` → `public sealed`; constructors stay internal.
- `QueryFrame.AsStream` → `public`; rewrite XML doc to pin the contract:
  - Streamable plan + chunk-capable source → one `NivaraFrame` per source chunk.
  - Boundary op (Sort/SortByExpression/GroupBy/Join/Distinct/Rolling/Cumulative/Shift/Rank)
    or window expression → single merged fallback frame, rows identical to `CollectAsync()`.
  - Source without `CanReadInChunks` → same single-frame fallback.
  - `chunkSize`: honored by CSV/JSON; advisory for Parquet (row-group aligned); default
    10,000; strategy-level derivation from `MemoryBudget` = `clamp(budget/10 ÷ 100, 1000, 100000)`.
- `ColumnExpression` → `public abstract`; `ColumnExpressions` → `public static`.
- `NivaraFrame.AsQueryFrame()` → `public`.
- `NivaraQuery<T>.AsQueryFrame()` → `public`; add `AsStream(int chunkSize = 10000, CancellationToken ct = default)` passthrough.

### 2. Public `ScanAsQueryFrame` factories
- `Csv` (Nivara.Extensions/IO/CsvExtensions.cs), `Json` (src/Nivara/IO/JsonExtensions.cs),
  `NivaraParquetReader` (Nivara.Extensions/IO/NivaraParquetReader.cs): each gains
  `public static QueryFrame ScanAsQueryFrame(string, options)` delegating to existing internal `ScanFrame`.

### 3. Boundary documentation (StreamingExecutionStrategy.cs:324-329)
- Expand `StreamChunksAsync` doc to name the boundary-op set + window gate and pin the
  fallback as one frame equal to `CollectAsync()`.

### 4. Tests (tests/Nivara.Tests/Query/AsyncStreamingTests.cs)
- `AsStream_NonStreamablePlan_ReturnsSingleMergedFrame` (CSV + Sort → 1 chunk, equals Collect).
- `AsStream_NonChunkCapableSource_ReturnsSingleFrame` (in-memory frame → 1 chunk, equals Collect).
- `ScanAsQueryFrame_PublicEntryPoint_StreamsChunks`.
- `NivaraQuery_T_AsStream_Passthrough`.
- `NivaraFrame_AsQueryFrame_Public_Streams`.

### 5. Docs
- `docs/PHASE4.md`: Resolved Decisions entries for boundary fallback + budget→chunk-size formula.
- New `docs/STREAMING.md`: concise behavior reference; link from `AsStream` XML doc.
- `CHANGELOG.md` entry.

## Verification

- `dotnet build Nivara.slnx` after each commit.
- `dotnet test` (targeted: AsyncStreamingTests, StreamingExecutionStrategyTests) — ask before running.

## Planned commits

1. `docs: plan issue #264 in TODO.md`
2. `feat: promote QueryFrame/ColumnExpression/ColumnExpressions to public API (issue #264)`
3. `feat: public AsStream on QueryFrame/NivaraQuery<T>, public AsQueryFrame (issue #264)`
4. `feat: public ScanAsQueryFrame factories for Csv/Json/Parquet (issue #264)`
5. `docs: streaming behavior reference + PHASE4 resolved decisions (issue #264)`
6. `test: AsStream boundary fallback and public entry points (issue #264)`
7. `docs: changelog for issue #264; remove TODO.md`

## GitHub issues log

- [ ] #275 — Public QueryFrame: ToQueryPlan/QueryPlan/ExecutionEngine remain internal (created while making QueryFrame public for #264)
