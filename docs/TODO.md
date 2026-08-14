# Plan: Issues #232 (IO/Extensions) and #234 (Schema)

Branch: `khurram/issues` (off `main` @ 29be0f5)

## Problem

Two open findings from `docs/REVIEW.md` were never filed/fixed and now have tracked issues:

- **#232 IO/Extensions** — duplicated `Read*`/`Scan*` entry points per format, mutable
  `JsonOptions.Default`/`CsvOptions.Default` singletons, `bool CsvOptions.TrimOptions`,
  and magic-string `ParquetWriteOptions.Compression`.
- **#234 Schema** — `ColumnMetadata.With()` cannot clear values (coalesces `??`),
  `Schema.Equals`/`GetHashCode` ignore `ColumnMetadata`, and `IsCompatibleWith` is
  name+type only.

## Blast radius

### #232

- `src/Nivara/IO/JsonExtensions.cs` — `Json` static class (10 methods → 3).
- `src/Nivara.Extensions/IO/CsvExtensions.cs` — `Csv` static class (10 methods → 3).
- `src/Nivara/IO/JsonDataSource.cs` — `JsonOptions` (immutability).
- `src/Nivara.Extensions/IO/CsvDataSource.cs` — `CsvOptions` (immutability + enum).
- `src/Nivara.Extensions/IO/ParquetWriteOptions.cs` — `ParquetWriteOptions` (immutability + enum).
- `src/Nivara.Extensions/IO/ParquetWriter.cs` — consume `Compression`, `RowGroupSize`, `WriteMetadata`.
- Downstream callers: **tests only** — `LazyDataSourceTests.cs`, `ExecutionIntegrationTests.cs`,
  `ResourceManagementPropertyTests.cs`, `NivaraQueryFeatureTests.cs`, `QueryOptimizerTests.cs`,
  `QueryOptimizationPropertyTests.cs`, `QueryExecutionTests.cs`, `QueryExecutionPropertyTests.cs`,
  `ParquetWriterTests.cs`, `NivaraFrameExtensionsTests.cs`, `ArrowParquetIntegrationTests.cs`.
- Docs: `README.md`, `GETTING-STARTED.md`, `docs/LINQ.md`. Samples do **not** use these APIs.

### #234

- `src/Nivara/Schema.cs` — `Schema` + `ColumnMetadata` equality/clear semantics.
- Downstream callers of `Schema.Equals`: `QueryExecutor.cs:141` (`ValidateQueryPlan`),
  `QueryPlan.cs:198` (diagnostics) — both compare schemas produced by identical
  `TransformSchema` chains, so metadata-aware equality must remain consistent; verify via
  query test suites.
- Downstream callers of `IsCompatibleWith`: `NivaraFrame.cs:1197` (frame ops,
  `requireExactMatch: false`) and `ConcatenationOperation.cs:136` (vertical concat,
  `requireExactMatch: true`). Adding `requireMetadataMatch` defaulting to `false` keeps
  both unaffected.
- Tests: `SchemaTests.cs` (update `ParquetWriteOptions`/`ColumnMetadata` usage), plus new tests.

## Changes (decided with human)

1. **Canonical methods** — `Json`/`Csv` each expose exactly:
   - `ReadFrame(path, options)` → eager `NivaraFrame`
   - `ScanFrame(path, options)` → lazy `QueryFrame`
   - `ScanQuery<T>(path, options) where T : class, new()` → lazy `NivaraQuery<T>`
   - Delete all other `Read*`/`Scan*` variants (public and internal). Dictionary-returning
     methods have zero repo callers.
2. **Immutable options** — get-only properties + `With(...)` builders; `Default` is a
   frozen instance. `JsonOptions.With` clones `SerializerOptions`
   (`new JsonSerializerOptions(x)`) to prevent aliasing.
3. **Enums** — `CsvTrimOptions { None, Trim }` (mapped in `ToCsvConfiguration()`);
   `ParquetCompression { None, Snappy, Gzip, Lzo, Brotli, LZ4, Zstd, Lz4Raw }` mirroring
   `Parquet.CompressionMethod`.
4. **ParquetWriter wiring (all three options)** —
   - `Compression` → `ParquetOptions.CompressionMethod` passed to `ParquetWriter.CreateAsync`.
   - `RowGroupSize` → chunk `ConvertNivaraFrameToParquet` into multiple row groups
     (extract column arrays once; slice per row group through the typed write path).
   - `WriteMetadata` → gate `parquetWriter.CustomMetadata` (`nivara.clrType.*`) attachment.
5. **Schema** —
   - `ColumnMetadata`: add `ClearDefaultValue()`, `ClearDescription()`, `ClearProperties()`;
     add `Equals`/`GetHashCode` (IsNullable, DefaultValue, Description, Properties).
   - `Schema`: `Equals`/`GetHashCode` include per-column metadata; implement
     `IEquatable<Schema>`.
   - `IsCompatibleWith`: add optional `bool requireMetadataMatch = false`.
6. **Docs** — `README.md`, `GETTING-STARTED.md`, `docs/LINQ.md`; CHANGELOG under `[Unreleased]`.
7. **Tests** — migrate existing call sites; add immutability, enum-mapping, parquet
   round-trip (incl. multi-row-group + WriteMetadata off), schema-equality-with-metadata,
   and clear-path tests.

## Planned commits (one logical unit each)

1. `docs: plan issues #232/#234 in TODO.md`
2. `refactor: canonical Json/Csv ReadFrame/ScanFrame/ScanQuery entry points (#232)` + test migration
3. `refactor: immutable JsonOptions/CsvOptions/ParquetWriteOptions with With() builders (#232)`
4. `feat: CsvTrimOptions and ParquetCompression enums replace bool/string options (#232)`
5. `feat: honor Compression/RowGroupSize/WriteMetadata in ParquetWriter (#232)`
6. `test: option immutability and parquet option wiring coverage (#232)`
7. `feat: ColumnMetadata clear path and metadata-aware Schema equality (#234)`
8. `test: Schema/ColumnMetadata metadata equality and clear path (#234)`
9. `docs: update README/GETTING-STARTED/LINQ and changelog for #232/#234`
10. `docs: remove TODO.md - plan executed`

## Verification

- `dotnet build Nivara.slnx` after each step.
- Run affected suites (ask human before `dotnet test`): `SchemaTests`, IO tests
  (`LazyDataSourceTests`, `ParquetWriterTests`, `ArrowParquetIntegrationTests`), and query
  suites (`QueryExecutionTests`, `QueryOptimizerTests`, `QueryOptimizationPropertyTests`,
  `QueryExecutionPropertyTests`, `NivaraQueryFeatureTests`).

## GitHub issues log

- [ ] (none yet) — create issues at discovery time via `gh issue create --repo khurram-uworx/Nivara` while executing, and record numbers here. Don't rely on memory.

## Reminder

As each task executes, if you find deferred work or a concern outside the current plan,
create a GitHub issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and
record its number in the GitHub issues log above. Don't wait until the end of the plan.
