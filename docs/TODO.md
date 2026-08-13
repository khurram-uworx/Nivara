# Issue #190 — Interop coverage for extended CLR types (Parquet/Arrow/ML)

## Problem

The extended CLR domain from #158 (`Half`, `nint`/`nuint`, `Int128`/`UInt128`,
`sbyte`/`ushort`/`uint`/`char`, `DateOnly`/`TimeOnly`, `DateTimeOffset`, `Guid`,
`TimeSpan`) is not covered by the interop layers. Extended types currently throw
`UnsupportedTypeException` from `TypeMapper.CreateParquetField` (TypeMapper.cs:135),
from Arrow dispatch switches, or are silently coerced to `float` in the ML.NET path
(`ConvertToFloat` returns `0f` for unsupported types).

Acceptance criterion: each interop layer round-trips the extended-domain types it
can represent and throws a clear, documented error otherwise. CSV/JSON intentionally
excluded (extended types unreachable there).

## Decisions (confirmed with human)

1. **Metadata round-trip** — widened Parquet/Arrow types (Half→float, nint→long,
   nuint→ulong, char→ushort, DateTimeOffset→DateTime, TimeSpan→long) store the
   original CLR type in Parquet `CustomMetadata` / Arrow schema metadata under key
   `nivara.clrType.<column>`; the reader restores the typed column. Foreign files
   without metadata read back as the base/widened type.
2. **ML.NET bounded scope** — faithful `ToNivaraFrame` (full primitive DataViewType
   coverage); `ToDataView` keeps its float/feature-vector contract but throws clear
   errors instead of silent `0f`. No custom IDataView implementation.
3. **Arrow Half native** — `HalfFloatType` via manual `ArrayData` (no `HalfFloatArray`
   class in Apache.Arrow 23.0.0).

## Capability matrix (verified by reflection on 6.1.0 / 23.0.0 / 5.0.0)

| CLR type | Parquet | Arrow | ML.NET |
|---|---|---|---|
| `Half` | widen→`float` + meta | native `HalfFloatType` | widen→`float` (features) |
| `nint` / `nuint` | widen→`long`/`ulong` + meta | widen→`Int64`/`UInt64` + meta | throw |
| `char` | widen→`ushort` + meta | widen→`StringType` + meta | throw |
| `DateTimeOffset` | widen→`DateTime`(UTC) + meta | `TimestampType` + meta | native |
| `TimeSpan` | widen→`long`(ticks) + meta | native `DurationType` | native |
| `DateOnly` | native `DataField<DateOnly>` | native `Date32Type` | throw |
| `TimeOnly` | native `DataField<TimeOnly>` (TIME nanos) | native `Time64Type` (nanos) | throw |
| `Guid` | native `DataField<Guid>` | native `FixedSizeBinaryType(16)` | throw |
| `Int128`/`UInt128` | **throw** (documented) | **throw** | **throw** |

## Blast radius

| File | What changes | Downstream | Tests covering |
|---|---|---|---|
| `src/Nivara.Extensions/IO/TypeMapper.cs` | new map entries, Parquet field arms, widening table, metadata keys, `IsMLNetSupported`, suggestions | `ArrowInterop`, `ParquetWriter`, `ParquetReader`, `MLNetInterop`, `NivaraFrameIOExtensions` | `TypeMapperTests.cs` |
| `src/Nivara.Extensions/IO/ParquetWriter.cs` | field/array arms, `CustomMetadata` write | public `WriteParquet*` APIs | `ParquetWriterTests.cs`, `ArrowParquetIntegrationTests.cs` |
| `src/Nivara.Extensions/IO/ParquetReader.cs` | metadata read, field/column arms, type restore, conversions | public `ReadParquet*` APIs | `ParquetReaderTests.cs`, `ArrowParquetIntegrationTests.cs` |
| `src/Nivara.Extensions/IO/ArrowInterop.cs` | per-type array creators, Half `ArrayData`, metadata, read-back arms | `NivaraFrameIOExtensions` (ToArrowTable/FromArrowTable) | `ArrowInteropTests.cs`, `ArrowInteropPerfTests.cs` |
| `src/Nivara.Extensions/MLNet/MLNetInterop.cs` | typed DataView getters, `ConvertToFloat` throws | `MLNetExtensions` (LoadFromNivaraFrame, ToNivaraFrame, Transform, Fit, Predict) | `MLNet/MLNetIntegrationTests.cs` |
| `docs/CHANGELOG.md`, `AGENTS.md` | interop coverage notes | — | — |

No changes to core `src/Nivara` (AutoDiff/storage untouched). Public API shape unchanged
— only type-coverage widening and error behavior.

## Implementation steps (one commit per step)

1. **TypeMapper** — extend Arrow maps (Half, DateOnly, TimeOnly, TimeSpan, Guid,
   DateTimeOffset, nint, nuint, char); native `CreateParquetField` arms (DateOnly,
   TimeOnly, Guid); widening field table; `nivara.clrType` metadata key helpers;
   `IsMLNetSupported`; `GetTypeSuggestions` updates. Commit: `feat(io): extend TypeMapper interop coverage for extended CLR domain`.
2. **Parquet writer** — field/array arms + `CustomMetadata` write for widened types.
   Commit: `feat(io): write extended CLR types to Parquet with clrType metadata`.
3. **Parquet reader** — metadata read, type restore, conversion helpers.
   Commit: `feat(io): read extended CLR types back from Parquet`.
4. **Parquet round-trip tests** — full-type round-trip + throw tests.
   Commit: `test(io): Parquet round-trip for extended CLR types`.
5. **Arrow interop** — writer/reader arms, Half `ArrayData`, schema metadata.
   Commit: `feat(io): Arrow interop for extended CLR types`.
6. **Arrow tests** — round-trip + throw tests.
   Commit: `test(io): Arrow round-trip for extended CLR types`.
7. **ML.NET** — typed DataView getters, `ConvertToFloat` throws.
   Commit: `feat(mlnet): faithful ToNivaraFrame and clear unsupported-type errors`.
8. **ML.NET tests** — typed columns + throw tests.
   Commit: `test(mlnet): extended DataViewType coverage`.
9. **Docs** — CHANGELOG + AGENTS.md follow-up note.
   Commit: `docs: document interop coverage for extended CLR types`.

## Verification

- `dotnet build Nivara.slnx` after each step.
- Ask human before running `dotnet test`. Filters: `Nivara.Tests.IO`, `Nivara.Tests.MLNet`.
- Confirm 1948-test baseline still green on full suite (human approval required).

## Known risks (verify at implementation, not blockers)

- Parquet.Net read-back shapes for `DataField<Guid>` / `DataField<TimeOnly>`
  (ClrType=Int64) — confirm via first round-trip test.
- Arrow `Date32Array.Builder` / `Time64Array.Builder` / `DurationArray.Builder` append
  semantics (unit offsets) — confirm via first round-trip test.
- `DataField<TimeOnly>` default precision — choose `TimeUnitPrecision.Nanos` for 100ns
  (tick) fidelity.

## GitHub issues log

- [ ] (none yet — create `gh issue create --repo khurram-uworx/Nivara` at discovery time for deferred work)
