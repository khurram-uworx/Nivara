# Plan: Issue #1075 — pin reference-type per-chunk Arrow round-trip (string)

## Problem

`ArrowInterop.ConvertArrowColumnToNivaraColumnTyped<T>` has two import routes whose
null semantics should be observably identical:

- **Flat (default):** for reference types it zeroes null slots and calls
  `NivaraColumn<T>.CreateForReferenceType`, detecting nulls from the values.
- **Per-chunk (`PreserveChunkStructure`):** builds `chunkValues` (null slots left as
  `default(T)`) and attaches an explicit null mask only when a chunk has a null,
  then wraps one `ColumnStorage<T>` per Arrow array.

The per-chunk route is only pinned with `int`. For `string`, the two routes derive
`HasNulls`/`NullCount` differently (explicit mask vs detected nulls), and nothing
pins agreement. Issue proposes a test in
`tests/Nivara.Tests/IO/ChunkedArrowRoundTripTests.cs`.

## Proposed changes

Test-only addition to `ChunkedArrowRoundTripTests.cs`:

- Add a nullable string field (`StringType.Default, nullable: true`) and a
  `TwoChunkStringTable()` helper (2 `RecordBatch`es, one null in each chunk).
- Add `FromArrowTable_PreserveOption_String_KeepsChunksAndNulls` asserting
  `Storage` is `ChunkedColumn<string>`, chunk count 2, `HasNulls`, `NullCount == 2`,
  `IsNull(i)` for positions 2 and 5, and non-null values.
- Add `ArrowRoundTrip_PreserveOption_String_KeepsChunksValuesAndNulls` asserting
  chunk count, values, and `IsNull` survive re-export.
- Add `FromArrowTable_DefaultAndPreserve_String_AgreeOnNullSemantics` asserting the
  default flattening import reports the same `HasNulls`/`NullCount`/values.

No production code change expected unless the test exposes a mismatch. If it does,
the likely site is the explicit-mask branch in
`ConvertArrowColumnToNivaraColumnTyped<T>` (lines ~1268-1294) or the flat route
(lines ~1349-1355).

## Verification

- `dotnet build Nivara.slnx`
- targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ChunkedArrowRoundTripTests"`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan #1075 reference-type per-chunk Arrow pin in TODO.md`
2. `test(io): pin string per-chunk Arrow round-trip (#1075)`
3. `docs: remove TODO.md — plan executed`

## Blast radius

- `tests/Nivara.Tests/IO/ChunkedArrowRoundTripTests.cs` only.
- No `docs/` count change (existing test file, no new ADR/doc).
- No production symbols changed.

## GitHub issues log

(none captured)
