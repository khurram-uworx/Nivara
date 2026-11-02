# Plan: Issue #843 — wire ArrowValidityMask into ArrowInterop

## Problem

Arrow validity conversion is per-element today:
- Export walks every value through Apache.Arrow builders (`ArrowInterop.cs` `Create*Array`), each maintaining its bitmap internally. Only the Guid path writes a bitmap by hand.
- Import walks every slot, calls `chunk.IsNull(i)`, and materialises a `List<object?>` of boxed values (`ConvertArrowColumnToNivaraColumnTyped` / `ConvertArrowColumnToNivaraColumnInt`).

The span-first `ArrowValidityMask` helper landed in #833 handles packing/unpacking, polarity inversion, padding, and absent-mask cases, but nothing calls it. Phase C of `docs/plan/ARROW-ROADMAP.md` keeps its "*Open: buffer-based*" criterion pending on this work.

## Proposed changes

### Export (buffer-based validity)

For each flat primitive `Create*Array` method, build an `ArrayData` directly:
- typed data buffer (element-wise conversion only where the Arrow representation differs from CLR, e.g. DateTime -> microseconds, Half -> ushort bits, Guid -> 16 bytes)
- null slots filled with `default`
- `nullCount = column.NullCount`
- validity buffer from `ArrowValidityMask.ToValidityBitmap(column null mask)` (empty when no nulls, matching builder elision)
- `buffers = [validity, data]` (variable-binary strings keep `[validity, offsets, data]` via the builder path)

Guid path: replace its hand-written bit loop with the helper call.

Strings/char keep the existing builder path (no behaviour change).

### Import (buffer-based validity)

In `ConvertArrowColumnToNivaraColumnTyped<T>` and `ConvertArrowColumnToNivaraColumnInt`:
- pre-allocate `bool[] nullMask[totalLength]` and `T[] values[totalLength]`
- per chunk, when `chunk.Data.Offset == 0`, derive the mask with `ArrowValidityMask.ToNullMask(validityBuffer, length)`; for non-zero offsets, fall back to `chunk.IsNull(i)` for that chunk
- fill `values` only for non-null slots via typed Arrow accessors (no `List<object?>` boxing)
- construct the column as today (CreateFromNullable for value types, CreateForReferenceType for reference types)

Chunked columns: every chunk's mask is decoded and concatenated in order.

### Docs

Update `docs/plan/ARROW-ROADMAP.md` Phase C criterion from `*Open:*` to `*Landed (issue #843):*`.

## Blast radius

- `src/Nivara.Extensions/IO/ArrowInterop.cs` (all export/import paths; Guid path)
- `docs/plan/ARROW-ROADMAP.md`
- Tests: `tests/Nivara.Tests/IO/ArrowInteropTests.cs`, `ArrowValidityMaskTests.cs`, `ArrowExtendedDomainRoundTripTests.cs`, `ArrowInteropPerfTests.cs`
- Downstream callers: Parquet interop, ML.NET conversion, IO tests. All go through `ArrowInterop`, so behaviour must remain identical.

## Verification steps

1. `dotnet build Nivara.slnx` (Release)
2. Targeted: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ArrowInterop|FullyQualifiedName~ArrowValidityMask|FullyQualifiedName~ArrowExtendedDomain"`
3. Docs snippet gate if docs change: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
4. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits

1. `docs: plan issue #843 in TODO.md`
2. `feat(io): buffer-based validity on Arrow export for flat primitives`
3. `feat(io): buffer-based validity on Arrow import`
4. `docs: mark Phase C buffer-based criterion landed`
5. `test(io): parity tests for buffer-based validity paths`
6. `docs: remove TODO.md — plan executed` (at G2)

## GitHub issues log

- (none yet; add here if deferred work is found during execution)
