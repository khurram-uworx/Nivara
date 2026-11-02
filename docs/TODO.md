# Plan: Arrow interop validity round-trip through span-first helper (#834)

## Problem

Step 2 of the Arrow validity fast-path work (depends on #832 docs + #833 helper, both landed).
The span-first `ArrowValidityMask` helper (#833) is unit-tested in isolation but has not yet been
checked against the real Arrow buffers that `ArrowInterop` produces and consumes. This issue is
intentionally validation-first: it proves the byte-wise `Span<bool>` mask and Apache Arrow's
bit-packed validity buffer translate exactly, in both directions, across empty/all-null/all-valid/
mixed shapes, while keeping the existing builders+`Append` data path.

## Blast radius

- New test file only: `tests/Nivara.Tests/IO/ArrowValidityInteropTests.cs`.
- Optional doc bullet in `docs/plan/ARROW-ROADMAP.md` (Phase C), no new `File.cs:NN` citations.
- No changes to `src/Nivara/`, `src/Nivara.Extensions/IO/ArrowInterop.cs`, or `ColumnStorage<T>`.
- Depends on `Nivara.IO.ArrowValidityMask` (internal, `InternalsVisibleTo` granted to `Nivara.Tests`).
- No performance claims; no zero-copy claims.

## Proposed changes

Add `ArrowValidityInteropTests` covering, for `int` (value type), `string` (reference type) and
`bool` (bit-packed data, validity is a separate buffer):

Export (Nivara -> Arrow): compare `IArrowArray.NullBitmapBuffer` against
`ArrowValidityMask.ToValidityBitmap(mask, length, dest)` over defined bits only, and assert
`NullCount`. Treat an elided Arrow buffer (length 0) as all-valid, matching Nivara's empty mask.

Import (Arrow -> Nivara): build arrays with builders and one hand-built `ArrayData` whose padding
bits are dirty; convert via `ArrowInterop.FromArrowArray<T>`; assert `IsNull(i)` matches
`ArrowValidityMask.ToNullMask(buffer, length)`.

Round-trip: empty / all-null / all-valid / mixed for `int`.

Table-level: multi-column frame with mixed nulls, per-column validity.

## Key conventions (must not be violated by the tests)

- Nivara stores an all-valid column as an EMPTY mask (length 0); Arrow elides the validity buffer.
  "Exact" is asserted under the helper's documented absent-mask convention, not raw buffer length.
- Padding bits (trailing bits of the final byte) are undefined by the Arrow spec; compare defined
  bits only. The hand-built test sets padding bits to 1 to prove import ignores them.
- Tests use non-sliced arrays only; offset handling is a follow-up wiring concern.
- No claim that `ArrowInterop` now uses the fast path; it still uses builders+`Append`.

## Verification steps

1. `dotnet build Nivara.slnx -c Release`
2. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ArrowValidityInteropTests"`
3. `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~ArrowInteropTests|FullyQualifiedName~ArrowValidityMaskTests"`
4. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
5. If roadmap edited: `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`

## Planned commits

1. `docs: plan Arrow interop validity round-trip in TODO.md` (this file)
2. `test(io): add Arrow validity interop round-trip tests`
3. (optional) `docs(arrow): note validity parity proven at interop boundary (#834)`

## GitHub issues log

- (none created during this work; follow-ups, if any, recorded at discovery time)
