# TODO — Issue #840: Arrow boundary buffer-level bool[] ↔ bit-packed validity conversion

## Problem

Issue #840 asks for the Arrow interop boundary to use a real buffer-level
`bool[]` ↔ bit-packed validity conversion instead of element-wise packing.
ADR-007 ratifies byte-wise masks in core and bit-packed validity only at the
Arrow boundary. `ArrowValidityMask` (helper) and most export paths are already
wired; remaining work: no-allocation for no-nulls columns, offset-aware import
for sliced arrays, variable-binary (`string`/`char`) limitation stated in code,
series import parity, and doc updates.

## Proposed changes

### 1. `src/Nivara.Extensions/IO/ArrowValidityMask.cs`
- `ToValidityBitmap(ReadOnlySpan<bool>)` empty case returns `Array.Empty<byte>()`
  (cached, no allocation) instead of `[]`.
- Add offset-aware `ToNullMask(ReadOnlySpan<byte> validityBitmap, int offset, Span<bool> destination)`
  plus `int length` factory overload, with buffer-length validation against
  both `offset + length` and `length` (fallback for trimmed buffers).
- Keep all existing signatures unchanged.

### 2. `src/Nivara.Extensions/IO/ArrowInterop.cs` (import)
- `ConvertArrowColumnToNivaraColumnTyped<T>` and `ConvertArrowColumnToNivaraColumnInt`:
  use offset-aware `ToNullMask` for the validity buffer when present, removing
  `chunk.IsNull(i)` from the null-detection path; keep per-slot value extraction.
- Add code comment on any remaining per-element branch stating the limitation.
- `FromArrowArray<T>`: preallocate `T[]`/`bool[]`, use `ArrowValidityMask.ToNullMask`
  with `arrowArray.Data.Offset`, extract values only for non-null slots.

### 3. `src/Nivara.Extensions/IO/ArrowInterop.cs` (export)
- Add doc comment on both `CreateStringArray` overloads stating the variable-binary
  layout requires a per-value pass; builder path retained; only remaining
  element-wise export path.

### 4. Docs
- `docs/adr/007-byte-wise-null-mask-fast-path.md` point 6: update to reflect wired state.
- `docs/plan/ARROW-ROADMAP.md` Phase C: mark wire-in as landed for #840 and state limitations.

## Verification steps
- Build: `dotnet build Nivara.slnx -c Release`
- Targeted tests: `ArrowValidityMaskTests`, `ArrowInteropBufferValidityTests`,
  `ArrowInteropTests`, `ArrowValidityInteropTests`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.

## Planned commits
1. docs: plan issue #840 in TODO.md
2. feat(io): make ArrowValidityMask.ToValidityBitmap empty case non-allocating
3. feat(io): add offset-aware ToNullMask overload
4. feat(io): buffer-based validity import for sliced arrays and series
5. docs(io): state remaining element-wise limitations in ArrowInterop
6. docs(adr): update ADR-007/roadmap for wired ArrowValidityMask
7. test(io): add byte-boundary and no-allocation coverage (if needed)
8. docs: remove TODO.md — plan executed

## Blast radius
- `ArrowValidityMask` (internal): signature additions only; existing callers/tests unchanged.
- `ArrowInterop`: import paths for all flat primitive types and series; behavior parity expected.
- Docs: ADR-007, ARROW-ROADMAP only.
- Tests: existing interop tests must pass unchanged.

## GitHub issues log
- (none yet)
