# Plan — Ratify the byte-wise `Span<bool>` null mask (issue #832)

## Problem

Issue #832 (documentation) asks us to ratify the existing byte-wise `bool[]` null mask as a
**deliberate, documented `Span<bool>` fast path** for the Arrow lens, instead of leaving it as an
undecided or accidental representation. It is the documentation/decision half of
`docs/plan/ARROW-ROADMAP.md` Phase C under **Option A**: core storage keeps a byte-wise validity
mask; Arrow interop converts to/from Apache Arrow's bit-packed validity buffer at the boundary.

Current state, verified against the code:

- `ColumnStorage<T>` holds `readonly bool[]? nullMask` (`src/Nivara/Storage/ColumnStorage.cs:35`),
  exposed as `ReadOnlySpan<bool> NullMask` (`:148-155`, `IColumnStorage<T>` at
  `src/Nivara/Interfaces.cs:92`) and internally as `ReadOnlyMemory<bool>? NullMaskMemory`
  (`:215-224`, interface at `Interfaces.cs:98`). `HasNulls` is `nullMask is not null`, so an
  absent mask *is* the no-null state — there is no all-false mask, and an explicit empty mask is
  normalised to absent.
- `ARCHITECTURE.md` §"Null Representation Decision" records explicit validity masks over sentinel
  values, but its `Consequences` say only "Slight memory overhead". It never states that the
  byte-wise `Span<bool>` form is the deliberate canonical representation.
- `docs/plan/ARROW-ROADMAP.md` Phase C still says the byte-wise mask "was never a documented design
  decision" and offers two representations to "pick one primary".
- `docs/plan/ARROW-REVIEW.md` §3 scorecard row 2 still says "no documented layout decision".

## Grounding (G1)

**Apache Arrow columnar format v1.5, "Validity bitmaps":**

- Validity is a dedicated bit-packed buffer, at least 1 bit per slot. `1` (set) means *not null*;
  `0` means null. `is_valid[j] -> bitmap[j / 8] & (1 << (j % 8))`, **LSB numbering**.
- Bitmaps are initialised all-unset at allocation time, including padding.
- Arrays having a 0 null count **may choose not to allocate** the validity bitmap; consumers must
  handle both the elided and always-allocated cases.

Two consequences for the wording, both verified in our own code:

1. **The boundary conversion inverts polarity as well as packing.** Nivara's mask is `true = null`;
   Arrow's bit is `1 = valid`. `CreateFixedSizeBinaryArray` already does exactly this — it sets the
   bit only for the non-null branch (`validity[i / 8] |= (byte)(1 << (i % 8))`).
2. **Both sides have an "absent" convention**, but they are different representations of the same
   idea. The bridge lives at the boundary, not in storage.

**What actually exists today at the boundary** (this is what makes the honest wording possible):

- *Export* is per-element through Apache.Arrow builders (`AppendNull()` / `Append(value)`), except
  `CreateFixedSizeBinaryArray` (Guid), which writes the bitmap by hand.
- *Import* is per-element: `ConvertArrowColumnToNivaraColumnTyped` walks chunks, calls
  `chunk.IsNull(i)`, and materialises a `List<object?>` of boxed values.
- There is **no** buffer-level `bool[]`↔bitmap helper. That is a Phase C deliverable that is
  still open, and the docs must not claim otherwise.

**Microsoft Learn corroboration** for the fast-path rationale: `Span<T>`/`Memory<T>` guidance
frames spans as the allocation/copy-avoiding currency of hot paths, which is the reason every
null-aware kernel already consumes `ReadOnlySpan<bool>`.

**Blast radius:** documentation only. No source file changes, no public API change, no behaviour
change. Affected: `ARCHITECTURE.md`, `docs/plan/ARROW-ROADMAP.md`, `docs/plan/ARROW-REVIEW.md`,
`AGENTS.md` (ADR index), a new `docs/adr/007-*.md`, and `tests/Nivara.Tests/Docs/DocsCoveragePins.cs`
(two pinned counts). Downstream callers of the mask are untouched; `NullHandlingPropertyTests` and
the existing mask tests stay green because nothing in the representation changes.

## Findings that widen the work (confirmed with the human)

The human asked for the adjacent defects to be fixed in this change too:

1. **Stale citations in the Arrow lens docs.** Every one of these resolves (so the citation gate is
   green) but points at the wrong symbol:

   | Citation | Claimed to show | Actually | Retarget |
   |---|---|---|---|
   | `ColumnStorage.cs:18` | strings are `T[]` reference arrays | immutability remarks | `:32` |
   | `ColumnStorage.cs:21` | nulls are `bool[]`, 1 byte/element | `tensorView` remarks | `:35` |
   | `ColumnStorage.cs:238-242` | `AsSpan()` returns a slice | body of `AsTensor()` | `:298-302` |
   | `:167-182` (unsuffixed) | `Slice` shares the backing array | the indexer | `:183-198` |
   | `ArrowInterop.cs:688-710` ×3 | chunked arrays flattened on import | `CreateInt64Array(nint)` | `:999-1015` |
   | `ArrowInterop.cs:824-849` | import boxes through `List<object?>` | `CreateTimestampArray(DateTimeOffset)` | `:973-1057` |
   | `Schema.cs:9-300` | immutable `Schema` | class is 10-309 | `:10-309` |

   All retargets are **count-preserving** (replace, never add or remove).

2. **Stale storage section in `ARCHITECTURE.md`.** §"Memory vs Tensor Storage Strategy"
   (`611-637`) describes a `MemoryStorage`/`TensorStorage` split that the storage consolidation
   removed; it is fully duplicated by §"Storage Backend Selection" and §"Tensor Interop Boundary".
   Adjacent wording at `607-609` is stale too ("`T[]`, `Memory<T>`, or tensor-backed storage",
   "validity bitmap").

## Proposed changes

### Commit 1 — `docs(adr): record the byte-wise validity mask as the canonical fast path`

- New `docs/adr/007-byte-wise-null-mask-fast-path.md` in the house format (matching ADR-006:
  `# ADR-007: …`, `**Status:** Accepted`, `**Date:**`, `## Context`, `## Decision`,
  `## Consequences`). `text` fences only, no `csharp` fences, and **no `File.cs:NN` citations**.
- Decision text: Option A. Byte-wise `ReadOnlySpan<bool>` is canonical in core; bit-packed validity
  exists only at the Arrow boundary. Records the absent-mask invariant, the `NullMask` /
  `NullMaskMemory` shapes, `Slice` keeping the mask aligned 1:1 with data, and the honest boundary
  statement (polarity inversion + packing; element-wise today; buffer-level helper still open).
- `AGENTS.md`: one-line index entry under "Architectural Decisions (ADRs)".
- `DocsCoveragePins`: `ExpectedDocumentCount` 53 → 54 and `ExpectedDocumentReferenceCount`
  226 → 229 (one for the AGENTS.md index line, two for the plan documents ADR-007 names),
  extending each doc comment's arithmetic rather than only bumping the number.

### Commit 2 — `docs(architecture): ratify the byte-wise Span<bool> null mask`

- Rewrite §"Null Representation Decision" into: Decision → Canonical representation → Arrow
  boundary → Alternatives → Why chosen → Consequences → Revisit trigger.
- Replace "Slight memory overhead" with the honest figure: **1 byte per element versus Arrow's
  1 bit — 8× on the mask, only when nulls are present** — plus why byte-wise wins: every null-aware
  kernel already consumes `ReadOnlySpan<bool>`, so bit-packing would charge pack/unpack to every
  kernel to save memory that is small beside the payload.
- State the boundary contract: packing **and** polarity inversion, per-element today, buffer-level
  helper not yet implemented (Phase C).
- Revisit trigger: validity becomes a dominant memory consumer, or Phase D needs a buffer-level
  validity handoff.
- Delete the stale §"Memory vs Tensor Storage Strategy"; fix the stale `607-609` wording.
- No `File.cs:NN` and no new `docs/…md` path in this file (it is in both pin scopes). Reference
  "ADR-007" by number only.

### Commit 3 — `docs(arrow): correct stale citations and mark Phase C's decision landed`

- `docs/plan/ARROW-ROADMAP.md` Phase C: motivation, scope (decision recorded, remaining work is the
  bulk conversion helper), acceptance criteria split into decided vs still open; link to ADR-007
  (free — `docs/plan/` is outside the widened scope).
- `docs/plan/ARROW-REVIEW.md`: dated ratification note on Pillar 2; scorecard row 2 keeps "1 byte
  per element, not bit-packed" (still true), drops "no documented layout decision", and keeps
  "no bool↔bitmap path" (still true).
- Apply every retarget in the table above, count-preservingly.

## Verification

```
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
```

Docs namespace is ~66 tests / ~25 s. Expected after the branch: all pins at their new values
(107 / 18 / 54 / 24 / 229), every citation resolves, `AdrIndexTests` green in both directions.

## Planned commits

1. `docs(adr): record the byte-wise validity mask as the canonical fast path`
2. `docs(architecture): ratify the byte-wise Span<bool> null mask`
3. `docs(arrow): correct stale citations and mark Phase C's decision landed`
4. `docs: remove TODO.md — plan executed` (after G2 clears)

## Follow-ups (file as issues, do not bundle)

- Buffer-level `bool[]`↔bitmap conversion for the Arrow boundary (the still-open Phase C item; the
  import path also still boxes through `List<object?>`).
- Sweep the same citation-drift class across `docs/plan/POLARS-*` and `AISTACK-*`.

## GitHub issues log

- [ ] #840 — Arrow boundary: buffer-level `bool[]` → bit-packed validity conversion (created while working on #832; the still-open Phase C item)
- [ ] #841 — Docs: sweep stale `File.cs:NN` citations across the remaining plan documents (created while working on #832)
