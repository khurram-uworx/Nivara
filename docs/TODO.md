# Plan: #836 — decision gate for null-bearing Arrow zero-copy

**Branch:** `khurram/836`
**Issue:** [#836](https://github.com/khurram-uworx/Nivara/issues/836) — Arrow interop: decision gate for
null-bearing zero-copy after #835
**Depends on:** #832 / #833 / #834 / #835, all landed. #835 landed as PR #847
(`0673cc83 feat(io): real zero-copy Arrow export for flat null-free columns`).

## Problem

#836 says Branch A (extend zero-copy to null-bearing columns) may start **only if #835
demonstrates a safe, measurable benefit**. It does not yet, and the gap is not in the mechanism:

1. **No correctness test covers `UseZeroCopy`.** `grep UseZeroCopy *.cs` yields three hits —
   `ArrowInterop.cs:376`, `ArrowConversionOptions.cs:20`, `ArrowInteropPerfTests.cs:85`. Nothing
   asserts that the Arrow buffer actually aliases the column, that a slice offset survives, that the
   lifetime survives disposal, that the loud failure fires, or that IPC serialization works for a
   buffer whose length is not an 8-byte multiple.
2. **The performance claim rests on a single-sided best-of-5** (`ArrowInteropPerfTests.cs:124`), with
   no interleaved A/B and no assertion. The PR comment's "4.00 ms vs ~0.00 ms" is that measurement.
3. **The docs say the option does not exist.** `CHANGELOG.md:1333`, `docs/plan/ARROW-REVIEW.md:9,97,138`
   and `docs/plan/ARROW-ROADMAP.md:17,23,143` all describe `UseZeroCopy` as *removed*. There is no
   CHANGELOG entry for #835.
4. **The loud failure is misclassified.** A null-bearing column throws `InvalidOperationException`,
   which `ToArrowTable`'s per-column catch (`ArrowInterop.cs:85`) rewraps as
   `DataCorruptionException` — "Failed to convert column 'x' …". Data is not corrupt; the request is
   unrepresentable. A frame containing one `string` column fails the same way, so the option is
   unusable on any realistic mixed frame.

Latent defects in the landed code:

- `UnmanagedMemoryManager<T>.Pin(int elementIndex)` (`ArrowInterop.cs:1028`) treats `elementIndex` as
  an element index, but for `MemoryManager<byte>` it is a **byte** index; the offset is multiplied by
  `Unsafe.SizeOf<T>()`. Correct at `0`, wrong otherwise. `Unpin()` (`:1039`) is a no-op while `Pin`
  allocates a `GCHandle`.
- `CreateZeroCopyHalf` (`:993`) duplicates the body of `CreateZeroCopyPrimitive` (`:974`) for `Half`.
  Branch A would have to change both.

## Decision already taken with the human

- **Sequencing:** evidence first, then Branch A. Steps 1–4 must be green before Step 5; if Step 1's
  IPC test fails, the shipped null-free path is unsound and the branch resolves as Branch B.
- **Ineligible-column behaviour:** throw `UnsupportedTypeException` **naming the column**, not
  `DataCorruptionException`. No silent copy fallback.

## Proposed changes

### Step 1 — Evidence for the null-free path as landed (no capability change)

New fixture `tests/Nivara.Tests/IO/ArrowZeroCopyTests.cs`:

| Test | Pins |
| --- | --- |
| `ToArrowTable_ZeroCopyFlatPrimitive_SharesColumnBackingArray` | Buffer identity: `MemoryMarshal.TryGetArray` over the Arrow data buffer returns the same `T[]` instance the column was built from (`CreateFromOwnedArray`), offset 0. |
| `ToArrowTable_ZeroCopySlicedColumn_PreservesOffsetAndValues` | `segment.Offset` → `UnmanagedMemoryManager._start` (`:979-984`) and value alignment. |
| `ToArrowTable_ZeroCopyColumn_AfterColumnDisposed_ArrowArrayStillReads` | The memory manager's strong `T[]` reference carries the lifetime. |
| `ToArrowTable_ZeroCopyThreeRowIntColumn_IPCRoundTrip` | **Decisive.** 3 rows = 12 bytes, not an 8-byte multiple. Write through the Arrow IPC writer and read back. |
| `ToArrowArray_ZeroCopySeries_MatchesTablePath` | The series path routes through the same `ConvertColumnToArrowArray` (`:218`). |
| `ToArrowTable_ZeroCopyNullBearingColumn_FailsLoudly` | Exception type + message names the column. Lands with Step 2. |

If the IPC round-trip fails, stop, report, and resolve the issue as Branch B.

### Step 2 — Report ineligible zero-copy requests as `UnsupportedTypeException`

- Add a `Type`+`reason` overload to `UnsupportedTypeException` (`src/Nivara/IO/IOExceptions.cs:62`) so
  the message can carry the column name without lying about the type ("`DateTime` is not supported
  for I/O" would be false — it is supported, just not zero-copy).
- Thread an optional column name into `ConvertColumnToArrowArray` → `ConvertColumnToArrowArrayTyped`
  → `CreateZeroCopyArrowArray`; `ToArrowTable` passes `columnName`, `ToArrowArray` passes
  `series.Name`.
- Replace the `InvalidOperationException` at `:952-955` with `UnsupportedTypeException`. It is already
  excluded from all three wrapping catches (`:85`, `:101`, `:220`), so it surfaces unrewrapped.

### Step 3 — Fix the landed defects

- `UnmanagedMemoryManager<T>.Pin` offset units; release the `GCHandle` in `Unpin`.
- Collapse `CreateZeroCopyHalf` into `CreateZeroCopyPrimitive` using `Unsafe.SizeOf<T>()` instead of
  the `elementSize` parameter (`Half` is `unmanaged`).

### Step 4 — Measurement harness (human-run)

Add an interleaved A/B median-ratio probe to `tests/Nivara.Tests/IO/ArrowInteropPerfTests.cs`
(`Category("Performance")`, `TimingGuards.RequireOptimizedBuildForTiming()`), alternating which side
runs first. No timing assertion — the claim is structural (no copy code path) with timing as
corroboration. The human runs it in Release; nothing here polls.

### Step 5 — Branch A: data-buffer zero-copy for null-bearing flat primitives

- In `CreateZeroCopyPrimitive`: build validity with `ArrowValidityMask.ToValidityBitmap` when
  `HasNulls` (it can never alias — polarity inverts and bits pack), pass `column.NullCount` instead of
  `0`, and drop the `HasNulls` throw.
- Tests: `NullCount` correct, validity bits match the copy path, `FromArrowTable` round-trip preserves
  nulls, sliced null-bearing column honours `maskStart`.
- Document the one new semantic: **values are a live view, null topology is a snapshot.** Only
  internal write-through can change values (`NivaraColumn.AsWritableSpan` is `internal`,
  `NivaraColumn.cs:1266`; public `Create(T[])` copies, `ColumnStorage.cs:59`).

### Step 6 — Documentation (Branch B substance, lands regardless)

- `ArrowConversionOptions.cs:13-19` — export-only, data-buffer sharing, and the exact eligibility rule.
- `CHANGELOG.md` `[Unreleased]` — the missing #835 entry, plus this branch's additions.
- `docs/plan/ARROW-ROADMAP.md` Phase D — status; cancel the "both directions" goal and the
  "Dependencies: Phases A–C" precondition for the export half (they were bypassed).
- `docs/plan/ARROW-REVIEW.md:9,97,138` and the §3 scorecard line 82 — stale for null-free export.
- `AGENTS.md` Known Issues — the "zero-copy Arrow arrays removed" bullet.
- Docs pins: a new `File.cs:NN` citation moves `DocsCoveragePins.ExpectedCodeCitationCount` (24); a new
  `docs/…md` reference moves `ExpectedDocumentReferenceCount` (229). `docs/` is scanned recursively, so
  `ExpectedDocumentCount` (54) moves only if a document is added.

## Verification

- Targeted: `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.IO"`.
- Docs change → whole namespace: `dotnet test -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"`
  (66 tests, ~25 s).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Full suite only with human confirmation, always carrying
  `--filter "Category!=Performance&Category!=ReleaseReadiness"`.
- Human-run: the interleaved A/B probe in Release.

## Planned commits

1. `docs: plan #836 decision gate in TODO.md`
2. `test(io): pin zero-copy Arrow export behaviour`
3. `fix(io): report ineligible zero-copy requests as UnsupportedTypeException`
4. `fix(io): correct UnmanagedMemoryManager pin offset units and fold the Half path`
5. `feat(io): data-buffer zero-copy for null-bearing flat primitive columns`
6. `docs(arrow): record the zero-copy boundary and the #835 landing`
7. `docs: remove TODO.md — plan executed`

## Blast radius

- `src/Nivara.Extensions/IO/ArrowInterop.cs` — export helpers; callers are `ToArrowTable`,
  `ToArrowArray<T>`, `FromArrowTable` (untouched).
- `src/Nivara.Extensions/IO/ArrowConversionOptions.cs` — doc comment only.
- `src/Nivara/IO/IOExceptions.cs` — additive constructor overload on `UnsupportedTypeException`;
  `IOExceptionTests` covers the existing surface, no existing constructor changes.
- `src/Nivara/Storage/ColumnStorage.cs` — read-only use of `Data`; no change.
- Tests: new `ArrowZeroCopyTests`; `ArrowInteropPerfTests` gains a probe. Existing `ArrowInteropTests`,
  `ArrowInteropBufferValidityTests`, `ArrowExtendedDomainRoundTripTests` must stay green — the default
  `UseZeroCopy = false` keeps every existing path byte-identical.
- Docs: `docs/plan/*` (in docs-gate scope), `CHANGELOG.md`, `AGENTS.md`, `DocsCoveragePins`.
- Risk: a public option whose default is `false`; the only behavioural change is the exception type and
  the newly eligible null-bearing export.

## GitHub issues log

- [ ] #836 — decision gate (this plan).

Create an issue immediately when deferred work or a concern is found during execution; record the
number here rather than holding it in memory.
