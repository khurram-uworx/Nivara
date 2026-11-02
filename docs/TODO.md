# Plan — span-first Arrow validity mask conversion (issue #833)

As each task executes, if a follow-up or concern appears that this plan does not own, create a GitHub
issue immediately (`gh issue create --repo khurram-uworx/Nivara`) and record its number in the
GitHub issues log below. Do not hold it in memory; compaction loses it.

## Problem

Nivara tracks nulls as a byte-wise `ReadOnlySpan<bool>` mask where **`true` means null** (ADR-007).
Apache Arrow tracks validity as a **bit-packed** buffer, LSB-first, where **`1` means valid**. The
conversion therefore packs *and* inverts polarity, and it happens at the interchange boundary only.

Today that conversion is element-wise: export walks every value through the Apache.Arrow builders
(`ArrowInterop.cs:409–780`), apart from the Guid path, which hand-writes the bitmap
(`ArrowInterop.cs:785–813`); import walks every slot and boxes into a `List<object?>`
(`ArrowInterop.cs:973–1057`). No buffer-level helper exists.

Issue #833 is the implementation half of ARROW-ROADMAP Phase C under Option A: add the span-first
conversion, with tests, and **do not** integrate it into `ArrowInterop` yet.

## Verified current state

| Fact | Evidence |
|---|---|
| Core mask is `ReadOnlySpan<bool>`, empty when the column has no nulls | `Interfaces.cs:92`, `NivaraColumn.cs:1601` |
| Polarity precedent — bit set only on the non-null branch | `ArrowInterop.cs:801` |
| Import boxes per element | `ArrowInterop.cs:1003–1014` |
| No bitmap helper anywhere in `src/` or `tests/` | `grep -riE "validity\|bitmap\|bitpack"` |
| `Nivara.Extensions` already exposes internals to `Nivara.Tests` | `Nivara.Extensions.csproj:18–24` |
| Core normalises an empty mask to "absent" | `ColumnStorage.cs:145–148`, `NivaraColumn.cs:1601–1606` |

## Decisions taken with the human (this session)

1. **`internal`, not `public`.** `ArrowInterop` lives in the same assembly, so the integration issue
   needs no visibility change, and no half-wired conversion enters the NuGet surface.
2. **Add a length-aware export overload and `GetBitmapLength`.** Without a row count, an *absent*
   mask (length 0) cannot mean "all valid" on export. Import can express it (`ToNullMask(bitmap,
   length)`), so both rules in the issue need the length overload to actually hold.
3. **Update the superseded "no buffer-level helper" claims in this change.** Origin/main already
   carries ADR-007 (merged after the plan-mode read), so the claim set is five files, not two. The
   edits stay prose-only: no `File.cs:NN` citation added or removed, and no `docs/…md` reference
   added or removed, so `DocsCoveragePins` does not move.

## Proposed changes

### Commit 1 — `feat(io): add span-first Arrow validity mask conversion`

New `src/Nivara.Extensions/IO/ArrowValidityMask.cs`, `namespace Nivara.IO`, `internal static class`,
XML docs on every member:

```csharp
internal static int GetBitmapLength(int length);                                    // ((length - 1) >> 3) + 1, 0 for 0
internal static void ToValidityBitmap(ReadOnlySpan<bool> nullMask, Span<byte> destination);
internal static void ToValidityBitmap(ReadOnlySpan<bool> nullMask, int length, Span<byte> destination);
internal static byte[] ToValidityBitmap(ReadOnlySpan<bool> nullMask);
internal static void ToNullMask(ReadOnlySpan<byte> validityBitmap, Span<bool> destination);
internal static bool[] ToNullMask(ReadOnlySpan<byte> validityBitmap, int length);
```

Contract:

- **Polarity and order** — `true` (null) writes a `0` bit; a value writes `1`, LSB-first. The
  Apache.Arrow builder is the authority for this, not our reading of the spec, so a test cross-checks
  against a real `Int32Array`'s validity buffer.
- **Padding** — bits past `length` in the final byte are always `0`, on the packed path and on the
  all-valid fill (`0xFF` then mask the last byte with `(1 << (length & 7)) - 1`).
- **Absent input** — an empty `nullMask` allocates an empty `byte[]`; an empty or short
  `validityBitmap` yields an all-false mask (every position valid).
- **Sizing** — `destination.Length >= GetBitmapLength(n)`; bytes past the required count stay
  untouched; too small throws `ArgumentException`; negative `length` throws
  `ArgumentOutOfRangeException`; a length-aware call whose `nullMask.Length` is neither `0` nor
  `length` throws `ArgumentException`.
- **No normalisation** — import returns a full-length `bool[]`; deciding absent-vs-materialised
  stays with `ColumnStorageFactory`.
- **Loop shape** — byte-building, eight slots per iteration with a scalar tail. No SIMD and no
  timing or allocation assertion, so no `[Category("Performance")]`.

New `tests/Nivara.Tests/IO/ArrowValidityMaskTests.cs`, NUnit 4, `Method_Scenario_ExpectedBehavior`:
`GetBitmapLength` (0, multiples of 8, round-up, `int.MaxValue` → 268435456), empty / all-valid /
all-null / mixed with hand-computed bytes (`{null, valid, null}` → `0x02`, exact, not a tolerance),
every length 1–16, seeded-random lengths to 1000, both round-trip directions, all four validation
failures, padding-bit assertions, tail-untouched, and the Apache.Arrow builder cross-check over the
first `ceil(n/8)` bytes.

### Commit 2 — `docs(arrow): record the validity helper as landed, not open`

Prose-only, in the five claim sites:

- `docs/adr/007-byte-wise-null-mask-fast-path.md` — §6 last sentence and the Consequences bullet.
- `ARCHITECTURE.md:159–163` and the consequences bullet at `:170`.
- `AGENTS.md` — the ADR-007 index line.
- `docs/plan/ARROW-ROADMAP.md` — Phase C remaining scope and the "*Open:* buffer-based" criterion.
- `docs/plan/ARROW-REVIEW.md:42` and `:81`.

Wording that must survive: the *boundary* is still element-wise, because integration is out of
scope. The honest statement is "the buffer-level helper exists and is unit-tested; `ArrowInterop`
does not call it yet."

## G1 grounding (microsoft-learn + code-memory)

- **Rejected alternative — `System.Collections.BitArray`.** .NET 11 added span constructors
  (`ReadOnlySpan<bool>`, `ReadOnlySpan<byte>`, `ReadOnlySpan<int>`) precisely to avoid an
  intermediate array. It is still wrong for this boundary: constructing one allocates a `BitArray`
  plus its internal `int[]` copy, and there is no `CopyTo(Span<byte>)` span export, so the pack
  direction would have to go through `Array` anyway. It also does not invert polarity. The helper
  keeps its own span loops.
- **Bit order corroborated.** The same documentation example shows `bits[0]` reading the
  least-significant bit of the first byte (`0b_0000_0011` → indices 0 and 1 true), which is the
  LSB-first numbering Arrow mandates and the helper implements. This is documented behaviour, not
  an assumption.
- **Span guidance.** The performance guidance frames `Span<T>` and `ref` safety as the
  allocation-reducing currency of hot paths, which is why the helper takes and returns spans and
  offers only the two allocation conveniences the issue asks for.
- **Blast radius re-confirmed.** code-memory records no symbol named `ArrowValidityMask`,
  `ToValidityBitmap` or `GetBitmapLength`, and no product code calls the new type by construction.
  The change adds nodes without adding edges: nothing in `src/` changes behaviour. The only
  consumers are the new tests.

No decision or red flag surfaced that the human has not already resolved.

## Verification

```
dotnet build Nivara.slnx -c Release
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.IO.ArrowValidityMask"
dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore
dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"
```

The full suite is not run without asking first.

## Blast radius

- **New code** — one internal static class plus one test file. Nothing calls it from product code
  until the follow-up integration issue, so `ArrowInterop` behaviour, the query engine, and storage
  are untouched. This is deliberate per the issue's non-goals.
- **Docs** — five files, prose only. Pinned counts stay at 107 / 18 / 54 / 24 / 229 unless an edit
  changes a citation or document reference; the closing review re-checks this.
- **Package surface** — unchanged (internal type).

## Planned commits

1. `feat(io): add span-first Arrow validity mask conversion`
2. `docs(arrow): record the validity helper as landed, not open`
3. `docs: remove TODO.md — plan executed` (after G2 clears)

## GitHub issues log

- [x] #843 — wire `ArrowValidityMask` into `ArrowInterop` and retire the per-element validity
      paths (created while working on #833; still open).

## G2 review outcome (before deletion)

**Review (a) — branch work as a whole vs issue #833.** All acceptance criteria met literally:
round-trips exact (mixed, 1–16, 1–1000 seeded), empty/all-null/all-valid covered, destination
length validated (too small, too short, mismatch, negative), no core or public API change
(new type is `internal`), `ArrowInterop` untouched, LSB-first order corroborated by the
Apache.Arrow builder cross-check. One finding: the two-argument pack overload silently wrote
nothing for an empty mask into a pre-sized buffer, which reads as all-null — fixed as commit
`cf39277c` with a regression test. No other gap.

**Review (b) — branch work vs this plan.** Every planned change landed: `GetBitmapLength`,
six pack/unpack methods, 30 tests, the five docs claim sites, and the follow-up issue. Blast
radius matches: one internal class, one test file, prose-only docs. Pins unchanged (107 / 18 /
54 / 24 / 229); the Docs namespace (66 tests) passes. Both reviews clear, so `docs/TODO.md`
is removed next.
