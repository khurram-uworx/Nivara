# ADR-007: The Byte-Wise `Span<bool>` Null Mask Is the Canonical Validity Fast Path

**Status:** Accepted  
**Date:** 2026-11-02

## Context

Nivara tracks nulls with an explicit validity mask rather than a sentinel value (ADR-001). The
storage consolidation left one sole-owner `ColumnStorage<T>` for every element type, holding a
`T[]` data buffer plus an optional `bool[]` null mask. Two properties of that mask are load-bearing
and were never written down as a decision:

- The mask is **byte-wise** — one `bool` (one byte) per slot, not bit-packed.
- The mask is **absent** when a column has no nulls. `HasNulls` is `nullMask is not null`, an
  explicit empty mask is normalised to absent, and there is no all-false mask.

Every null-aware kernel consumes the mask as `ReadOnlySpan<bool>`, and one internal member,
`NullMaskMemory`, exposes it as `ReadOnlyMemory<bool>?` so the fused span kernel can OR an output
mask in the same pass as the values. Apache Arrow instead mandates a **bit-packed** validity
buffer, 1 bit per slot, least-significant-bit numbering, where a set bit means *valid* and a zero
bit means *null*; Arrow arrays with a null count of zero may omit that buffer entirely.

`docs/plan/ARROW-ROADMAP.md` Phase C framed this as an open choice between a bit-packed primary
representation and a documented byte-wise fast path, and `docs/plan/ARROW-REVIEW.md` scored the
byte-wise mask as "no documented layout decision". That left an accidental-looking representation
reading as an undecided one.

## Decision

**Adopt Option A: the byte-wise `ReadOnlySpan<bool>` mask is the canonical validity representation
in core. Bit-packed validity exists only at the Arrow boundary.**

1. **Core storage never stores a bitmap.** `ColumnStorage<T>` keeps its optional `bool[]`, sliced
   together with the data so the mask stays aligned 1:1 with the values in every view.
2. **`true` means null.** The polarity is part of the contract and is not inverted anywhere inside
   the engine.
3. **Absent means no nulls.** A null mask is absent unless a column actually contains nulls; no
   all-false mask is materialised.
4. **The public and internal shapes are the fast path.** `IColumnStorage<T>.NullMask` is
   `ReadOnlySpan<bool>`, and `NullMaskMemory` is `ReadOnlyMemory<bool>?` for the fused output-mask
   OR. No core member unpacks or packs bits.
5. **Packing, and polarity inversion, happen at the Arrow boundary only.** Apache Arrow's validity
   buffer is bit-packed, LSB-first, with `1` meaning *valid*; converting Nivara's mask therefore
   inverts polarity as well as packing. That conversion belongs to the interop layer and never
   leaks into storage or kernels.
6. **The boundary conversion is element-wise today.** Export walks each value through the
   Apache.Arrow builders, except the fixed-size-binary (Guid) path, which writes the bitmap
   directly; import reads each slot and materialises boxed values. A buffer-level mask-to-bitmap
   helper now exists and is unit-tested (`ArrowValidityMask`, internal to the interop assembly),
   but `ArrowInterop` does not call it yet, so the conversion it performs is still element-wise.

## Consequences

**Positive:**

- No pack/unpack cost at any kernel boundary. The mask is already the shape every span kernel
  wants, including the fused pass that ORs leaf masks into an output mask.
- No public API change and no change to null semantics: mask-OR propagation, comparisons
  false-at-null, and the ADR-001 boundary all stay exactly as they are.
- A column with no nulls pays nothing for validity, which matches Arrow's own allowance for
  eliding the validity buffer when the null count is zero. Both sides have an "absent" convention;
  only the boundary has to reconcile them.
- The representation is inspectable — a debugger shows null positions as plain booleans.

**Negative / limits:**

- 1 byte per element against Arrow's 1 bit: **8× the validity memory, only when nulls exist**. It is
  additive to the data payload, which for typical element widths is the larger term, but it is real
  and is not "slight".
- The Arrow boundary converts per element today, so interop pays both a packing cost and a boxing
  cost until the buffer-level helper is wired into the boundary.

**Revisit when:**

- validity becomes a dominant memory consumer (for example very wide frames where nulls are dense
  and element widths are small), or
- Phase D needs a buffer-level validity handoff for real zero-copy, at which point the packing cost
  is paid once at the boundary rather than per kernel.
