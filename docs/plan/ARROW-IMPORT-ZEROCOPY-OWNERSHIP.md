# Design Note: Arrow Import Zero-Copy Ownership Rule

Status: **Design only.** No implementation is landed by this document.
Issue: #861. Related: #835, #836, #840, #843, `docs/plan/ARROW-ROADMAP.md` Phase D.

This note defines the ownership/lifetime rule required before `ArrowInterop.FromArrowTable` /
`FromArrowArray` can alias an externally owned Arrow buffer instead of copying. It is a
prerequisite for the remaining import half of Phase D. **It does not by itself land Phase A
(chunked columns) and does not land Phase D import.** Chunked compute kernels and any
memory-backed storage work remain separate. No changes to `ColumnStorage<T>` or
`ArrowInterop` are made here.

## 1. Current storage model (unchanged by this note)

`src/Nivara/Storage/ColumnStorage.cs` is intentionally a sole-owner, contiguous `T[]` plus an
optional `bool[]` null mask (true at null). It has no reference counting and no disposal of
underlying arrays:

- `Dispose()` only sets a logical `disposed` flag, drops the cached `Tensor<T>` view, and
  does **not** free the backing arrays. Arrays therefore stay valid for consumers (including
  a previously exported Arrow array) as long as GC keeps them alive. This is what makes the
  export zero-copy path in `CreateZeroCopyPrimitive` safe.
- `Slice(start, length)` is a true zero-copy view over the same buffer with `dataStart` /
  `maskStart` offsets; it already shares arrays rather than copying.
- Kernels and tensor views assume sole ownership and treat `AsSpan()` / `TryGetSpan(...)` as
  views over one contiguous owner.

External buffers therefore cannot be substituted transparently: a storage must either own a
managed `T[]`, or storage must grow a memory-backed form. That larger decision is out of scope
for this note; this note only defines the rule such a change would have to satisfy.

## 2. Arrow memory ownership (reference)

Apache.Arrow represents buffers as:

- `ArrowBuffer` (a value type) wrapping a `SharedMemoryHandle`,
- which wraps a `SharedMemoryOwner`, which wraps an `IMemoryOwner<byte>` plus a refcount;
- `ArrowBuffer.Retain()` increments the refcount and returns a sharing buffer;
- `ArrowBuffer.Dispose()` decrements it and releases the inner owner when it reaches zero.

`ArrayData` carries `Offset` (in element units), `Length`, `Buffers[]` and children; a
`Column` / `ChunkedArray` is a sequence of those buffers, not one. IPC readers and decompressors
may recycle or dispose their buffers; a zero-copy import must `Retain()` every buffer it
aliases before the reader leaves scope. If Nivara never retains, the aliased memory becomes
invalid once the reader drops its last reference.

## 3. Ownership options considered

| Option | Mechanism | Verdict |
|---|---|---|
| A. Managed array alias | Cast Arrow's `Memory<byte>` to a `T[]` view | Not available: Apache.Arrow stores `byte[]`/`Memory<byte>`, not typed `T[]`; `MemoryMarshal` cannot create a managed `T[]` over bytes. Would require copying or changing storage's core type. |
| B. Retained lease | Hold a refcounted `ArrowBuffer`/`IMemoryOwner<byte>` lease alongside a span/view into it; dispose the lease when the storage is disposed | The intended vehicle. It keeps the sole-owner *value* semantics while extending lifetime to external buffers. Requires a memory-backed storage variant (not landed here). |
| C. Pinned native pointer | Pin and cast an Arrow (native or pooled) buffer to `T*` and expose spans | Possible but charges pinning, alignment and endian checks; buffers repinned per column lifetime; pinning pressure and GC fragmentation. Defer unless profiling demands it. |
| D. Copy (status quo) | Materialise a managed `T[]` | Always safe and the default. |

Recommendation: **(B) for a future implementation, (D) until then.** This note does not select
(C), and never silently switches from (D) to an alias under a flag.

## 4. Aliasing vs. copying conditions

A zero-copy import may engage **only when all** of the following hold:

1. **Opt-in.** The caller explicitly requests zero-copy on the import path. Today
   `ArrowConversionOptions.UseZeroCopy` is documented as export-only; its import semantics must
   be updated only when the implementation lands.
2. **Flat, fixed-width, unmanaged element type.** `T` must be unmanaged and have a fixed Arrow
   layout (`byte`, `sbyte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `float`,
   `double`, `Half`). `bool` (Arrow bit-packed), `string`/`char`, dates, decimals, lists,
   structs and dictionary-encoded arrays are never aliased; they copy.
3. **Element-aligned slice.** `ArrayData.Offset * sizeof(T)` must be a multiple of the
   element size and the buffer length must hold exactly `count * sizeof(T)` bytes from that
   offset.
4. **Host endianness.** Arrow is little-endian; aliasing requires the host (and storage) to
   be little-endian. Otherwise copy.
5. **Chunk-preserving.** Under `PreserveChunkStructure`, each Arrow chunk must be imported as
   its own storage; no chunk flattening or coalescing across aliased buffers. Offsets are
   retained, not normalized into a fresh contiguous array.
6. **Retained lifetime.** Every aliased buffer is `Retain()`ed before its Arrow reader/column
   goes out of scope, and released exactly once by Nivara on storage disposal. Slices share the
   retained whole buffer (amplification is documented, matching existing `Slice` behavior).
7. **Validity is rebuilt, not aliased.** Arrow validity is bit-packed with a set bit meaning
   *valid*; Nivara's mask is byte-wise with true at *null*. Polarity and representation
   both differ, so the bitmap cannot alias even in theory. Conversion uses the existing
   `ArrowValidityMask` buffer-level path. This keeps import consistent with export, which
   likewise shares only the values buffer.
8. **Value-at-null policy.** Arrow leaves data at null positions undefined, while Nivara
   tensor/views assume `default(T)` there. Either (a) define the imported column as
   mask-authoritative with unspecified values at null positions, or (b) require a
   normalization pass (a copy) before aliasing. This must be stated before any implementation
   is accepted.

When any condition fails:

- if zero-copy was **not** requested, use the copying path (today's behavior, no diagnostics
  change required);
- if zero-copy **was** requested, **throw**, never silently copy. Throw
  `UnsupportedTypeException` naming the column and the ineligible condition for type/layout
  failures, and `NivaraIOException` / `InvalidOperationException` with an `OperationContext`
  of `ArrowInterop.FromArrowTable` (or `FromArrowArray`) for retention or lifetime failures.
  Following the export contract (`CreateZeroCopyArrowArray`): an explicit request on an
  incompatible shape fails loudly.

A successful alias decision should be observable via `OperationDiagnostics`
(`OperationType` noting the import path, `KernelUsed`, length, element type, `HadNulls`, and
`Notes` recording the alias decision, retained chunk count, and the rebuilt-validity/undefined-
null-slot policy applied).

## 5. Lifetime interaction with slices and chunks

- **Slices.** A sliced import aliases the retained underlying buffer with an offsets adjustment,
  exactly like `ColumnStorage.Slice`. It does not copy unless normalization is required.
- **Chunk boundaries.** Aliasing never crosses chunk boundaries in a way that loses boundaries;
  `PreserveChunkStructure` keeps one Nivara storage per Arrow array chunk, each with its own
  lease. Coalescing or compacting requires a copy and is therefore a separate operation.
- **IPC readers.** `Retain()` must occur before the `ArrowStreamReader`/`ArrowMemoryReader`
  is disposed or a record batch is released. Nivara's storage then keeps the buffer alive
  beyond the reader's lifetime; `Dispose()` on the storage releases Nivara's reference.
- **Disposal contract extension.** Today `ColumnStorage.Dispose()` is non-releasing. To support
  external leases safely, aliased storage must (future) dispose its lease, drop `tensorView`,
  raise `ObjectDisposedException` from `Data`/`AsSpan()`/indexers afterwards, and ensure no
  later reader can touch released memory. Shared-buffer semantics of `Slice` (source array
  visible through slice) remain unchanged.
- **GC aliasing hazard.** Retained Arrow buffers must be kept reachable by the storage lease
  (not just by a raw span) so the GC cannot collect a managed backing array while the lease
  is still in use. This is the central invariant this rule protects.

## 6. Explicit statement on Phase A

This document is a design note only. It does not implement, modify, or land Phase A of
`docs/plan/ARROW-ROADMAP.md` (chunked columns / chunk-wise compute / chunk-aware
import/export), nor Phase D (import half), nor any memory-backed storage. It records the
ownership rule any such implementation must satisfy. Today's Arrow import remains materialising
and copying, exactly as before.