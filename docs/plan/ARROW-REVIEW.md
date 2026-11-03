# If Arrow-inspired columnar analytics originated in .NET — the lens and the review

**Status:** architecture reference · **Scope:** the columnar engine (`src/Nivara`) + Arrow/Parquet interop (`src/Nivara.Extensions/IO/`) · **Companion:** `docs/plan/ARROW-ROADMAP.md`, `docs/plan/POLARS-REVIEW.md`

This document evaluates Nivara through a second lens: **"What would Arrow-inspired columnar analytics look like if it had been conceived in .NET?"** It is deliberately *not* "how closely do we match Apache Arrow." Arrow is a format standard; this lens is about the *physics* of columnar analytics — the memory model, the layout rules, the sharing and streaming mechanics — that Arrow made popular, translated to the substrate .NET actually provides.

The Polars lens (`docs/plan/POLARS-REVIEW.md`) asked whether Nivara behaves like a *query engine* born in .NET. This lens asks whether Nivara behaves like a *columnar memory system* born in .NET. The verdict is sharper than the Polars one: **Nivara adopted Arrow's semantics (columnar processing, explicit nulls, immutability, mask-OR kernels) but skipped its physics (bit-packed validity, layout separation, shareable buffers, chunked columns) — and the one place it reached for the physics, interop zero-copy, is left as commented-out scaffolding.**

> **Reconciled 2026-08-02 (issue #94):** the `ArrowConversionOptions.UseZeroCopy` option and `TryCreateZeroCopy*Array` placeholders cited below were **removed** by the claims-integrity triage (recorded in `CHANGELOG.md`; Tasks 1–2 of the triage). §4.1's finding stands — the code could not deliver zero-copy at the time — and the remedy changed from "make the option honest" to "add real zero-copy APIs when ready" (`docs/plan/ARROW-ROADMAP.md` Phase D). At that point no public API advertised zero-copy; that ceased to be true once the export half of Phase D landed (see the 2026-11-03 note below).

> **Reconciled 2026-11-03 (issues #835, #836):** real zero-copy **export** now exists behind `ArrowConversionOptions.UseZeroCopy` — flat primitive columns hand their data buffer to the Arrow array, null-bearing columns included, and ineligible layouts throw rather than copy. "No public API advertises zero-copy" therefore no longer holds for export; **import** and Parquet still copy, so Pillar 3 stays partial.

> **Superseded 2026-08-03 (storage consolidation, PR #110):** the storage-specific evidence in the §3 scorecard and §4 findings is **historical** — it describes the pre-consolidation `TensorStorage`/`MemoryStorage` split. Post-consolidation there is a single sole-owner `ColumnStorage<T>` (`src/Nivara/Storage/ColumnStorage.cs`): span access is a genuine zero-copy view, and internal column→`Tensor<T>` views are real and **public** (`NivaraColumn<T>.AsTensorView()`/`NivaraSeries<T>.AsTensorView()`, issue #107). The *Arrow/Parquet interchange* conclusion in §4.1 still holds — that boundary remains element-by-element copying until `docs/plan/ARROW-ROADMAP.md` Phase D. Pillar 2's validity-bitmap gap and Pillar 4's chunked-column gap are unchanged.

---

## 1. Why a physics lens

Arrow's transferable contribution is not its IPC bytes. It is a set of structural ideas about how columnar data should be *held* in memory:

- Data is organized as **buffers**, not arrays of objects.
- **Nullability is a separate buffer**, not a sentinel.
- **Logical types are decoupled from physical layouts**, so one kernel serves many types.
- Columns are **shareable and sliceable** without copying.
- Data volume lives in **chunks**; tables are chunk-aligned.
- **Encoding** (dictionary, etc.) is a layout strategy, not a special case.

A .NET-native implementation of these ideas does not import Arrow's C++ memory manager or ownership model. It uses `ReadOnlyMemory<T>`, `Span<T>`, `MemoryPool`, `MemoryMarshal`, and immutability — the GC does Arrow's refcounting for free, and structural sharing replaces ownership-based zero-copy. That is the native translation. The pillars below are that translation.

---

## 2. The eight pillars of Arrow-inspired columnar analytics in .NET

### Pillar 1 — Logical/physical layout separation

- **Principle:** a column is `(logical field: name, type, nullability, metadata) × (physical layout: buffers)`. Kernels operate on layouts; logical types are metadata.
- **.NET substrate:** a `ColumnLayout` kind (`Flat<T>`, `VariableBinary`, `Dictionary`, …) with `ReadOnlyMemory<T>` buffers; generic kernels dispatch on the layout.
- **Critical consequence — strings:** Arrow stores string data as a **variable-binary layout** — one contiguous byte buffer plus an offset buffer (indices into the data buffer). Sorting, hashing, joining, and grouping over strings are then cache-friendly and vectorizable. Storing `Memory<string>` (an array of heap references) is the *reference-type* layout — idiomatic .NET, but analytically weaker.

### Pillar 2 — Validity as a first-class bitmap

- **Principle:** nulls are a separate buffer, **bit-packed** (1 bit per element), decoupled from the data buffer. Null propagation in kernels is a bitmap AND/OR.
- **.NET substrate:** `Memory<byte>` validity bitmap with SIMD-friendly mask helpers — or a *deliberate, documented* byte-wise `Span<bool>` fast path — plus a cheap bool↔bitmap conversion at the interchange boundary.
- **Ratified (2026-11-02, issue #832):** Nivara keeps the `bool[]`/`Tensor<bool>` mask — 1 byte per element, **8× the memory** of Arrow's bitmap — as a *deliberate, documented* fast path, because every null-aware kernel consumes `ReadOnlySpan<bool>` and a bit-packed primary form would charge pack/unpack to all of them. Recorded as `docs/adr/007-byte-wise-null-mask-fast-path.md`. Bit-packing is a boundary conversion; the span-first `ArrowValidityMask` helper now exists (issue #833) and is wired into `ArrowInterop` for flat primitive layouts, sliced arrays and the series path (issues #840, #843). Packing remains element-wise only for variable-binary (`string`/`char`) layouts, where per-value conversion is unavoidable.

### Pillar 3 — Buffers, views, and true zero-copy

- **Principle:** columns are immutable buffers; slice/concat/append are **buffer views**; interop is a **buffer handoff**. The entire point of the columnar memory model is that data does not move.
- **.NET substrate:** `ReadOnlyMemory<T>` as the column currency, `MemoryPool`/`ArrayPool` for scratch, `MemoryMarshal` for bytes↔typed, immutability + structural sharing in place of refcounting.

### Pillar 4 — Chunked columns / record batches

- **Principle:** a column is a **list of contiguous chunks**; a frame is a set of chunk-aligned columns. This is the abstraction that makes append, streaming, columnar parallelism, and IPC batch boundaries natural instead of special cases.
- **.NET substrate:** a `ChunkedColumn<T>` (array of contiguous `Chunk<T>`) in core; NivaraFrame holds chunk-aligned columns.

### Pillar 5 — Schema as an immutable, typed, first-class artifact

- **Principle:** schema = ordered name/type/nullability/metadata, immutable, shared by reference, capable of structural compatibility checks.
- **.NET substrate:** an immutable `Schema` type + source-generated typed accessors (see POLARS-ROADMAP Phase 5).

### Pillar 6 — Null-propagating compute kernels over layouts

- **Principle:** kernels consume `(data span, validity)` and produce `(result span, validity)`; result nulls = OR of input validities; comparisons place `false` at null positions (SQL-like).
- **.NET substrate:** generic `INumber<T>` / `IFloatingPointIeee754<T>` kernels over `Span<T>` with `TensorPrimitives` on the vectorizable path.

### Pillar 7 — Encoding as a layout strategy

- **Principle:** low-cardinality data is *dictionary-encoded* (a keys buffer + an indices buffer) as a first-class layout — because it pays off in every hashing operation (groupby, join, sort, distinct).
- **.NET substrate:** a `Dictionary` layout with `ReadOnlyMemory<T> keys` + `ReadOnlyMemory<int> indices`.

### Pillar 8 — Interchange-native scanning

- **Principle:** lazy scans yield **record-batch-shaped chunks**; Arrow IPC and Parquet are the zero-copy handoff at the edges, not a conversion ritual.
- **.NET substrate:** `IAsyncEnumerable` chunk streams, `MemoryMarshal` buffer handoff into Arrow arrays/tensors.

---

## 3. Nivara scorecard against the Arrow lens

| Pillar | Status | Evidence |
| --- | --- | --- |
| 1 · Logical/physical separation | 🟡 Partial | Storage is consolidated into one sole-owner `ColumnStorage<T>` — the pre-consolidation `TensorStorage`/`MemoryStorage` split no longer exists — but there is still no *layout* abstraction, and strings are `T[]` reference arrays (`src/Nivara/Storage/ColumnStorage.cs::data`), not variable-binary: an analytic gap for sort/hash/groupby on strings |
| 2 · Validity bitmap | 🟡 Divergent, and now decided | Nulls are a `bool[]` mask — **1 byte per element**, not bit-packed (`src/Nivara/Storage/ColumnStorage.cs::nullMask`); byte-wise is ratified as the deliberate fast path (ADR-007). The buffer-level `ArrowValidityMask` helper (issue #833) is wired into `ArrowInterop` for flat primitives, sliced arrays and the series path (issues #840, #843); packing remains element-wise for variable-binary (`string`/`char`) layouts. |
| 3 · Buffers + zero-copy | 🟡 Partial | *Internally* access is a genuine zero-copy view — `AsSpan()` returns a slice of the owned array (`src/Nivara/Storage/ColumnStorage.cs::AsSpan()`) and `Slice` shares the same backing array (`src/Nivara/Storage/ColumnStorage.cs::Slice(int, int)`). *Export* to Arrow is now genuinely zero-copy for flat primitive columns through `ArrowConversionOptions.UseZeroCopy` — values buffer shared, validity bitmap rebuilt — with aliasing, slice, lifetime and IPC tests (issues #835, #836). *Import* still materialises element by element (`src/Nivara.Extensions/IO/ArrowInterop.cs::FromArrowTable`), and Parquet copies both ways |
| 4 · Chunked columns / RecordBatch | ❌ Absent | No chunked column in core; Arrow `ChunkedArray` is *flattened* on import into single arrays (`src/Nivara.Extensions/IO/ArrowInterop.cs::ConvertArrowColumnToNivaraColumn`), losing chunk structure |
| 5 · Schema artifact | ✅ Strong | Immutable `Schema` with metadata + compatibility checks (`src/Nivara/Schema.cs::Schema`); source-generated typed accessors missing |
| 6 · Null-propagating kernels | ✅ Strong | Column kernels are null-aware, mask-OR, vectorized (`TensorPrimitives` in `src/Nivara/NivaraColumn.cs`) — *except* the boxed query expression path (see `docs/plan/POLARS-REVIEW.md` §4) |
| 7 · Dictionary encoding | ❌ Absent | Listed only as "future" in `docs/IDEA.md` (retired); tracked as [issue #132](https://github.com/khurram-uworx/Nivara/issues/132) |
| 8 · Interchange-native scanning | 🟡 Partial | Lazy sources + streaming strategies exist (`StreamingExecutionStrategy`); pull-based, not async-first (see `docs/plan/POLARS-ROADMAP.md` Phase 4) |

Legend: ✅ native-aligned · 🟡 partially aligned · ❌ gap.

---

## 4. The two things that are actually wrong

### 4.1 "Zero-copy" was a placeholder — removed, then really implemented for export

The design promised Arrow-style zero-copy sharing, but the code only had scaffolding. **Resolution (claims-integrity triage, recorded in `CHANGELOG.md`, issue #94):** the `ArrowConversionOptions.UseZeroCopy` option and the `TryCreateZeroCopy*Array` placeholders were **removed** as a breaking change rather than left lying. The real implementation landed later as `docs/plan/ARROW-ROADMAP.md` Phase D.

- **Export is now genuinely zero-copy for flat primitive layouts (issues #835, #836).** With `UseZeroCopy` enabled, `ToArrowTable`/`ToArrowArray` build the Arrow data buffer over the column's own array through a `MemoryManager<byte>`; there is no element loop. Null-bearing columns share the values buffer and rebuild the bit-packed validity bitmap from the byte-wise null mask — the one buffer that cannot be shared, because the two representations differ in packing *and* polarity. Ineligible layouts throw `UnsupportedTypeException` naming the column instead of silently copying, and the option is export-only.
- **Import still materialises.** `FromArrowTable` walks each value through a typed conversion into a fresh nullable array (`src/Nivara.Extensions/IO/ArrowInterop.cs::FromArrowTable`), and Parquet copies in both directions. The import half of Phase D is open.
- Historically even *internal* span access copied: `TensorStorage.GetFlattenedSpan()` flattened into a freshly allocated `T[]`, and `TensorStorage.Slice` copied. Both members are gone. Today `AsSpan()` hands back a slice of the owned array (`src/Nivara/Storage/ColumnStorage.cs::AsSpan()`) and `Slice` re-uses that same array with an offset (`src/Nivara/Storage/ColumnStorage.cs::Slice(int, int)`) — so this half of the old finding is **resolved**.
- The plumbing is no longer half-present. Zero-copy used to work only on the memory-storage path, with the tensor path — supposedly the fast path — copying; today there is a single path and it shares (`src/Nivara/Storage/ColumnStorage.cs::Slice(int, int)`). What remains is zero-copy across the Arrow/Parquet **import** boundary, and no amount of internal sharing substitutes for that.

**Consequence:** the core Arrow promise — share buffers, don't copy — now holds in the export direction and is advertised as such; it is still missing on import and in Parquet. `docs/plan/ARROW-ROADMAP.md` Phase D records both halves.

### 4.2 No chunked model, so the Arrow shapes collapse at the boundary

Apache Arrow `Table` is a set of chunk-aligned columns; NivaraFrame is a set of single contiguous arrays. As a result:

- Every Arrow `ChunkedArray` is flattened on import into one Nivara column (`src/Nivara.Extensions/IO/ArrowInterop.cs::ConvertArrowColumnToNivaraColumn`), discarding chunk structure.
- Every `Concat`/`Append` is a full materialization; there is no zero-copy horizontal or vertical sharing.
- Streaming materializes whole columns per operation rather than pipeline-shaped chunks.

The chunked-column abstraction is the single structural piece that would make Arrow-inspired analytics feel native, and it is absent.

---

## 5. What is already right (and should not be changed)

- **The semantics are Arrow-faithful.** Explicit null masks, no NaN-based nulls, mask-OR propagation, comparisons false-at-null, SQL-like semantics — all match Arrow's compute-kernel model and ADR-001's explicit-null boundary (`docs/adr/001-autodiff-nonnullable-domain.md`).
- **Schema is genuinely Arrow-inspired and native.** Immutable, metadata-carrying, compatibility-checked (`src/Nivara/Schema.cs::Schema`).
- **Column kernels are null-aware and vectorized.** `TensorPrimitives` on spans, `KernelSelector` heuristics — the compute layer is in the right shape.
- **Arrow and Parquet are correctly positioned as interchange** in `Nivara.Extensions`, not core — the native-.NET boundary decision from POLARS-REVIEW Pillar 7 holds.
- **The single storage class is the right skeleton** — one sole-owner `ColumnStorage<T>` for every element type is the beginning of a layout model (Pillar 1); what it needs is the layout concept made explicit, not a second storage implementation.

---

## 6. Summary

The one-line verdict:

> Nivara adopted Arrow's **semantics** — columnar processing, explicit null masks, immutability, mask-OR kernels — but skipped its **physics**: bit-packed validity, logical/physical layout separation, genuinely shareable buffers, and chunked columns. The one place it reached for the physics (interop zero-copy) was scaffolding that silently copied; it is now real and tested for Arrow **export** and still missing on import and in Parquet (issues #835, #836).

The roadmap to close that gap is in **`docs/plan/ARROW-ROADMAP.md`**.

## Related documents

- `docs/plan/ARROW-ROADMAP.md` — the roadmap to close that gap.
- `CHANGELOG.md` — records both the claims-integrity removal of the `UseZeroCopy` placeholder (issue #94) and the real zero-copy export that replaced it (issues #835, #836).
- `docs/plan/POLARS-REVIEW.md` / `docs/plan/POLARS-ROADMAP.md` — the query-engine lens and roadmap.
- `docs/plan/AISTACK-REVIEW.md` / `docs/plan/AISTACK-ROADMAP.md` — the third lens: complementing the local .NET AI stack, which consumes the chunked/zero-copy foundation this roadmap builds.
- `docs/IDEA.md` (retired 2026-08-06; its storage/Arrow-adjacent items are tracked as GitHub issues: [dictionary encoding #132](https://github.com/khurram-uworx/Nivara/issues/132), [custom storage plug-in seam #133](https://github.com/khurram-uworx/Nivara/issues/133), [GPU offload #135](https://github.com/khurram-uworx/Nivara/issues/135)), `docs/TENSORS.md`, `docs/adr/001-autodiff-nonnullable-domain.md` — product vision, strategic framing, null-boundary rule.
