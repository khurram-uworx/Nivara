# If Arrow-inspired columnar analytics originated in .NET — the lens and the review

**Status:** architecture reference · **Scope:** the columnar engine (`src/Nivara`) + Arrow/Parquet interop (`src/Nivara.Extensions/IO/`) · **Companion:** `docs/ARROW-ROADMAP.md`, `docs/POLARS-REVIEW.md`

This document evaluates Nivara through a second lens: **"What would Arrow-inspired columnar analytics look like if it had been conceived in .NET?"** It is deliberately *not* "how closely do we match Apache Arrow." Arrow is a format standard; this lens is about the *physics* of columnar analytics — the memory model, the layout rules, the sharing and streaming mechanics — that Arrow made popular, translated to the substrate .NET actually provides.

The Polars lens (`docs/POLARS-REVIEW.md`) asked whether Nivara behaves like a *query engine* born in .NET. This lens asks whether Nivara behaves like a *columnar memory system* born in .NET. The verdict is sharper than the Polars one: **Nivara adopted Arrow's semantics (columnar processing, explicit nulls, immutability, mask-OR kernels) but skipped its physics (bit-packed validity, layout separation, shareable buffers, chunked columns) — and the one place it reached for the physics, interop zero-copy, is left as commented-out scaffolding.**

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
- **Divergence to flag:** `bool[]`/`Tensor<bool>` (1 byte per element) is 8× denser in Arrow terms and is *not* a layout decision Nivara has made — it is an inheritance.

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
| 1 · Logical/physical separation | 🟡 Partial | Storage split exists (`TensorStorage`/`MemoryStorage`), but no *layout* abstraction; strings are `Memory<string>` reference arrays (`src/Nivara/Storage/MemoryStorage.cs:12-13`), not variable-binary — an analytic gap for sort/hash/groupby on strings |
| 2 · Validity bitmap | 🟡 Divergent | Nulls are `ReadOnlyMemory<bool>` / `Tensor<bool>` — **1 byte per element**, not bit-packed (`TensorStorage.cs:14`, `MemoryStorage.cs:13`); no documented layout decision, no bool↔bitmap path |
| 3 · Buffers + zero-copy | ❌ **Scaffolding only** | `GetFlattenedSpan()` allocates `new T[...]` + `FlattenTo` — a **copy**, cached, not a view (`src/Nivara/Storage/TensorStorage.cs:207-210`); construction copies (`TensorStorage.cs:33`); interop zero-copy methods always `return null` (`src/Nivara.Extensions/IO/ArrowInterop.cs:977-1065`) |
| 4 · Chunked columns / RecordBatch | ❌ Absent | No chunked column in core; Arrow `ChunkedArray` is *flattened* on import into single arrays (`ArrowInterop.cs:688-710`), losing chunk structure |
| 5 · Schema artifact | ✅ Strong | Immutable `Schema` with metadata + compatibility checks (`src/Nivara/Schema.cs:9-300`); source-generated typed accessors missing |
| 6 · Null-propagating kernels | ✅ Strong | Column kernels are null-aware, mask-OR, vectorized (`TensorPrimitives` in `src/Nivara/NivaraColumn.cs`) — *except* the boxed query expression path (see `docs/POLARS-REVIEW.md` §4) |
| 7 · Dictionary encoding | ❌ Absent | Listed only as "future" in `docs/IDEA.md` |
| 8 · Interchange-native scanning | 🟡 Partial | Lazy sources + streaming strategies exist (`StreamingExecutionStrategy`); pull-based, not async-first (see `docs/POLARS-ROADMAP.md` Phase 4) |

Legend: ✅ native-aligned · 🟡 partially aligned · ❌ gap.

---

## 4. The two things that are actually wrong

### 4.1 "Zero-copy" is a placeholder — including internally

The design promises Arrow-style zero-copy sharing, and the code has the scaffolding — but nothing engages:

- `ArrowConversionOptions.UseZeroCopy = true` is the **default** (`src/Nivara.Extensions/IO/ArrowConversionOptions.cs:18`), yet every `TryCreateZeroCopy*Array` method is a commented placeholder that unconditionally returns `null` (`src/Nivara.Extensions/IO/ArrowInterop.cs:977-1065`). The option is effectively a lie: all interop falls back to element-by-element copying through `List<object?>` and `GetValue` boxing (`ArrowInterop.cs:677-710`, `731-806`, `824-849`).
- Even *internal* span access is a copy: `TensorStorage.GetFlattenedSpan()` flattens into a freshly allocated `T[]` (`src/Nivara/Storage/TensorStorage.cs:207-210`), and `TensorStorage.Slice` copies (`TensorStorage.cs:145-156`). The "zero-copy tensor path" is not zero-copy.
- The one place zero-copy genuinely works is the memory storage path: `ReadOnlyMemory.Slice` shares the underlying array (`src/Nivara/Storage/MemoryStorage.cs:130-139`). So the plumbing is half-present — the tensor path (which is supposed to be the fast path) is the one that copies.

**Consequence:** the core Arrow promise — share buffers, don't copy — is the single capability the project currently cannot deliver, and it is promised in an option whose default is `true`.

### 4.2 No chunked model, so the Arrow shapes collapse at the boundary

Apache Arrow `Table` is a set of chunk-aligned columns; NivaraFrame is a set of single contiguous arrays. As a result:

- Every Arrow `ChunkedArray` is flattened on import into one Nivara column (`ArrowInterop.cs:688-710`, `731-806`), discarding chunk structure.
- Every `Concat`/`Append` is a full materialization; there is no zero-copy horizontal or vertical sharing.
- Streaming materializes whole columns per operation rather than pipeline-shaped chunks.

The chunked-column abstraction is the single structural piece that would make Arrow-inspired analytics feel native, and it is absent.

---

## 5. What is already right (and should not be changed)

- **The semantics are Arrow-faithful.** Explicit null masks, no NaN-based nulls, mask-OR propagation, comparisons false-at-null, SQL-like semantics — all match Arrow's compute-kernel model and ADR-001's explicit-null boundary (`docs/adr/001-autodiff-nonnullable-domain.md`).
- **Schema is genuinely Arrow-inspired and native.** Immutable, metadata-carrying, compatibility-checked (`src/Nivara/Schema.cs:9-300`).
- **Column kernels are null-aware and vectorized.** `TensorPrimitives` on spans, `KernelSelector` heuristics — the compute layer is in the right shape.
- **Arrow and Parquet are correctly positioned as interchange** in `Nivara.Extensions`, not core — the native-.NET boundary decision from POLARS-REVIEW Pillar 7 holds.
- **The storage split (`Tensor`/`Memory`) is the right skeleton** — it is the beginning of a layout model (Pillar 1), it just needs the layout concept made explicit.

---

## 6. Summary

The one-line verdict:

> Nivara adopted Arrow's **semantics** — columnar processing, explicit null masks, immutability, mask-OR kernels — but skipped its **physics**: bit-packed validity, logical/physical layout separation, genuinely shareable buffers, and chunked columns. The one place it reached for the physics (interop zero-copy) is scaffolding that silently copies.

The roadmap to close that gap is in **`docs/ARROW-ROADMAP.md`**.

## Related documents

- `docs/ARROW-ROADMAP.md` — the roadmap to close that gap.
- `docs/POLARS-REVIEW.md` / `docs/POLARS-ROADMAP.md` — the query-engine lens and roadmap.
- `docs/AISTACK-REVIEW.md` / `docs/AISTACK-ROADMAP.md` — the third lens: complementing the local .NET AI stack, which consumes the chunked/zero-copy foundation this roadmap builds.
- `docs/IDEA.md`, `docs/TENSORS.md`, `docs/adr/001-autodiff-nonnullable-domain.md` — product vision, strategic framing, null-boundary rule.
