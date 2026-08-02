# ARROW-ROADMAP — the road to Arrow-inspired columnar physics in .NET

**Status:** planning reference · **Scope:** the columnar engine (`src/Nivara`) + Arrow/Parquet interop (`src/Nivara.Extensions/IO/`) · **Lens:** `docs/ARROW-REVIEW.md`

This is the travel plan for giving Nivara the *physics* of Arrow-inspired columnar analytics — the memory model, layouts, and sharing mechanics — translated to a native-.NET substrate. It complements `docs/POLARS-ROADMAP.md` (the query/expression engine) and together they close out the two lenses in `docs/POLARS-REVIEW.md` and `docs/ARROW-REVIEW.md`.

---

## 0. Vision

> **Arrow semantics. .NET physics. Shared buffers, not copied ones.**

Concretely, when this roadmap is done:

- A column has an explicit **physical layout** (`Flat<T>`, `VariableBinary`, `Dictionary`), decoupled from its logical type; strings are offset+data buffers, not reference arrays.
- **Validity is a first-class bitmap** with a documented representation and a cheap bool↔bitmap conversion at the boundaries.
- **Zero-copy actually works** — internally (slices/views share buffers) and at the interop boundary (Arrow arrays and tensors are built from existing `Memory<T>` buffers, exposed through real APIs — the placeholder `UseZeroCopy` option is gone, not kept lying).
- **Columns are chunked** — a frame is chunk-aligned columns, so append/concat are cheap and streaming/record-batch boundaries are natural.
- Arrow IPC and Parquet are the **zero-copy handoff** at the edges, not a conversion ritual.

### Where we are today

The semantics are right: explicit null masks, mask-OR kernels, immutable columns, Arrow/Parquet positioned as interchange. The physics are not: validity is byte-wise `bool` (`src/Nivara/Storage/TensorStorage.cs:14`, `MemoryStorage.cs:13`), internal span access flattens to a copy (`TensorStorage.cs:207-210`), interop has **no zero-copy path** (the placeholder `UseZeroCopy` option and `TryCreateZeroCopy*Array` methods were removed by the claims-integrity triage, `docs/TASKS-IMMEDIATELY.md`, so all conversion copies), and there is no chunked-column model in core (Arrow chunked arrays are flattened on import, `ArrowInterop.cs:688-710`).

### Non-goals (explicit)

- **No Arrow-internal storage rewrite.** Internal representation stays `Tensor<T>`/`Memory<T>`; Arrow remains an interchange boundary (POLARS-REVIEW Pillar 7).
- **No custom ownership model.** Sharing is achieved with immutable buffers + GC + structural sharing; no refcounting, no manual memory management.
- **No change to public null semantics.** ADR-001 and mask-OR behavior are preserved throughout; only the *representation* of the mask changes.
- **No unsafe hand-rolled SIMD** in the validity path; use `System.Runtime.Intrinsics`/`TensorPrimitives` where worth it, scalar mask logic elsewhere.

---

## 1. The roadmap

### Phase A — Chunked column model in core *(structural foundation)*

**Motivation:** Every Arrow shape collapses at the boundary today because the frame model is single-contiguous-array. Chunked columns are the one structural change everything else hangs off: append/concat, streaming, record-batch boundaries, and zero-copy interop all become natural.

**Scope:**
- Introduce `Chunk<T>` (contiguous buffer + validity chunk) and `ChunkedColumn<T>` (immutable list of chunks) as the core column representation; `NivaraColumn<T>`/storage becomes a thin facade over it.
- Make `Slice`/`Concat`/`Append` chunk-level operations (zero-copy where chunk boundaries allow).
- Import Arrow `ChunkedArray`/`RecordBatch` **without flattening** — chunks map 1:1; export the reverse.
- Feeds `StreamingExecutionStrategy` and the async work in POLARS-ROADMAP Phase 4.

**Key files:** new `src/Nivara/Storage/` chunk types, `src/Nivara/NivaraColumn.cs`, `src/Nivara/NivaraFrame.cs`, `src/Nivara.Extensions/IO/ArrowInterop.cs`.

**Dependencies:** none (foundation for B–F).

**Acceptance criteria:**
- Arrow `Table` round-trips with chunk structure preserved (chunk counts match; no forced flatten).
- `Concat` of aligned-chunk inputs does not copy the shared chunks.
- Existing frame/column behavior unchanged at the API level; `dotnet test` green.

**Risks:** behavioral churn across a large public surface (`NivaraColumn.cs` is ~3k lines); mitigation: keep the facade, land chunking as an internal storage concern first (matches AGENTS.md "storage is an internal concern").

---

### Phase B — Layout separation + variable-binary strings

**Motivation:** Kernels should dispatch on *layout*, not logical type, and strings should be analytically friendly (contiguous bytes + offsets) rather than `Memory<string>` reference arrays.

**Scope:**
- Introduce `ColumnLayout` kind on storage (`Flat<T>`, `VariableBinary`, `Dictionary`, …) as an explicit, documented concept.
- Move string columns to a **variable-binary layout**: contiguous `byte` data buffer + `int` offset buffer, with a span/string projection layer so the public API (`string` values) is unchanged.
- Generic kernels dispatch on layout; `KernelSelector` gains a layout input.

**Key files:** new `src/Nivara/Storage/ColumnLayout.cs`, `MemoryStorage<T>` rework for strings, `src/Nivara/KernelSelector.cs`, `src/Nivara/NivaraColumn.cs`.

**Dependencies:** Phase A (chunks are the natural unit of a variable-binary buffer).

**Acceptance criteria:**
- String sort/hash/groupby produce identical results through the new layout.
- Benchmark shows cache-friendlier string kernels (sort/join/groupby) for large string columns.
- Public `NivaraColumn<string>` API unchanged.

**Risks:** string layout rework is invasive (indexers, null semantics for strings, interop); mitigation: implement behind the existing `IColumnStorage<T>` seam, keep the legacy `Memory<string>` path for non-analytic columns until benchmarks justify migration.

---

### Phase C — Validity as an explicit bitmap

**Motivation:** Byte-wise `bool` masks are 8× denser than Arrow's bit-packed validity and were never a documented design decision. Make the representation explicit and boundary-friendly.

**Scope:**
- Define the canonical validity representation: **bit-packed `Memory<byte>`** (Arrow-compatible) with SIMD-usable mask helpers, **or** a documented byte-wise `Span<bool>` fast path — pick one primary, keep the other as an explicit interop conversion.
- Provide cheap `bool[]`↔bitmap conversions used at the Arrow boundary (no element boxing).
- Preserve mask-OR semantics exactly; add property tests proving behavior parity across both representations.

**Key files:** `src/Nivara/Storage/` validity types + helpers, `src/Nivara/NivaraColumn.cs` null ops, `src/Nivara.Extensions/IO/ArrowInterop.cs`.

**Dependencies:** Phase A (validity chunked with data).

**Acceptance criteria:**
- Null-mask property tests (existing patterns in `tests/Nivara.Tests/`) pass unchanged under the new representation.
- Arrow validity bitmap import/export is buffer-based, not per-element.
- No NaN/sentinel semantics introduced (ADR-001 intact).

**Risks:** bitwise mask access is slower than `Span<bool>` for scalar paths; mitigation: SIMD helpers + the documented byte-wise fast path; never regress null-correctness.

---

### Phase D — Real zero-copy, both directions *(credibility win)*

**Motivation:** Every interop path copies today — the placeholder `UseZeroCopy` option was removed rather than kept lying (claims-integrity triage, `docs/TASKS-IMMEDIATELY.md` Tasks 1–2, issue #94), and even internal span access flattens. This phase adds the real zero-copy APIs back.

**Scope:**
- Re-introduce the zero-copy interop path (the placeholder `TryCreateZeroCopy*Array` methods were removed) with real `MemoryMarshal`/buffer-handoff implementations: build Apache.Arrow arrays from existing `Memory<T>` (data buffer + validity bitmap) instead of builders+`Append`.
- Make `TensorStorage.GetFlattenedSpan()` a true **view** where the layout permits (no per-access `FlattenTo` allocation); keep caching as an optimization, not a necessity.
- Expose zero-copy through a dedicated option (e.g. a re-added `UseZeroCopy`) that engages only when the layout is compatible, throws/fails loudly when it cannot, and **never silently copies while reporting zero-copy**.
- `MemoryMarshal.AsBytes` for unmanaged flat layouts ↔ Arrow buffers; `ReadOnlyMemory<T>` sharing into `Tensor<T>`/`TensorSpan<T>` where the BCL allows.

**Key files:** `src/Nivara.Extensions/IO/ArrowInterop.cs`, `src/Nivara.Extensions/IO/ArrowConversionOptions.cs` (option re-added here), `src/Nivara/Storage/TensorStorage.cs`, `src/Nivara/Tensors/TensorInteropExtensions.cs`.

**Dependencies:** Phases A–C (zero-copy needs chunked, layout-explicit, bitmap-validity columns).

**Acceptance criteria:**
- Interop benchmark shows zero-copy paths actually share buffers (no element-loop copies) for flat, null-free and bitmapped cases.
- Round-trip tests confirm shared-buffer reads are consistent (immutability guarantees no mutation).
- An explicit zero-copy request on an incompatible layout fails loudly (clear exception) rather than silently copying.

**Risks:** Apache.Arrow's buffer APIs vs. `Tensor<T>` internal layout (nint dims, padding); memory-lifecycle safety of handing managed `Memory<T>` to native-bound arrays. Mitigation: pin/`MemoryManager<T>` ownership, keep owned-buffer semantics explicit (AGENTS.md zero-copy notes already flag this).

---

### Phase E — Dictionary encoding

**Motivation:** Low-cardinality columns dominate analytics; dictionary encoding pays off in every hashing operation (groupby, join, sort, distinct). It is the concrete payoff of the layout model from Phase B.

**Scope:**
- Add `ColumnLayout.Dictionary`: `ReadOnlyMemory<T> keys` + `ReadOnlyMemory<int> indices`, plus builder + decoder.
- Expose opt-in dictionary-encoded storage (per column/option) and make groupby/join/sort consume the indices buffer.
- Round-trip through Arrow `DictionaryArray` without full decode.

**Key files:** new `src/Nivara/Storage/` dictionary layout + builder, `src/Nivara/Operations/` (GroupBy/Join/Sort), `src/Nivara.Extensions/IO/ArrowInterop.cs`.

**Dependencies:** Phases B–C (layout model + validity bitmap).

**Acceptance criteria:**
- Groupby/join/sort over a dictionary-encoded column produce identical results and benchmark faster than decoded (per `KernelSelector` heuristics).
- Arrow `DictionaryArray` ↔ Nivara round-trip preserves the encoding.
- Null semantics preserved through encode/decode.

**Risks:** mutability semantics of indices under filters/slices; cardinality-spillover (dict becomes costlier than flat at high cardinality) — must measure and pick the right layout automatically.

---

### Phase F — Interchange-native scanning

**Motivation:** Lazy scans should yield record-batch-shaped chunks and stream through Arrow/Parquet as a buffer handoff, not a conversion ritual — converging with POLARS-ROADMAP Phase 4 (async streaming).

**Scope:**
- Lazy sources (`CsvDataSource`, `JsonDataSource`, Parquet) emit `ChunkedColumn`-shaped record batches directly.
- Async IPC/Parquet scanning via `IAsyncEnumerable` chunk streams with the existing memory budget (`StreamingBufferManager`).
- `CollectAsync`/streaming entry points consume chunks end to end.

**Key files:** `src/Nivara/IO/` lazy sources, `src/Nivara/Execution/StreamingExecutionStrategy.cs`, `src/Nivara/Query/IQueryInterfaces.cs` (async seams already exist), `src/Nivara.Extensions/IO/ParquetReader.cs`, `ParquetWriter.cs`.

**Dependencies:** Phases A + D; converges with POLARS-ROADMAP Phase 4.

**Acceptance criteria:**
- Streaming scan == eager results (property tests over chunk sizes); bounded memory under the configured budget.
- Parquet/CSV/JSON lazy sources materialize chunked columns without an intermediate whole-column array.
- Cancellation respected end to end.

**Risks:** fused expression operators (POLARS-ROADMAP Phase 2) must consume chunk-at-a-time; coordinate the two roadmaps' operator work to stay chunk-safe.

---

### Post-roadmap (aspirational)

- **Nested layouts** — `List`, `Struct`, `Map` as composable layouts once flat + variable-binary + dictionary exist (Arrow-inspired, optional).
- **C Data Interface / Flight** — if native interop demand appears, add via `LibraryImport` at the boundary (POLARS-REVIEW Pillar 9).

---

## 2. Sequencing rationale

**Why chunks first:** Phase A is the structural keystone — B (layout), C (bitmap validity), and D (zero-copy) all need a chunked, boundary-shaped column model to be natural. Without it, every other phase is bolted onto the wrong shape.

**Credibility vs. polish:** Phase D (real zero-copy) is the user-visible credibility win — it re-introduces the zero-copy API the claims-integrity triage removed (issue #94). Phases B and C are the physics that make D cheap and correct. Phase E is the analytical payoff; F is the streaming payoff.

**Cross-roadmap convergence:**

| ARROW phase | POLARS phase | Why they touch |
| --- | --- | --- |
| A · Chunked columns | 4 · Async streaming | Streaming operates on chunk-shaped data |
| D · Zero-copy | 2 · Kernel fusion | Both need span/view-first columns |
| F · Interchange scanning | 4 · Async streaming | Same async chunk pipeline |
| C · Bitmap validity | 2 · Generic-math collapse | Same null-aware kernel machinery |

Both roadmaps sit on the same foundation (typed, null-aware, span-first columns). Phase A here should be sequenced against POLARS-ROADMAP Phase 1 (typed expression engine): they are independent — expression typing does not need chunks, and chunks do not need expression typing — but both must land before Phase D/F merge points.

**What we leverage, not reinvent:** the existing storage seam (`IColumnStorage<T>`, `TensorStorage`/`MemoryStorage`), the async members already on `IQuerySource` (`ReadChunkAsync`, `ToAsyncEnumerable` in `src/Nivara/Query/IQueryInterfaces.cs:34-52`), `MemoryStorage.Slice` (already a true `ReadOnlyMemory` view — proof the pattern works), `KernelSelector`, and the Arrow/Parquet interop test suite in `tests/Nivara.Tests/IO/`.

---

## 3. Cross-cutting conventions (every phase)

- **Diagnostics:** record kernel/layout/zero-copy decisions via `OperationDiagnostics` / `ExecutionEngine.LastDiagnostics`; a zero-copy fallback must be observable.
- **Testing:** NUnit 4.x, `Method_Scenario_ExpectedBehavior` naming, property tests for null-mask parity across representations and chunk boundaries, round-trip tests for Arrow/Parquet with chunk structure preserved.
- **Null semantics:** ADR-001 holds; only the representation changes. Comparisons remain false-at-null; arithmetic remains mask-OR.
- **No comments in code** beyond non-obvious decisions; `.editorconfig` is authoritative.
- **Allocation discipline:** rented/pooled buffers (`ArrayPool`/`BufferPool`) for scratch; no per-element `object?` in the hot paths (ties to POLARS-REVIEW §4).

---

## 4. Definition of done

The vision at §0 holds, verified by:

- Core columns are chunked with explicit layouts; strings use variable-binary buffers.
- Validity has one canonical representation with parity tests and boundary conversion.
- Zero-copy engages where the layout allows and fails loudly where it cannot; the dedicated zero-copy option (re-added by Phase D) is honest.
- Dictionary encoding exists for low-cardinality columns and accelerates groupby/join/sort.
- Lazy scanning is record-batch-shaped and async-capable.
- Full `dotnet build Nivara.slnx` + `dotnet test` green; no silently-copying zero-copy paths remain.

---

## Related documents

- `docs/ARROW-REVIEW.md` — the lens, the eight pillars, and the scorecard this roadmap closes out.
- `docs/POLARS-REVIEW.md` — the query-engine lens (the boxed expression evaluator finding).
- `docs/POLARS-ROADMAP.md` — the query/expression roadmap this roadmap converges with.
- `docs/IDEA.md` — original product vision (storage abstraction, Arrow interop sections).
- `docs/TENSORS.md` — tensor-vs-Polars strategic framing.
- `docs/adr/001-autodiff-nonnullable-domain.md` — the explicit-null boundary all validity work must respect.
