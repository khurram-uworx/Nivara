# ARROW-ROADMAP — the road to Arrow-inspired columnar physics in .NET

**Status:** planning reference · **Scope:** the columnar engine (`src/Nivara`) + Arrow/Parquet interop (`src/Nivara.Extensions/IO/`) · **Lens:** `docs/plan/ARROW-REVIEW.md`

This is the travel plan for giving Nivara the *physics* of Arrow-inspired columnar analytics — the memory model, layouts, and sharing mechanics — translated to a native-.NET substrate. It complements `docs/plan/POLARS-ROADMAP.md` (the query/expression engine) and together they close out the two lenses in `docs/plan/POLARS-REVIEW.md` and `docs/plan/ARROW-REVIEW.md`.

---

## 0. Vision

> **Arrow semantics. .NET physics. Shared buffers, not copied ones.**

Concretely, when this roadmap is done:

- A column has an explicit **physical layout** (`Flat<T>`, `VariableBinary`, `Dictionary`), decoupled from its logical type; strings are offset+data buffers, not reference arrays.
- **Validity is a first-class bitmap** with a documented representation and a cheap bool↔bitmap conversion at the boundaries.
- **Zero-copy actually works** — internally (slices/views share buffers) and at the interop boundary, where the opt-in `ArrowConversionOptions.UseZeroCopy` hands a flat primitive column's data buffer straight to the Arrow array (Phase D, export only so far).
- **Columns are chunked** — a frame is chunk-aligned columns, so append/concat are cheap and streaming/record-batch boundaries are natural.
- Arrow IPC and Parquet are the **zero-copy handoff** at the edges, not a conversion ritual.

### Where we are today

The semantics are right: explicit null masks, mask-OR kernels, immutable columns, Arrow/Parquet positioned as interchange. The physics are partly there: validity is byte-wise `bool` (`src/Nivara/Storage/ColumnStorage.cs`); Arrow **export** shares the values buffer of a flat primitive column through the opt-in `UseZeroCopy` option (`ArrowInterop.cs::CreateZeroCopyArrowArray`), while **import** and Parquet conversion still copy; and there is no chunked-column model in core (Arrow chunked arrays are flattened on import, `ArrowInterop.cs::ConvertArrowColumnToNivaraColumnInt`). Internal column→tensor access is now a genuine zero-copy view (`NivaraColumn<T>.AsTensorView()`/`NivaraSeries<T>.AsTensorView()`, backed by `ColumnStorage<T>.AsTensor()`); that is a Nivara-internal capability and does not carry over to the Arrow/Parquet boundary.

### Non-goals (explicit)

- **No Arrow-internal storage rewrite.** Internal representation stays sole-owner `ColumnStorage<T>` (`T[]` + optional `bool[]` null mask, single contiguous buffer); Arrow remains an interchange boundary (POLARS-REVIEW Pillar 7).
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

**Key files:** new `src/Nivara/Storage/ColumnLayout.cs`, `ColumnStorage<T>` layout work for strings, `src/Nivara/KernelSelector.cs`, `src/Nivara/NivaraColumn.cs`.

**Dependencies:** Phase A (chunks are the natural unit of a variable-binary buffer).

**Acceptance criteria:**
- String sort/hash/groupby produce identical results through the new layout.
- Benchmark shows cache-friendlier string kernels (sort/join/groupby) for large string columns.
- Public `NivaraColumn<string>` API unchanged.

**Risks:** string layout rework is invasive (indexers, null semantics for strings, interop); mitigation: implement behind the existing `IColumnStorage<T>` seam, keep the legacy `Memory<string>` path for non-analytic columns until benchmarks justify migration.

---

### Phase C — Validity representation decided; boundary conversion remains

**Motivation:** Byte-wise `bool` masks cost **8× the memory** of Arrow's bit-packed validity (1 byte
per element against 1 bit), and that cost was never a written decision — so the representation read as
accidental. Phase C exists to make it explicit and boundary-friendly. The first half of that is now
done: the representation is ratified, and what remains is the conversion.

**Decision (2026-11-02, issue #832) — Option A:** byte-wise `ReadOnlySpan<bool>` is the canonical
validity representation in core; bit-packed `Memory<byte>` exists **only** at the Arrow boundary.
Core storage does not change. The rationale is that every null-aware kernel already consumes
`ReadOnlySpan<bool>` — including the fused pass that ORs leaf masks into an output mask — so a
bit-packed primary form would charge unpack-on-entry and pack-on-exit to every kernel in order to
save mask memory that is small beside the payload. The cost (8× on the mask, and only when nulls
exist) is accepted deliberately, with a stated revisit trigger: validity becoming a dominant memory
consumer, or a buffer-level handoff being needed by Phase D.

**What that means for the boundary:** Arrow's buffer is bit-packed, LSB-first, with a set bit meaning
*valid*, while Nivara's mask is byte-wise and true-at-null. Conversion therefore **packs and inverts
polarity**, and it belongs solely to interop. Both sides also allow an elided validity buffer when
there are no nulls; the boundary reconciles the two conventions.

**Scope (remaining):**
- Provide cheap `bool[]`↔bitmap conversions used at the Arrow boundary (no element boxing). *Landed
  (issue #833):* the span-first `ArrowValidityMask` helper packs and unpacks over spans, inverts
  polarity, zeroes padding bits, and handles the absent-mask cases. *Landed (issue #840):* wired into
  `ArrowInterop` for flat primitive layouts, sliced arrays (offset-aware), and series import; the
  element-wise path remains only for variable-binary (`string`/`char`) layouts, documented in code.
- Preserve mask-OR semantics exactly; add property tests proving behavior parity across both
  representations.
- Keep core storage byte-wise — this phase does not move a bitmap into `ColumnStorage<T>`.

**Key files:** `src/Nivara.Extensions/IO/ArrowInterop.cs` (the boundary) and
`src/Nivara.Extensions/IO/ArrowValidityMask.cs` (the landed helper); core storage gains nothing, because
the conversion is not shared.

**Dependencies:** Phase A (validity chunked with data) for the chunk-level form of the conversion.

**Acceptance criteria:**
- *Decided:* the byte-wise mask is documented as deliberate and the Arrow boundary is identified as
  the only place packing happens — landed as ADR-007 and in `ARCHITECTURE.md`.
- Null-mask property tests (existing patterns in `tests/Nivara.Tests/`) pass unchanged; the
  representation did not change, so this is a no-op guard rather than new work.
- *Landed (issue #833):* a span-first bitmap↔mask helper exists, round-trips exactly, and covers the
  absent-mask cases; it is internal and wired into `ArrowInterop` for flat primitives (issue #840).
- *Landed (issue #843):* Arrow validity bitmap import/export is buffer-based, not per-element, for
  the flat primitive types — `ArrowValidityMask` is wired into `ArrowInterop` in both directions.
- *Landed (issue #840):* buffer-level conversion covers sliced arrays (offset-aware) and the
  `FromArrowArray` series path; no-null columns export with `ArrowBuffer.Empty`; variable-binary
  layouts remain element-wise by documented limitation.
- *Landed (issue #834):* round-trip validity tests at the interop boundary (empty / all-null /
  all-valid / mixed) for value, reference and bool columns, plus a hand-built dirty-padding-bits
  case; proves the helper's encoding matches the buffers Arrow produces and consumes.
- No NaN/sentinel semantics introduced (ADR-001 intact).

**Risks:** the remaining work touches the interop hot path in both directions; mitigation: land the
conversion behind the existing seam, keep the element-wise path until the buffer-level one is proven
by round-trip parity in the interop layer, and never regress null-correctness.

---

### Phase D — Real zero-copy, both directions *(credibility win; export half landed)*

**Motivation:** Every Arrow/Parquet interchange path copied, and the placeholder `UseZeroCopy` option was removed rather than kept lying (claims-integrity triage, recorded in `CHANGELOG.md`, issue #94). *Landed (issues #835, #836):* the export half is real — `ArrowConversionOptions.UseZeroCopy` shares a flat primitive column's data buffer with the Arrow array instead of copying, and is backed by aliasing, slice, lifetime, IPC and null-parity tests.

**Scope:**
- ✅ *Done (export, #835/#836):* `ArrowInterop.ToArrowTable`/`ToArrowArray` build Apache.Arrow arrays from the column's existing `Memory<T>` for flat `byte`/`sbyte`/`short`/`ushort`/`int`/`uint`/`long`/`ulong`/`float`/`double`/`Half` columns. Null-bearing columns share the values buffer and rebuild the validity bitmap (a byte-wise, true-at-null mask cannot alias a bit-packed, true-at-valid one). An ineligible layout throws `UnsupportedTypeException` naming the column rather than silently copying. Export-only: `FromArrowTable` ignores the option.
- *Remaining:* the **import** half — `FromArrowTable` still materialises columns element by element, and Parquet conversion copies in both directions.
- ✅ *Done as of the storage consolidation:* the internal column→`Tensor<T>` view is now real and public — `NivaraColumn<T>.AsTensorView()`/`NivaraSeries<T>.AsTensorView()` return `ColumnStorage<T>.AsTensor()`, a lazy zero-copy view (no per-access `FlattenTo` allocation, cached).
- Extend the dedicated `UseZeroCopy` option to import on the same terms: engage only when the layout is compatible and **never silently copy while reporting zero-copy**.
- `MemoryMarshal.AsBytes` for unmanaged flat layouts + Arrow buffers; `ReadOnlyMemory<T>` sharing into `Tensor<T>`/`TensorSpan<T>` where the BCL allows.

**Key files:** `src/Nivara.Extensions/IO/ArrowInterop.cs`, `src/Nivara.Extensions/IO/ArrowConversionOptions.cs`, `src/Nivara.Extensions/IO/ArrowValidityMask.cs`, `src/Nivara/Storage/ColumnStorage.cs`, `src/Nivara/Tensors/TensorInteropExtensions.cs`.

**Dependencies:** Phases A–C for the **import** half (sharing a validity buffer needs the bitmap representation and the layout model). The export half bypassed them: only the values buffer is shared, and the validity bitmap is rebuilt per call.

**Acceptance criteria:**
- ✅ Export: tests show the Arrow data buffer aliases the column's array (no element loop) for flat primitive columns, null-free and null-bearing.
- ✅ Export: shared-buffer reads agree with the copying path bit for bit, including the validity bitmap.
- ✅ An explicit zero-copy request on an incompatible layout fails loudly with a clear exception.
- Remaining: the same three properties for `FromArrowTable` and for Parquet in both directions.

**Risks:** Apache.Arrow's buffer APIs vs. `Tensor<T>` internal layout (nint dims, padding); memory-lifecycle safety of handing managed `Memory<T>` to native-bound arrays. Mitigation: pin/`MemoryManager<T>` ownership, keep owned-buffer semantics explicit (AGENTS.md zero-copy notes already flag this). Import adds the reverse risk: an Arrow buffer owned by an IPC reader must not outlive the reader, so an aliased import needs an ownership rule before it is safe.

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

**Key files:** `src/Nivara/IO/` lazy sources, `src/Nivara/Execution/StreamingExecutionStrategy.cs`, `src/Nivara/Query/IQueryInterfaces.cs` (async seams already exist), `src/Nivara.Extensions/IO/NivaraParquetReader.cs`, `NivaraParquetWriter.cs`.

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

**What we leverage, not reinvent:** the existing storage seam (`IColumnStorage<T>`, sole-owner `ColumnStorage<T>`), the async members already on `IQuerySource` (`ReadChunkAsync`, `ToAsyncEnumerable` in `src/Nivara/Query/IQueryInterfaces.cs::IQuerySource`), `ColumnStorage<T>.Slice` (already a true `ReadOnlyMemory` view — proof the pattern works), `KernelSelector`, and the Arrow/Parquet interop test suite in `tests/Nivara.Tests/IO/`.

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

- `docs/plan/ARROW-REVIEW.md` — the lens, the eight pillars, and the scorecard this roadmap closes out.
- `docs/plan/POLARS-REVIEW.md` — the query-engine lens (the boxed expression evaluator finding).
- `docs/plan/POLARS-ROADMAP.md` — the query/expression roadmap this roadmap converges with.
- `docs/IDEA.md` — original product vision (storage abstraction, Arrow interop sections). Retired 2026-08-06; the remaining storage/Arrow-adjacent items are tracked as GitHub issues: [dictionary encoding #132](https://github.com/khurram-uworx/Nivara/issues/132), [custom storage plug-in seam #133](https://github.com/khurram-uworx/Nivara/issues/133).
- `docs/TENSORS.md` — tensor-vs-Polars strategic framing.
- `docs/adr/001-autodiff-nonnullable-domain.md` — the explicit-null boundary all validity work must respect.
