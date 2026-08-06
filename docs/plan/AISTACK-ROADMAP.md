# AISTACK-ROADMAP — the road to complementing the .NET AI stack

**Status:** planning reference · **Scope:** AI/data plane — AutoDiff, tensor interop, ML.NET interop, dataset/eval layer · **Lens:** `docs/AISTACK-REVIEW.md`

This is the travel plan for closing the last mile between Nivara and the **local .NET AI stack** (Azure/cloud off the table): `System.Numerics.Tensors`, ML.NET, ONNX Runtime, TorchSharp, `Microsoft.Extensions.AI`, `Microsoft.ML.Tokenizers`. It complements `docs/POLARS-ROADMAP.md` (expression engine) and `docs/ARROW-ROADMAP.md` (columnar physics), which together supply the span-first, null-aware foundation this roadmap builds on.

---

## 0. Vision

> **Train in Nivara. Exchange with the .NET AI ecosystem. Data in, models out — no cloud required.**

Concretely, when this roadmap is done:

- AutoDiff modules **export to ONNX** and run anywhere ONNX Runtime runs; ONNX models **import** and infer over Nivara frames.
- **Embedding columns** are a first-class column kind — vector-valued, similarity/top-k — wired natively to `Microsoft.Extensions.AI.Embedding<T>`.
- **Dataset operations** (split / shuffle / batch) are frame operations that feed `TrainingLoop`; **metrics** (accuracy, F1, loss curves, confusion) are computed on frames.
- Data↔tensor conversion is zero-boxing and read-only-safe: no writable span ever exposed over immutable columns.

### Where we are today

The strongest lens of the three. AutoDiff is a complete on-device stack (`src/Nivara/AutoDiff/`); ML.NET interop is first-class (`src/Nivara.Extensions/MLNet/`); MEAI embedding integration is proven in samples (`samples/NivaraChess/ChessEmbeddingGenerator.cs`, `samples/NivaraChat/` uses MEAI + Agents + OllamaSharp); PyTorch-validated kernels bridge the weight ecosystem (55 tests, `tests/Nivara.Tests/NivaraTorch/`). The gaps are the last mile: no ONNX export/import, no embedding-column type, no dataset/eval layer, and tensor conversion is element-copy (deliberately, for immutability — `src/Nivara/Tensors/TensorInteropExtensions.cs:84-88`).

### Non-goals (explicit)

- **No Azure/cloud.** No Azure OpenAI, Azure ML, AI Foundry, or cloud vector stores — this roadmap stays fully local.
- **No reimplementation of ONNX Runtime or ML.NET.** They are *interchange targets*, not competitors. This roadmap bridges to them.
- **No runtime ONNX dependency in core.** ONNX interop belongs in `Nivara.Extensions` (same boundary rule as Arrow/Parquet); core keeps `System.Numerics.Tensors` only.
- **No mutable tensor exposure over immutable columns.** Read-only views or copies only — immutability is non-negotiable.

---

## 1. The roadmap

### Phase 1 — ONNX export/import *(the highest-value bridge)*

**Motivation:** Nivara-trained modules currently speak only their own JSON state dicts and SafeTensors. ONNX is the universal local model format; export/import is what turns Nivara from a self-contained training stack into the data layer of the local .NET AI ecosystem.

**Scope:**
- **Export:** traverse the module graph (`Module<T>`, `StateDict()`) and emit a minimal ONNX graph for the common module set — `Linear`, `Embedding`, `Conv1d`/`Conv2d`, `BatchNorm`, `LayerNorm`/`RMSNorm`, `TransformerBlock`/`MultiheadAttention`, activations, `Sequential` — including weights and inference-mode ops.
- **Import:** parse ONNX protobuf and materialize inference into Nivara (via a span-based ONNX reader or `Microsoft.ML.OnnxRuntime` in `Nivara.Extensions`), landing output tensors into frames.
- Target `Nivara.Extensions` (`Nivara.Onnx` namespace) with an `onnx` dependency there, never in core.

**Key files:** new `src/Nivara.Extensions/Onnx/` (exporter + importer), `src/Nivara/AutoDiff/Nn/` (module metadata/op enumeration for export), `src/Nivara/Tensors/TensorInteropExtensions.cs` (output → frame).

**Dependencies:** the span/null-aware foundation from POLARS-ROADMAP Phases 1–2 and ARROW-ROADMAP Phase D (zero-copy output landing).

**Acceptance criteria:**
- A trained `Linear`→`ReLU`→`Linear` and a `TransformerBlock` module export to `.onnx`; the file loads in ONNX Runtime and reproduces inference output within float tolerance.
- An ONNX model imports and runs over a `NivaraFrame` input, producing a frame output.
- Round-trip tests + parity vs. the same module run through Nivara directly.

**Risks:** ONNX op coverage (attention, layer-norm variance) is non-trivial; mitigation: start with the transformer/MLP subset actually exercised by samples, and use ONNX Runtime (not a hand-rolled interpreter) for import inference.

---

### Phase 2 — Embedding columns as a data type

**Motivation:** The MEAI integration pattern exists at the app layer (`ChessEmbeddingGenerator`), but the data layer has only scalar columns — embeddings land as raw floats, losing the "column of vectors" shape. A native complementer makes embeddings a *type*, not a projection.

**Scope:**
- New column kind: vector-valued elements (e.g., `NivaraColumn<T[]>`-backed by contiguous 2D storage or the ARROW-ROADMAP variable-binary layout) with `Similarity`/`CosineSimilarity`/`TopK` operations already present as kernels (`Dot`, `CosineSimilarity`, `TopKDescending`).
- Native interop with `Microsoft.Extensions.AI.Embedding<T>`: `NivaraFrame` ↔ `GeneratedEmbeddings<T>` conversion helpers (in `Nivara.Extensions`, where `Microsoft.Extensions.AI` already lives via samples).
- Wire an `IEmbeddingGenerator` over a frame column (the `ChessEmbeddingGenerator` pattern promoted to the data layer).

**Key files:** new `src/Nivara/` embedding column type (or `src/Nivara.Extensions/` if it needs MEAI types), `src/Nivara/Tensors/TensorsHelper.cs` (similarity kernels), new `src/Nivara.Extensions/AI/` MEAI conversions.

**Dependencies:** ARROW-ROADMAP Phase B (vector layout) if embeddings ride on variable-binary; otherwise standalone.

**Acceptance criteria:**
- An embedding column round-trips frame ↔ `Embedding<T>` list with values preserved.
- `TopKDescending`/`CosineSimilarity` over an embedding column match the scalar-kernel results.
- Null semantics preserved (a null embedding = null element, no NaN sentinel).

**Risks:** vector element storage adds layout complexity; mitigation: start with a contiguous `[rows, dim]` flat representation (frame `ToTensor<T>` already supports 2D) and layer the MEAI interop on top.

---

### Phase 3 — Dataset & data-loader layer

**Motivation:** Split/shuffle/batch are data operations today only by hand-rolling in each sample (`NivaraFineTuning`, `NivaraClassifier` each reinvent them). A complementer exposes them as frame operations that feed `TrainingLoop`.

**Scope:**
- Frame operations: `SplitTrainTest` (fraction/seed), `Shuffle` (seeded, reproducible), `Batch`/`Chunk` (frame → batch enumerables), `StratifiedSplit` for classification targets.
- A `Dataset<T>`/`DataLoader`-shaped API that yields batches as frames/tensors into `TrainingLoop`/`DataParallelTrainer`, replacing per-sample hand-rolled loaders.
- Seeded reproducibility so NivaraTorch parity and eval stay deterministic.

**Key files:** new `src/Nivara/Data/` (dataset ops), `src/Nivara/AutoDiff/Training/TrainingLoop.cs` (batch consumption), samples updated to use the new API.

**Dependencies:** POLARS-ROADMAP Phase 1 (typed expressions make split/shuffle predicates clean) and Phase 4 / ARROW-ROADMAP Phase F (async batching).

**Acceptance criteria:**
- Split/shuffle/batch results match hand-rolled equivalents in existing samples (parity tests).
- Same seed ⇒ same batches across runs.
- `TrainingLoop` consumes the new batching API without behavior change.

**Risks:** API shape churn against `TrainingLoop`'s existing batch management; mitigation: additive — keep `TrainingLoop`'s current entry points, add the dataset layer alongside, migrate samples incrementally.

---

### Phase 4 — Evaluation & metrics module

**Motivation:** Evaluation is currently diagnostics + manual metrics in samples. The dataframe is the natural surface for metrics — vectorized and null-aware.

**Scope:**
- Metric functions over frames: classification (`Accuracy`, `Precision`, `Recall`, `F1`, `ConfusionMatrix`), regression (`MAE`, `MSE`, `RMSE`, `R²`), and training telemetry (loss-curve aggregation from `TrainingLoop`).
- Null-aware semantics (predicted/target null pairs excluded, per ADR-001).
- Vectorized on existing `TensorPrimitives` kernels.

**Key files:** new `src/Nivara/Metrics/` (metrics module), `src/Nivara/AutoDiff/Training/TrainingLoop.cs` (loss-curve telemetry hook).

**Dependencies:** POLARS-ROADMAP Phase 2 (fused kernels) for single-pass metric computation; otherwise standalone on column kernels.

**Acceptance criteria:**
- Accuracy/F1/confusion match hand-computed references on known fixtures (including null rows).
- Loss curves aggregate correctly across epochs from `TrainingLoop`.
- Metrics run vectorized for large frames (per `OperationDiagnostics`).

**Risks:** null-handling ambiguity in metrics (ignore vs. error); mitigation: documented null-exclusion rule + property tests, matching the established mask semantics.

---

### Phase 5 — Safe read-only tensor views ✅ *delivered via public `AsTensorView()` (#107)*

**Status note:** delivered as the public zero-copy `NivaraColumn<T>.AsTensorView()` / `NivaraSeries<T>.AsTensorView()` surface (issue #107) — a lazy `Tensor<T>` view over the sole-owner backing array, guarded (nulls/reference types throw) rather than a `TryGet` contract. Callers treat the view as read-only; immutability is convention, not a `ReadOnlyTensorSpan<T>` type. `ToTensorSpan` still copies to protect immutability.

**Motivation:** `ToTensorSpan` copies to protect immutability (`TensorInteropExtensions.cs:84-88`) — the *right* principle, but a safe read-only view eliminates the copy for flat, null-free columns without ever exposing a writable span.

**Scope:**
- Add `TryGetReadOnlyTensorSpan<T>()` on `NivaraColumn<T>`/`NivaraSeries<T>` that returns a `ReadOnlyTensorSpan<T>` **view** over storage when the column is null-free and layout-compatible (flat), else `false` (caller falls back to the copy path).
- Aligns with ARROW-ROADMAP Phase D (true views) and feeds tensor-first kernels (`TensorPrimitives` chains) without copies.

**Key files:** `src/Nivara/NivaraColumn.cs`, `src/Nivara/NivaraSeries.cs`, `src/Nivara/Storage/ColumnStorage.cs` (the landed `AsTensorView`/`AsTensor` implementation).

**Dependencies:** ARROW-ROADMAP Phase D (internal zero-copy views) — this phase is its tensor-facing consumer.

**Acceptance criteria:**
- Null-free flat columns produce read-only views (no copy); null-bearing or non-flat columns fall back to the existing copy path.
- Immutability preserved: the view is `ReadOnlyTensorSpan<T>` and never a writable `TensorSpan<T>`.
- Tensor-kernel results via view == results via copy (parity tests).

**Risks:** tensor layout assumptions (nint dims, padding) vs. flat column buffers; mitigation: conservative `TryGet` contract — compatibility check, return `false` on any doubt.

---

## 2. Sequencing rationale

**Why ONNX first:** it is the highest-value, most ecosystem-visible bridge, and the samples already exercise the module subset it needs. It also forces the module-graph metadata that later phases (export of datasets, eval) benefit from.

**Why embeddings and dataset/eval follow:** both are data-layer types the samples already approximate at the app layer — promoting them is low-risk, high-consistency work.

**Why read-only views last:** it depends on ARROW-ROADMAP Phase D (internal zero-copy) and is an optimization, not a capability — correct to sequence after the correctness work.

**Cross-roadmap convergence:**

| AISTACK phase | Depends on | Because |
| --- | --- | --- |
| 1 · ONNX | POLARS 1–2, ARROW D | span/null-aware kernels + zero-copy output |
| 2 · Embedding columns | ARROW B | vector layout option |
| 3 · Dataset layer | POLARS 1, POLARS 4 / ARROW F | typed expressions + async batching |
| 4 · Metrics | POLARS 2 | fused single-pass kernels |
| 5 · Read-only views | ARROW D | internal zero-copy views |

**What we leverage, not reinvent:** `TrainingLoop`/`DataParallelTrainer` (batch machinery), `ModelSerializer`/`StateDict` (module graph enumeration for export), `SafeTensorsLoader` (dtype-aware read pattern), the MEAI `IEmbeddingGenerator` implementation in `samples/NivaraChess/`, `TensorInteropExtensions` (conversion + the immutability principle), and the PyTorch parity fixture machinery in `tests/Nivara.Tests/NivaraTorch/`.

---

## 3. Cross-cutting conventions (every phase)

- **Diagnostics:** record ONNX export/import, layout, and metric-kernel decisions via `OperationDiagnostics` / `ExecutionEngine.LastDiagnostics`.
- **Testing:** NUnit 4.x, `Method_Scenario_ExpectedBehavior`, parity fixtures against PyTorch references (extend the NivaraTorch pattern), round-trip tests (train→export→ONNX Runtime→compare), null-semantics property tests.
- **Null semantics:** ADR-001 holds; predicted/target nulls excluded from metrics; no NaN sentinels anywhere.
- **No runtime cloud dependency:** any `Microsoft.Extensions.AI`-typed interop stays in `Nivara.Extensions`/samples, never core.
- **No comments in code** beyond non-obvious decisions; `.editorconfig` is authoritative.

---

## 4. Definition of done

The vision at §0 holds, verified by:

- A Nivara-trained transformer exports to ONNX and runs in ONNX Runtime with parity; ONNX models import and infer over frames.
- Embedding columns are a first-class type with MEAI `Embedding<T>` interop and working similarity/top-k.
- Split/shuffle/batch are frame operations consumed by `TrainingLoop`; metrics (accuracy/F1/confusion, regression, loss curves) compute on frames.
- Read-only tensor views eliminate copies for safe layouts without exposing writable spans.
- Full `dotnet build Nivara.slnx` + `dotnet test` green; zero cloud dependencies in core.

---

## Related documents

- `docs/AISTACK-REVIEW.md` — the lens, the eight pillars, and the scorecard this roadmap closes out.
- `docs/POLARS-ROADMAP.md` / `docs/POLARS-REVIEW.md` — the expression-engine roadmap and lens this roadmap builds on.
- `docs/ARROW-ROADMAP.md` / `docs/ARROW-REVIEW.md` — the columnar-physics roadmap and lens (Phase D here depends on ARROW Phase D).
- `docs/AUTODIFF.md` — the AutoDiff subsystem all phases touch.
- `docs/TENSORS.md` — tensor/AI strategic framing (Nivara's standing and committed direction).
- `docs/adr/001-autodiff-nonnullable-domain.md` — the null-boundary rule metrics and embeddings must respect.
