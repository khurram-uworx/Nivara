# If AI data infrastructure complemented the .NET AI stack — the lens and the review

**Status:** architecture reference · **Scope:** the AI/data plane — AutoDiff (`src/Nivara/AutoDiff/`), tensor interop (`src/Nivara/Tensors/`), ML.NET interop (`src/Nivara.Extensions/MLNet/`), AI samples · **Companion:** `docs/plan/AISTACK-ROADMAP.md`, `docs/plan/POLARS-REVIEW.md`, `docs/plan/ARROW-REVIEW.md`

**Reassessed 2026-11-18 (issue #1292):** Phases 1–5 delivered on `main`. ONNX export (#866 closed), flat embedding convention (#867 closed), frame-level dataset ops (#1291), frame-scoped metrics (#868 closed), and read-only views incl. `TryGetReadOnlyTensorSpan` (#107, #1290) are all implemented. SafeTensors import + model layer (Bert/ModernBert/Laya/Llama/Qwen/SmolLM) + `TensorDataset`/`DataLoader` + `NivaraEmbeddingGenerator` grew substantially beyond the original scope. Remaining items are bounded open follow-ups (ONNX supported subset, TrainingLoop↔frame batching wiring, chunked-column view expansion). Non-goals remain explicit: no ONNX import/inference, no embedding column kind, no pipeline-level metrics, no mutable tensor exposure.
**Reassessed 2026-11-03:** Phase 5 (read-only views) delivered (#107); Phases 1–4 open. SafeTensors import + model layer (Bert/ModernBert/Laya/Llama/Qwen/SmolLM) + `TensorDataset`/`DataLoader` + `NivaraEmbeddingGenerator` grew substantially beyond the original scope, but the three headline gaps (ONNX export, embedding convention, frame-scoped metrics) remain open. Scope update (2026-11-03): Phase 1 re-scoped to **ONNX export only**, Phase 2 re-scoped to a **flat `[rows, dim]` frame convention**, Phase 4 re-scoped to a **minimal frame-scoped metrics module**. See `docs/plan/AISTACK-ROADMAP.md`.
**Reassessed 2026-10-23:** Phase 5 (read-only views) delivered; Phases 1–4 open.

This document evaluates Nivara through a third lens: **"What would AI data infrastructure look like if it were born in .NET to complement the .NET AI stack — with Azure and cloud explicitly off the table?"**

The first two lenses asked whether Nivara behaves like a query engine born in .NET (`docs/plan/POLARS-REVIEW.md`) and like a columnar memory system born in .NET (`docs/plan/ARROW-REVIEW.md`). This lens asks whether Nivara is the *data layer* the local .NET AI ecosystem is missing.

**Verdict:** this is Nivara's **strongest** lens. The Polars lens found a fatal flaw (the boxed expression evaluator); the Arrow lens found zero-copy scaffolding. This lens finds neither — the project has already invested here. As of 2026-11-18 (issue #1292) Nivara complements the .NET AI stack at ~90%: ONNX *export* (#866), the flat embedding convention (#867), frame-scoped metrics (#868), frame-level dataset ops (#1291), and read-only tensor views (#107/#1290) have all landed. The remaining gaps are narrow and *additive*: ONNX supported-subset bounding, wiring `TrainingLoop` to frame-level batching, and chunked-column view expansion. Non-goals (no ONNX import/inference, no embedding column kind, no pipeline-level metrics, no mutable tensor exposure) remain explicit.
---

## 1. The target: the .NET AI stack, local only

"Complement the .NET AI stack" must not be confused with "augment Microsoft's AI stack." Azure services — Azure OpenAI, Azure ML, AI Foundry, cloud vector stores — are **off the table**. The relevant stack is what .NET ships locally:

| Component | Role |
| --- | --- |
| `System.Numerics.Tensors` (`Tensor<T>`, `TensorPrimitives`) | The BCL tensor substrate — memory + vectorized kernels |
| ML.NET | On-device training + `IDataView` pipelines |
| ONNX Runtime | The universal local model runtime (`.onnx` interchange) |
| TorchSharp | PyTorch bindings for .NET |
| `Microsoft.Extensions.AI` | `IEmbeddingGenerator`, `Embedding<T>`, chat abstractions |
| `Microsoft.ML.Tokenizers` | BERT/BBPE tokenization |

A data layer born to complement *this* stack makes the boundary between "structured data" and "model input" disappear — without a cloud dependency anywhere.

---

## 2. The eight pillars

### Pillar 1 — Tensor-first data plane

- **Principle:** columns and frames interconvert with `Tensor<T>`/`ReadOnlyTensorSpan<T>`/`TensorPrimitives` with no boxing and no element-level ceremony — and *without ever breaking immutability*.
- **.NET substrate:** `Tensor<T>`, `ReadOnlyTensorSpan<T>`, `TensorPrimitives`, generic `INumber<T>` kernels.
- **Key tension:** tensors want writable spans; Nivara columns are immutable. The native answer is *read-only views where safe*, copies only when unavoidable — never a writable span over immutable data.

### Pillar 2 — Feature engineering into ML.NET

- **Principle:** the dataframe is the feature-engineering surface; ML.NET is the training surface; results round-trip back into frames. `IDataView` and `VBuffer` are the currency.
- **.NET substrate:** `MLContext`, `IDataView`, `VBuffer<T>`, `IEstimator`/`ITransformer`.

### Pillar 3 — Model interchange for the local runtimes

- **Principle:** weights move between runtimes in universal or interoperable formats: **ONNX** (the universal local format), **PyTorch state dicts** (TorchSharp's neighbor), **SafeTensors**, and ML.NET models — plus the engine's own serialization.
- **.NET substrate:** ONNX Runtime (`Microsoft.ML.OnnxRuntime`), TorchSharp, `Microsoft.ML`, SafeTensors.
- **Why it matters:** an engine that can export its trained modules to ONNX plugs into the entire local .NET AI ecosystem. One that cannot is an island.

### Pillar 4 — On-device training & inference, inference-default

- **Principle:** a native-trainable stack (AutoDiff) coexists with ONNX Runtime and TorchSharp rather than competing — inference is the default path, training is explicit.
- **.NET substrate:** reverse-mode AD, module system, optimizers, `GradientUtils.Grad()`.

### Pillar 5 — Embeddings as a data type

- **Principle:** an *embedding column* is a first-class column kind — vector-valued elements, similarity/top-k operations — wired directly into `Microsoft.Extensions.AI.Embedding<T>` so generated embeddings land in frames as data.
- **.NET substrate:** `Microsoft.Extensions.AI` `Embedding<T>`, `IEmbeddingGenerator`, `GeneratedEmbeddings<T>`.

### Pillar 6 — Dataset & data-loader layer

- **Principle:** split / shuffle / batch are *data operations* on frames that feed training loops and evaluation — not training-loop internals.
- **.NET substrate:** frame ops + `TrainingLoop` / `DataParallelTrainer` consuming them.

### Pillar 7 — Evaluation & metrics on data

- **Principle:** accuracy, F1, loss curves, confusion matrices are computed on frames — the dataframe is the evaluation surface.
- **.NET substrate:** vectorized frame kernels (already present).

### Pillar 8 — Allocation discipline + safe views

- **Principle:** AI hot paths obey `ArrayPool`/`Span<T>` discipline; data-to-tensor views are read-only (immutability preserved); NativeAOT is the publishing story.
- **.NET substrate:** `ArrayPool`, `Span`/`Memory`, pooled buffers (shared with the other two reviews).

---

## 3. Nivara scorecard

| Pillar | Status | Evidence |
| --- | --- | --- |
| 1 · Tensor-first data plane | 🟡 Partial | `TensorInteropExtensions` (`src/Nivara/Tensors/TensorInteropExtensions.cs`): frame↔`Tensor<T>`, 1D `FromTensor`, `FlattenFromTensor`; `ToTensorSpan` deliberately copy-based to preserve immutability; zero-copy lazy view via `AsTensorView()` for flat, null-free columns (nulls/reference types throw); `TryGetReadOnlyTensorSpan` delivered (#1290) for flat, contiguous, null-free, unmanaged columns; conversions still element-copy; chunked-column view expansion open |
| 2 · Feature engineering → ML.NET | ✅ Strong | `MLNetInterop`/`MLNetExtensions` (`src/Nivara.Extensions/MLNet/`): `LoadFromNivaraFrame`/`ToNivaraFrame`, `Fit`, `Predict`, `Transform`, `VBuffer` batch tensors; covered by `tests/Nivara.Tests/MLNet/MLNetIntegrationTests.cs` |
| 3 · Model interchange | ✅ Strong | Own `ModelSerializer` (JSON state dicts) + dtype-aware SafeTensors import (Bert/ModernBert/Laya/Llama/Qwen/SmolLM) + `StateDictLoader`; **ONNX export delivered** (`src/Nivara.Extensions/Onnx/OnnxExporter.cs`, `IOnnxExportContext`, `Module<T>.ExportToOnnx`, `docs/ONNX.md`, `tests/Nivara.Tests/Onnx/*`); runtime parity verified via ONNX Runtime for the supported subset. Import/inference remains a non-goal (first-party ML.NET/ONNX Runtime) |
| 4 · On-device training/inference | ✅ Very strong | Full AutoDiff subsystem: modules (`Linear`, `Conv*`, `BatchNorm`, `TransformerBlock`, `MultiheadAttention`, `VAE`…), optimizers (`SGD`/`Adam`/`AdamW`), `TrainingLoop`, `DataParallelTrainer`, inference-default `GradientUtils.Grad()` |
| 5 · Embeddings as data type | ✅ Strong | Flat `[rows, dim]` convention delivered: `NivaraEmbeddingGenerator` (MEAI) in `src/Nivara.Extensions/AI/`, `NivaraEmbeddingExtensions`, `docs/EMBEDDINGS.md`, `NivaraFrame.RowCosineSimilarity`/`RowDot`; `Dot`/`CosineSimilarity`/`TopKDescending`. No new column kind (non-goal) |
| 6 · Dataset/data-loader layer | ✅ Strong | `TrainingLoop`/`DataParallelTrainer` manage batches; `TensorDataset<T>` + `DataLoader` at training-API level; frame-level split/shuffle/batch/chunk/stratified delivered (`src/Nivara/Data/FrameDataset.cs`, `Nivara.Dataset`/`FrameDataset`). `TrainingLoop` still consumes `DataLoader` directly (frame-level batching wiring is an additive follow-up) |
| 7 · Evaluation & metrics | ✅ Strong | Frame-scoped metrics delivered: `src/Nivara/Metrics/*` (`NivaraMetrics.ConfusionMatrixFrame`, `PerClassMetricsFrame`, `RegressionSummaryFrame`), documented in `docs/METRICS.md`. Pipeline-level evaluation remains a non-goal (first-party ML.NET) |
| 8 · Allocation discipline | ✅ Strong | `ArrayPool`/`BufferPool`/spans throughout AutoDiff (`AccumulateGradient`, `Adam`/`AdamW` state, fused kernels) |

Legend: ✅ native-aligned · 🟡 partially aligned · ❌ gap.

**Lens comparison:** Polars ≈ 75%, Arrow-physics ≈ gap-y on zero-copy/chunking, **this lens ≈ 80–85%**. The earlier estimate that Nivara only "partially augments Microsoft's AI stack" was scored against the wrong target (Azure/cloud). Against the *local .NET AI stack*, the score is high and the remaining work is additive, not architectural.

---

## 4. What is already right

- **The app layer already is the complement.** `samples/NivaraChat/` uses `Microsoft.Extensions.AI` + `Microsoft.Agents.AI` + **OllamaSharp** (local LLM via Ollama — no cloud) + `Microsoft.ML.Tokenizers`; `samples/NivaraChess/` implements the MEAI embedding abstraction. The integration pattern is proven end to end.
- **PyTorch-validated kernels are the ecosystem bridge.** <0.04% loss-curve divergence across the grown NivaraTorch suite (Llama decoder blocks, banded/batched attention, RMSNorm, rotary embeddings, …) means Nivara-trained weights exchange cleanly with the PyTorch ecosystem (TorchSharp's neighbor). "Train in Nivara, exchange weights with the world."
- **ML.NET interop is first-class**, not an afterthought.
- **Inference-default is the right native model.** `GradientUtils.Grad()`-gated graph construction matches how a local complementer should behave — predict by default, train explicitly.
- **`docs/TENSORS.md` already named the right categories** for the AI direction — structured AI datasets, embedding columns, vector search, RAG data preparation, feature engineering, evaluation datasets. This review makes that direction concrete against the local stack.

---

## 5. The gaps — the delta to "how it should have looked"

Remaining work is additive; nothing architectural needs tearing down (reconciled 2026-11-18, issue #1292). The three headline gaps are closed:

1. **ONNX export (#866 closed).** Delivered in `src/Nivara.Extensions/Onnx/OnnxExporter.cs` with `IOnnxExportContext`/`Module<T>.ExportToOnnx`, documented in `docs/ONNX.md`, and runtime-parity verified via ONNX Runtime (`tests/Nivara.Tests/Onnx/*`). Import/inference remains a non-goal (first-party ML.NET/ONNX Runtime). Open follow-ups: `ConvTranspose2d`, `MaxPool`/`AvgPool`, standalone `MultiheadAttention` (use `TransformerBlock`), batched `TransformerBlock` input, grouped/asymmetric conv constraints (validated at export).
2. **Embedding convention (#867 closed).** Delivered as a flat `[rows, dim]` frame convention — `NivaraEmbeddingGenerator` (MEAI), `NivaraEmbeddingExtensions`, `docs/EMBEDDINGS.md`, `NivaraFrame.RowCosineSimilarity`/`RowDot`. No embedding column type (non-goal).
3. **Dataset + evaluation layer (#868 closed; #1291).** Frame-level split/shuffle/batch/chunk/stratified delivered (`src/Nivara/Data/FrameDataset.cs`); frame-scoped metrics delivered (`src/Nivara/Metrics/*`, `docs/METRICS.md`). Open follow-up: wiring `TrainingLoop` to frame-level batching (it still consumes `DataLoader` directly). Pipeline-level evaluation stays in ML.NET (non-goal).

Additional delivered: read-only tensor views via `AsTensorView()` plus `TryGetReadOnlyTensorSpan` (#107, #1290); open follow-up is chunked-column view expansion.
---

## 6. Summary

The one-line verdict:

> Nivara is already the best-placed of the three lenses — it complements the **local .NET AI stack** at ~90% (reconciled 2026-11-18, issue #1292), proving the pattern in samples (MEAI, ML.NET, Ollama, SafeTensors, PyTorch parity). The last mile of *interchange and convention* has landed: ONNX *export* (#866), the flat embedding convention (#867), frame-scoped metric ops (#868), frame-level dataset ops (#1291), and read-only tensor views (#107/#1290). Remaining open follow-ups are bounded: ONNX supported-subset bounding, `TrainingLoop`↔frame batching wiring, and chunked-column view expansion. See issues #866, #867, #868 (closed) and #1290, #1291, #1292.
The roadmap to close that last mile is in **`docs/plan/AISTACK-ROADMAP.md`**.

## Related documents

- `docs/plan/AISTACK-ROADMAP.md` — the roadmap that closes out this review.
- `docs/plan/POLARS-REVIEW.md` / `docs/plan/POLARS-ROADMAP.md` — the query-engine lens and roadmap.
- `docs/plan/ARROW-REVIEW.md` / `docs/plan/ARROW-ROADMAP.md` — the columnar-physics lens and roadmap.
- `docs/AUTODIFF.md` — the AutoDiff subsystem this lens evaluates.
- `docs/TENSORS.md` — tensor/AI strategic framing.
