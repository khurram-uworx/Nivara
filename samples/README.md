# Examples

This folder contains sample projects and documentation demonstrating Nivara's capabilities in .NET-native machine learning.

## [PyTorch.md](PyTorch.md) — Cross-Framework Parity: PyTorch ↔ Nivara

Nivara provides .NET developers correct autograd without leaving the ecosystem — no Python runtime, no 900 MB PyTorch install, no GPU required. These parity examples prove it: for CPU-based training, inference, and gradient computation, Nivara's forward and backward autograd produce effectively identical results to PyTorch.

The examples include:
- **Backward-mode (MLP FraudNet)**: Trains an identical 3-layer MLP in both frameworks and compares loss curves, validating reverse-mode autograd, optimizers, and training loop correctness.
- **Forward-mode (JVP Parity)**: Computes Jacobian-vector products for 6 canonical operations and compares, validating forward-mode autograd.

Results show <0.04% loss-curve divergence and 1e-5 JVP tolerance.

## [MicroGpt/README.md](MicroGpt/README.md) — Character-level Transformer on Nivara AutoDiff

A faithful per-position port of Andrej Karpathy's microgpt.py that trains a miniature GPT language model on the makemore names dataset (~32K names). This is the first Nivara showcase example, proving that Nivara's AutoDiff engine can train a real transformer — not just MLPs — with correct gradients, comparable performance to PyTorch (2.4× faster on CPU), and no external dependencies beyond the Nivara core library.

Key characteristics:
- Per-position forward/backward (not batched) — each token attends only to cached past tokens
- Weight tying by default (output projection reuses token embedding matrix)
- Uses `Embedding<T>`, `Linear<T>`, RMSNorm, SoftmaxList, and ConcatHeads via PadRight/PadLeft selection matrices

## [NivaraGpt/README.md](NivaraGpt/README.md) — Character-level Transformer (Nivara-Native)

A miniature GPT language model built the **Nivara way** — using `Module<T>`, `TrainingLoop<T>`, `DataLoader<T>`, `TensorDataset<T>`, and batched causal attention. Same task as MicroGpt (character-level name generation on names.txt), but built on Nivara's high-level APIs.

Key characteristics:
- Batched full-sequence forward with upper-triangular causal mask (not per-position)
- `Module<T>` subclass with `RegisterModules`/`RegisterParameters` — `StateDict()`, `LoadStateDict()`, `ModelSerializer` work out of the box
- `TransformerBlock<T>` — reusable core library building block for multi-head attention + MLP
- `CrossEntropyLoss<T>` with integer labels, `Dropout<T>`, `Sampler<T>`
- **7x higher throughput** than MicroGpt (3,400 vs 460 tok/s) due to batched MatMul kernels and SIMD-accelerated TensorPrimitives

## [NivaraClassifier/README.md](NivaraClassifier/README.md) — Word-Level Text Classifier

A word-level text classifier that trains a sentiment model (positive/negative) using learned embeddings and an MLP head. Exercises the full autograd training pipeline with sequence data: synthetic data generation → tokenization → embedding → mean pool → MLP → cross-entropy loss → training → inference.

Key characteristics:
- `Embedding<T>` → `MeanPool` → `Linear(ReLU)` → `Linear` architecture
- `ReverseGradOperations.MeanPool<T>` — new core autograd operation for `[B, L, D]` → `[B, D]` sequence reduction
- Reusable `TextTokenizer` with vocab building, encode/decode, special tokens
- Synthetic data generator — no external datasets required
- `TrainingLoop<T>`, `DataLoader<T>`, `TensorDataset<T>`, `CrossEntropyLoss<T>` with integer labels
- Interactive wizard, CLI commands (`generate`, `train`, `predict`), model save/load
- **100% test accuracy** on synthetic data after 20 epochs (~1.5s)

## [NivaraChess/README.md](NivaraChess/README.md) — Neural Chess Position Evaluator

Trains a neural network to evaluate chess positions using Nivara's autograd engine. Demonstrates non-NLP use of the library: sparse embeddings (`SparseEmbedding<T>` for NNUE halfKP features), Stockfish knowledge distillation via UCI (`eval` command with `ucinewgame` sync), and `IEmbeddingGenerator<T>` integration.

Three phases: material counting (MLP), NNUE halfKP (sparse embedding), and Stockfish-labeled training. Includes save/load, interactive wizard, interactive REPL, UCI engine mode, and embedding demo.

## [NivaraChatClient.md](NivaraChatClient.md) — Hybrid Agent Workflow (Planned)

**Status:** Spec/plan — not yet implemented as a runnable project.

Demonstrates Nivara-trained domain-specific models as custom `Executor` subclasses in `Microsoft.Agents.AI.Workflows` graphs, mixed with LLM-backed `ChatClientAgent` nodes.

Architecture (conceptual):
```
Input → [NivaraSentimentExecutor] → [NivaraEntityExtractor] → [LLMAgent] → [NivaraValidator] → Output
```

Key characteristics of the approach:
- No `IChatClient` implementation — the prior standalone-chat-client direction was abandoned
- Training pipelines are separate from inference workflows (different projects/phases)
- Each domain model (sentiment, entity, validator) is a `Module<T>` subclass trained via `TrainingLoop<T>`
- Each is wrapped in an `Executor` subclass with `[MessageHandler]` for type-safe routing
- `ModelSerializer` bridges training output to inference input (JSON save/load)
- The example builds toward a hybrid workflow: deterministic Nivara nodes + stochastic LLM node

---

# Gap Analysis

## Resolved Gaps

The following gaps were identified during early example development and have been resolved.

| Gap | Resolution |
|-----|------------|
| **A: Embedding\<T\> not a Module\<T\>** | `Embedding<T>` now extends `Module<T>` |
| **B: No batched Embedding Forward** | `Embedding<T>.Forward(ReverseGradTensor<T>)` accepts batched token IDs |
| **C: No Gather operation** | `ReverseGradOperations.Gather<T>` implemented with backward scatter-add |
| **D: No integer-to-differentiable-tensor path** | `Gather` accepts `int[]` indices directly |
| **E: No batched TransformerBlock** | `TransformerBlock<T>` in core — pre-LN, causal mask, dropout, residual |
| **H: No integer-label CrossEntropyLoss** | `CrossEntropyLoss<T>.Forward(logits, int[])` overload added |
| **K: No sampling utilities** | `Sampler<T>` with temperature and top-k |
| **O: No MeanPool operation** | `ReverseGradOperations.MeanPool<T>` added — core autograd op for `[B,L,D]` → `[B,D]` with backward gradient distribution |

## Open Gaps

### Gap F: No LinearClassifier\<T\> convenience class

**Impact:** The simplest classifier (linear → softmax) requires `Sequential(Linear, Softmax)` each time. Not a correctness issue, but a developer-experience gap.

**Fix:** Add `LinearClassifier<T> : Module<T>` in `AutoDiff/Nn/`.

### Gap G: Vocabulary/feature extraction pipeline is not part of the library

**Impact:** Every example project must write its own tokenizer/feature extractor. No "load text → feature vector" pipeline exists in the library.

**Fix:** Application-level concern. A `Nivara.Extensions.Text` package with bag-of-words or TF-IDF would help. NivaraClassifier's `TextTokenizer` could be promoted if useful.

### Gap L: ModelSerializer cannot load into non-Module\<T\> models

**Impact:** `ModelSerializer.Load<T>(model, path)` requires an already-instantiated `Module<T>`. A factory method that reads JSON and instantiates the correct subclass would be convenient.

### Gap M: Agent Framework packages not referenced anywhere

**Impact:** The NivaraChatClient example project must reference `Microsoft.Agents.AI`, `Microsoft.Agents.AI.Workflows`, and `Microsoft.Extensions.AI`. Core stays clean. Documentation/example concern only.

### Gap N: No runnable NivaraChatClient example yet

**Impact:** The best showcase of Nivara's value proposition (deterministic model in an Agent Framework workflow) only exists as a markdown spec. Building `samples/NivaraWorkflow/` is the next implementation step.

---

## Priority Order for Open Gaps

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| P0 | N: Build the NivaraChatClient example | Large | Validates everything; biggest single improvement |
| P2 | F: LinearClassifier | Tiny | Developer convenience |
| P3 | G: Text feature extraction | Large | Application-level, could be an extension |
| P3 | L: LoadModel factory | Small | Nice-to-have API convenience |
| P3 | M: Agent Framework refs | None | Example concern only |
