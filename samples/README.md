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

## [NivaraChat/README.md](NivaraChat/README.md) — Hybrid Agent Workflow

Demonstrates Nivara-trained domain-specific models as first-class participants in `Microsoft.Agents.AI.Workflows` graphs, mixed with an Ollama-backed `ChatClientAgent` node.

Key characteristics:
- Four trained models (sentiment, entity, workflow validator, agents validator) wired into a workflow graph
- Two execution modes: `--workflow` (fan-out/fan-in executors) and `--agents` (sequential `IChatClient` → `AsAIAgent()` pipeline)
- `NivaraChatClient : IChatClient` wraps each model for Agent Framework participation
- Hybrid deterministic (Nivara) + stochastic (LLM) pipeline
- `TextClassifierModel<T>`, `TokenClassifierModel<T>`, `TextTokenizer` — core APIs exercised
- `ModelSerializer` bridges training output to inference input
- Ollama optional — pass `--ollama` to include LLM agent


