# Examples

This folder contains sample projects and documentation demonstrating Nivara's capabilities including .NET-native machine learning.

## [NivaraIncident/README.md](NivaraIncident/README.md) — Production Incident Replay & Investigation

A reference application that models a production telemetry environment and replays/investigates incidents entirely through Nivara's columnar pipeline — typed expressions, rank family, rolling windows, partitioned windows, chunked/streaming `AsStream` execution, and execution diagnostics. Doubles as a forcing function for the core library: its gap inventory (percentile/quantile/median + stddev aggregation, public execution diagnostics, Parquet chunk streaming) is tracked in the README and planned in `Incident-PLAN.md`.

## [NivaraTorch/README.md](NivaraTorch/README.md)

### Cross-Framework Parity: PyTorch ↔ Nivara

Nivara provides .NET developers correct autograd without leaving the ecosystem — no Python runtime, no large PyTorch install, no GPU required. These parity examples prove that for CPU-based training, inference, and gradient computation, Nivara's forward and backward autograd produce effectively identical results to PyTorch.

The examples include:
- **Backward-mode (MLP FraudNet)**: Trains an identical 3-layer MLP in both frameworks and compares loss curves, validating reverse-mode autograd, optimizers, and training loop correctness.
- **Forward-mode (JVP Parity)**: Computes Jacobian-vector products for 6 canonical operations and compares, validating forward-mode autograd.

### Per-Layer PyTorch ↔ Nivara Comparison

Formal A/B validation of every NN layer type. PyTorch generates reference tensors via `gen_reference.py`; Nivara reproduces them to machine precision across all supported layer types, activations, loss functions, and full-model logits.

## [MicroGpt/README.md](MicroGpt/README.md) — Character-level Transformer on Nivara AutoDiff

A faithful per-position port of Andrej Karpathy's microgpt.py that trains a miniature GPT language model on the makemore names dataset. This is the first Nivara showcase example, proving that Nivara's AutoDiff engine can train a real transformer — not just MLPs — with correct gradients and no external dependencies beyond the Nivara core library.

## [NivaraGpt/README.md](NivaraGpt/README.md) — Character-level Transformer (Nivara-Native)

A miniature GPT language model built the **Nivara way** — using `Module<T>`, `TransformerBlock<T>`, `CrossEntropyLoss<T>`, `Sampler<T>`, and batched causal attention. Same task as MicroGpt, but built on Nivara's high-level APIs with significantly higher throughput due to batched MatMul and SIMD-accelerated kernels.

## [NivaraClassifier/README.md](NivaraClassifier/README.md) — Word-Level Text Classifier

A word-level text classifier that trains a sentiment model (positive/negative) using learned embeddings and an MLP head. Exercises the full autograd training pipeline with sequence data: synthetic data generation → tokenization → embedding → mean pool → MLP → cross-entropy loss → training → inference.

## [NivaraFineTuning/README.md](NivaraFineTuning/README.md) — DistilBERT Fine-Tuning on GLUE SST-2

Fine-tunes a pre-trained DistilBERT model for binary sentiment classification on the GLUE SST-2 dataset — entirely in C#, no Python runtime required for inference.

## [NivaraChess/README.md](NivaraChess/README.md) — Neural Chess Position Evaluator

Trains a neural network to evaluate chess positions using Nivara's autograd engine. Demonstrates non-NLP use of the library: sparse embeddings (`SparseEmbedding<T>` for NNUE halfKP features), Stockfish knowledge distillation via UCI, and `IEmbeddingGenerator<T>` integration.

## [NivaraChat/README.md](NivaraChat/README.md) — Hybrid Agent Workflow

Demonstrates Nivara-trained domain-specific models as first-class participants in `Microsoft.Agents.AI.Workflows` graphs, mixed with an Ollama-backed `ChatClientAgent` node. Also hosts the batched TinyShakespeare transformer (`--tinyshakespeare` mode) served as an `IChatClient` via DI — the single home for all Microsoft.Extensions.AI / Agent Framework integration.

## [NivaraTimeSeries/README.md](NivaraTimeSeries/README.md) — Server Monitoring Anomaly Detection

Trains a Conv1d-based Variational Autoencoder on synthetic server monitoring metrics (CPU, memory, disk I/O, network traffic), then detects anomalies by measuring reconstruction error against learned normal patterns.

## [NivaraInference/README.md](NivaraInference/README.md) — HuggingFace Vision Model Inference

Loads pre-trained HuggingFace models (MobileNetV2, ResNet-18, MiniLM-L6-v2) using a custom zero-dependency SafeTensors reader and runs forward inference entirely within Nivara's AutoDiff engine. No third-party ML framework dependencies.

## [NivaraVAE/README.md](NivaraVAE/README.md) — Variational Autoencoder for Synthetic Pattern Generation

A variational autoencoder that learns latent representations of synthetic 2D patterns (circles, stripes, blobs, checkerboards, corners, crosses). Demonstrates encoder–decoder architecture, reparameterization trick, and latent space exploration — all powered by Nivara's autograd engine.
