# Examples

This folder contains sample projects and documentation demonstrating Nivara's capabilities including .NET-native machine learning.

## [NivaraIncident/README.md](NivaraIncident/README.md) — Production Incident Replay & Investigation

Models a production telemetry environment and replays/investigates incidents entirely through Nivara's columnar pipeline — typed expressions, rolling windows, rank family, chunked streaming, and execution diagnostics. Ships four deterministic incident scenarios as a forcing function for core library gaps.

## [NivaraTorch/README.md](NivaraTorch/README.md) — Cross-Framework Parity: PyTorch ↔ Nivara

Proves Nivara's autograd produces effectively identical results to PyTorch for CPU-based training, inference, and gradient computation — no Python runtime, no GPU required. Covers backward-mode, forward-mode, and per-layer validation across 21+ NN layer types.

## [MicroGpt/README.md](MicroGpt/README.md) — Character-level Transformer on Nivara AutoDiff

A faithful per-position port of Karpathy's microgpt.py. The first Nivara showcase example — proves the AutoDiff engine can train a real transformer with correct gradients and no external dependencies.

## [NivaraGpt/README.md](NivaraGpt/README.md) — Character-level Transformer (Nivara-Native)

Same task as MicroGpt, but built on Nivara's high-level APIs — `Module<T>`, `TransformerBlock<T>`, `CrossEntropyLoss<T>`, batched causal attention, and model serialization. Demonstrates the idiomatic way to compose Nivara's NN building blocks.

## [NivaraClassifier/README.md](NivaraClassifier/README.md) — Word-Level Text Classifier

Trains a sentiment classifier using learned embeddings and an MLP head (or a multi-branch TextCNN). Exercises the full autograd training pipeline: synthetic data generation → tokenization → embedding → pooling → classification loss → training → inference.

## [NivaraFineTuning/README.md](NivaraFineTuning/README.md) — DistilBERT Fine-Tuning on GLUE SST-2

Fine-tunes a pre-trained DistilBERT model for binary sentiment classification on GLUE SST-2 — entirely in C#, no Python runtime. Demonstrates transfer learning with SafeTensors weight loading, AdamW optimization, and model persistence.

## [NivaraChess/README.md](NivaraChess/README.md) — Neural Chess Position Evaluator

Trains a neural network to approximate Stockfish's position evaluation via knowledge distillation. Demonstrates non-NLP use of the library: sparse embeddings, Stockfish UCI integration, and position embedding generation.

## [NivaraChat/README.md](NivaraChat/README.md) — Hybrid Agent Workflow

Demonstrates Nivara-trained domain-specific models as first-class participants in `Microsoft.Agents.AI.Workflows` graphs, mixed with Ollama-backed LLM agents. Includes confidence-based handoff, tool calling, writer-critic loops, RAG pipelines, online learning, and a batched TinyShakespeare transformer served as `IChatClient`.

## [NivaraTimeSeries/README.md](NivaraTimeSeries/README.md) — Server Monitoring Anomaly Detection

Trains a Conv1d-based Variational Autoencoder on synthetic server monitoring metrics, then detects anomalies by measuring reconstruction error against learned normal patterns. Demonstrates temporal feature extraction with `BatchNorm1d` on 3D input.

## [NivaraInference/README.md](NivaraInference/README.md) — HuggingFace Model Inference

Loads pre-trained HuggingFace models and runs forward inference entirely within Nivara's AutoDiff engine — no Python runtime, no CUDA, no third-party ML framework. Covers both vision models (MobileNetV2, ResNet-18) and text/NLP models (MiniLM, DistilBERT, DistilBERT SST-2). Includes a custom zero-dependency SafeTensors reader and PyTorch reference comparisons.

## [NivaraVAE/README.md](NivaraVAE/README.md) — Variational Autoencoder for Synthetic Pattern Generation

Trains a VAE to learn latent representations of synthetic 2D patterns. Supports both MLP and convolutional (`Conv2d`/`ConvTranspose2d`) architectures. Demonstrates encoder–decoder design, reparameterization trick, and latent space exploration (generation, interpolation, per-dimension walks).
