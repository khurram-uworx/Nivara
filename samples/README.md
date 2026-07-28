# Examples

This folder contains sample projects and documentation demonstrating Nivara's capabilities in .NET-native machine learning.

## [NivaraTorch/README.md](NivaraTorch/README.md)

### Cross-Framework Parity: PyTorch ↔ Nivara

Nivara provides .NET developers correct autograd without leaving the ecosystem — no Python runtime, no 900 MB PyTorch install, no GPU required. These parity examples prove it: for CPU-based training, inference, and gradient computation, Nivara's forward and backward autograd produce effectively identical results to PyTorch.

The examples include:
- **Backward-mode (MLP FraudNet)**: Trains an identical 3-layer MLP in both frameworks and compares loss curves, validating reverse-mode autograd, optimizers, and training loop correctness.
- **Forward-mode (JVP Parity)**: Computes Jacobian-vector products for 6 canonical operations and compares, validating forward-mode autograd.

Results show <0.04% loss-curve divergence and 1e-5 JVP tolerance.


### Per-Layer PyTorch ↔ Nivara Comparison

Formal A/B validation of every NN layer type. PyTorch generates reference tensors via `gen_reference.py`, Nivara reproduces them to machine precision across 47 test cases covering Conv2d, Conv1d, BatchNorm, ReLU, LeakyRelu, Sigmoid, Tanh, MaxPool, AdaptiveAvgPool, Linear, Embedding, Dropout, RMSNorm, LayerNorm, Softmax, LogSoftmax, MatMul, GELU, and all loss functions (BCE, CrossEntropy, MSE, L1). Full-model logits match Python to 6+ decimal places for both MobileNetV2 and ResNet-18.

Key characteristics:
- **47 PyTorch reference fixtures** — float32 binary files in `samples/data/torch-comparison/`
- **Per-layer NUnit tests** — `tests/Nivara.Tests/NivaraTorch/` organized by layer type
- **Fixture generator** — `gen_reference.py` with deterministic seed=42, covers every NN layer type used across all samples
- Documents normalization scope (`RMSNorm` global vs `PerRowRMSNorm` per-row). Conv1d weight layout matches PyTorch.

## [MicroGpt/README.md](MicroGpt/README.md) — Character-level Transformer on Nivara AutoDiff

A faithful per-position port of Andrej Karpathy's microgpt.py that trains a miniature GPT language model on the makemore names dataset (~32K names). This is the first Nivara showcase example, proving that Nivara's AutoDiff engine can train a real transformer — not just MLPs — with correct gradients, comparable performance to PyTorch (2.4× faster on CPU), and no external dependencies beyond the Nivara core library.

Key characteristics:
- Per-position forward/backward (not batched) — each token attends only to cached past tokens
- Weight tying by default (output projection reuses token embedding matrix)
- Uses `Embedding<T>`, `Linear<T>`, RMSNorm, SoftmaxList, and ConcatHeads via PadRight/PadLeft selection matrices

## [NivaraGpt/README.md](NivaraGpt/README.md) — Character-level Transformer (Nivara-Native)

A miniature GPT language model built the **Nivara way** — using `Module<T>`, `TransformerBlock<T>`, `CrossEntropyLoss<T>`, `Sampler<T>`, and batched causal attention. Same task as MicroGpt (character-level name generation on names.txt), but built on Nivara's high-level APIs.

Key characteristics:
- Batched full-sequence forward with upper-triangular causal mask (not per-position)
- `Module<T>` subclass with `RegisterModules`/`RegisterParameters` — `StateDict()`, `LoadStateDict()`, `ModelSerializer` work out of the box
- `TransformerBlock<T>` — reusable core library building block for multi-head attention + MLP
- **`--norm-type rmsnorm|layernorm`** — configurable normalization: RMSNorm (default, faster) or standard LayerNorm with mean+variance
- `CrossEntropyLoss<T>` with integer labels, `Dropout<T>`, `Sampler<T>`
- **7x higher throughput** than MicroGpt (3,400 vs 460 tok/s) due to batched MatMul kernels and SIMD-accelerated TensorPrimitives

## [NivaraClassifier/README.md](NivaraClassifier/README.md) — Word-Level Text Classifier

A word-level text classifier that trains a sentiment model (positive/negative) using learned embeddings and an MLP head. Exercises the full autograd training pipeline with sequence data: synthetic data generation → tokenization → embedding → mean pool → MLP → cross-entropy loss → training → inference.

Key characteristics:
- `Embedding<T>` → `MeanPool` → `Linear(ReLU)` → `Linear` architecture (default `--mode linear`)
- **`--mode conv`**: Multi-branch TextCNN using `Conv1d<T>` with parallel kernel sizes (3, 5, 7) for n-gram feature extraction, demonstrating `TransposeAxes` and `Concat`
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
- Two execution modes: `--workflow` (fan-out/fan-in executors) and `--agents`/`--interactive` (sequential `IChatClient` → `AsAIAgent()` pipeline with live input)
- Single-shot mode: `--text <message>` runs the pipeline on one message and exits
- `NivaraChatClient : IChatClient` wraps each model for Agent Framework participation
- Hybrid deterministic (Nivara) + stochastic (LLM) pipeline
- `TextClassifierModel<T>`, `TokenClassifierModel<T>`, `TextTokenizer` — core APIs exercised
- `ModelSerializer` bridges training output to inference input
- Ollama optional — pass `--ollama` to include LLM agent

## [NivaraTimeSeries/README.md](NivaraTimeSeries/README.md) — Server Monitoring Anomaly Detection

Trains a Conv1d-based Variational Autoencoder on synthetic server monitoring metrics (CPU, memory, disk I/O, network traffic), then detects anomalies by measuring reconstruction error against learned normal patterns. Demonstrates temporal feature extraction with `Conv1d<T>`, `BatchNorm1d<T>` with 3D `[B, C, L]` input, and practical anomaly detection — all pure C# with no external dependencies.

Key characteristics:
- **Conv1d encoder** — 3-layer convolutional feature extraction from multivariate time series
- **`BatchNorm1d<T>`** — normalizes across batch and time dimensions (exercises the new 3D input support)
- **`MSELoss<T>` with `reduceToMean`** — reconstruction loss for continuous-valued metrics
- **`KlDivergence` + `SampleNormal`** — VAE latent regularization with reparameterization trick
- **Threshold-based anomaly detection** — reconstruction error exceeds `mean + 2σ` threshold
- **Synthetic data generation** — realistic diurnal patterns with injected anomalies (spikes, level shifts, trend changes)
- **ASCII visualization** — Unicode block characters for metric rendering
- **Model save/load** via `ModelSerializer`
- Exposed 4 library gaps: BatchNorm1d 3D input rejection, xHat latent allocation bug, xHat scalar path not written when affine=false, MSELoss lacking `reduceToMean`

## [NivaraInference/README.md](NivaraInference/README.md) — HuggingFace Vision Model Inference

Loads pre-trained HuggingFace models (MobileNetV2, ResNet-18, MiniLM-L6-v2) using a custom zero-dependency SafeTensors reader and runs forward inference entirely within Nivara's AutoDiff engine. No third-party ML framework dependencies.

Key characteristics:
- **Custom SafeTensors loader** — zero-dependency binary parser with MemoryMarshal.Cast for F32, throws on unsupported dtypes (F16, BF16)
- **MobileNetV2** — 16 inverted residual blocks, depthwise separable convolutions, ReLU6 activation, residual skip connections (3.4M params)
- **ResNet-18** — BasicBlock with 3x3 convolutions, identity/1x1 shortcut, 7x7 stem with MaxPool2d (11.7M params)
- **MiniLM-L6-v2** — 6-layer pre-norm BERT encoder with GELU activation, [CLS] pooling, L2 normalization (22.7M params, 384-dim embeddings)
- Exercises vision: `Conv2d<T>`, `BatchNorm2d<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, `Linear<T>`
- Exercises NLP: `Embedding<T>` (Gather), `LayerNorm<T>`, `GELU`, `MultiheadAttention<T>` (padding mask), `Linear<T>`
- Models downloaded via `hf` CLI to `samples/data/`

## [NivaraVAE/README.md](NivaraVAE/README.md) — Variational Autoencoder for Synthetic Pattern Generation

A variational autoencoder that learns latent representations of synthetic 2D patterns (circles, stripes, blobs, checkerboards, corners, crosses). Demonstrates encoder–decoder architecture, reparameterization trick, and latent space exploration — all powered by Nivara's autograd engine.

Key characteristics:
- **`--mode linear|conv`**: `linear` uses `VaeModel<T>` (MLP), `conv` uses `ConvVAE<T>` (Conv2d/ConvTranspose2d encoder/decoder)
- `Module<T>` subclass with `Linear<T>`, `Dropout<T>`, `Activation.LeakyRelu<T>` — individual layer fields
- Manual training loop with `GradientUtils.Grad()` — demonstrates explicit autograd scope control
- `SampleNormal` (reparameterization trick), `KlDivergence`, `BCEWithLogitsLoss` (fused backward)
- Latent space exploration — generation, interpolation between encoded patterns, per-dimension walks
- Drives core library improvements: fused BCE backward (correct gradient at x=0), ADR-001 null cleanup


