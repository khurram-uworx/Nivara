# Nivara HuggingFace Inference Sample

Load pre-trained HuggingFace models (MobileNetV2, ResNet-18, MiniLM, DistilBERT) into Nivara's zero-dependency tensor engine and run forward inference in pure managed .NET — no Python runtime, no CUDA, no third-party ML framework.

The same architecture is also implemented in PyTorch (`samples/NivaraInference/Python/`) for direct CPU performance comparison.

## Quick start

```bash
# Download model weights via HuggingFace CLI
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download timm/resnet18.augreg_in1k --local-dir samples/data/resnet18
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir samples/data/minilm
# (distilbert-base-uncased already present under samples/data/distilbert)

# Run inference
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2
dotnet run --project samples/NivaraInference -c Release -- resnet18
dotnet run --project samples/NivaraInference -c Release -- minilm
dotnet run --project samples/NivaraInference -c Release -- distilbert

# Benchmark (10 passes each)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
```

## Supported models

| Model | Type | Weight size | Tensors | Parameters | Output |
|-------|------|-------------|---------|------------|--------|
| MobileNetV2 | Vision (classification) | 13.5 MB | 262 | 3.4M | 1001 classes |
| ResNet-18 | Vision (classification) | 44.6 MB | 102 | 11.7M | 1000 classes |
| MiniLM (L6-v2) | Text (embedding) | 91 MB | 104 | 22.7M | 384-dim embedding |
| DistilBERT (base-uncased) | Text (encoder) | 255.5 MB | 105 | 67.0M | `[seqLen, 768]` hidden states |

## Usage

### C# (Nivara)

**Vision models:**
```bash
# Random-data inference
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18

# Benchmark (10 synthetic + real-image passes)
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- resnet18 benchmark

# Compare output with PyTorch reference
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare
dotnet run --project samples/NivaraInference -- resnet18 compare

# Step-by-step layer diagnostics
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare_diag
dotnet run --project samples/NivaraInference -- resnet18 compare_diag

# Single image inference
dotnet run --project samples/NivaraInference -- mobilenet_v2 path/to/image.jpg
dotnet run --project samples/NivaraInference -- resnet18 path/to/image.jpg
```

**MiniLM:**
```bash
# Tokenize and embed a sentence
dotnet run --project samples/NivaraInference -- minilm

# Benchmark (10 passes)
dotnet run --project samples/NivaraInference -- minilm benchmark

# Pairwise cosine similarity demo
dotnet run --project samples/NivaraInference -- minilm similarity
```

**DistilBERT:**
```bash
# Forward a sentence through the base encoder (output: [128, 768] hidden states)
dotnet run --project samples/NivaraInference -- distilbert

# Benchmark (3 warmup + 10 timed passes)
dotnet run --project samples/NivaraInference -- distilbert benchmark

# Compare hidden states with a PyTorch reference (run the Python script first)
python samples/NivaraInference/Python/distilbert_compare.py
dotnet run --project samples/NivaraInference -- distilbert compare
```

### Python (PyTorch reference)

```bash
cd samples/NivaraInference/Python
pip install -r requirements.txt

python mobilenet.py           # Basic inference
python resnet18.py

python mobilenet_compare.py   # Forward pass on shared input for C# comparison
python resnet18_compare.py

python mobilenet_diag.py      # Step-by-step layer diagnostics
python resnet18_diag.py

python minilm_benchmark.py     # MiniLM CPU timing (same methodology as C#)
python distilbert_benchmark.py # DistilBERT CPU timing (same methodology as C#)
python minilm_compare.py       # MiniLM reference embeddings for C# comparison
python distilbert_compare.py   # DistilBERT reference hidden states for C# comparison

python generate_input.py      # Regenerate shared comparison fixture
```

## Model architectures

### MobileNetV2

A lightweight classification network built from inverted residual blocks:

- **Stem**: 3×3 conv → BatchNorm → ReLU6
- **16 inverted residual blocks** with expansion/depthwise/project phases
- **Depthwise separable convolutions** (groups = input channels) for 3×3 layers
- **ReLU6** activation via `Clip(Relu(x), 0, 6)`
- **Residual shortcuts** only when `stride == 1 && inChannels == outChannels`
- **Head**: 1×1 conv → global avg pool → 1001-class linear classifier

Nivara modules used: `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `ReLU6` via `Clip` + `Relu`, depthwise grouped convolutions.

### ResNet-18

A standard 18-layer residual network:

- **Stem**: 7×7 conv → BatchNorm → ReLU → 3×3 MaxPool
- **4 stages** with channel progression: 64 → 128 → 256 → 512
- **BasicBlock**: two 3×3 convs with BatchNorm + ReLU, identity shortcut (or 1×1 conv when dimensions change)
- **Head**: global average pooling → 1000-class linear classifier
- **Downsampling** at stage boundaries via strided convolution in the shortcut path

Nivara modules used: `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, residual addition via `ReverseGradOperations.Add`.

### MiniLM (sentence-transformers/all-MiniLM-L6-v2)

A 6-layer Post-LN BERT encoder producing 384-dimensional sentence embeddings:

- **Embedding stack**: token + position + segment embeddings summed, then LayerNorm
- **6× Post-LN BERT layers**: LayerNorm → Self-Attention → residual → LayerNorm → FFN → residual
- **GELU activation** in the FFN intermediate (exact erf)
- **Bidirectional self-attention** with optional padding mask (via `MultiheadAttention<T>`)
- **[CLS] token pooling** — extracts the first token's embedding from the output sequence
- **L2 normalization** — output embedding normalized to unit length for cosine similarity
- **Tokenization** via `Microsoft.ML.Tokenizers.BertTokenizer` (sample-only dependency)

Nivara modules used: `Embedding<T>` (Gather path), `LayerNorm<T>`, `Linear<T>`, `MultiheadAttention<T>`, `ReverseGradOperations.GeluExact`, `ReverseGradOperations.Add`.

### DistilBERT (distilbert-base-uncased)

The 6-layer, 768-dim pre-trained encoder (the baby-step before the fine-tuned SST-2 showcase):

- **Embedding stack**: word + position embeddings (no token-type embeddings) summed, then LayerNorm
- **6× Post-LN DistilBERT layers**: self-attention → residual → `sa_layer_norm` → FFN (`lin1` → GELU → `lin2`) → residual → `output_layer_norm`
- **GELU activation** in the FFN intermediate (exact erf)
- **Weight mapping** from `distilbert.*` SafeTensors keys via `DistilBertLoader.LoadEncoderWeights`
- **Verification**: `last_hidden_state` matches HuggingFace to `max abs diff 5e-6` (cosine 0.99999988)

Nivara modules used: `Embedding<T>`, `LayerNorm<T>`, `Linear<T>`, `BertSelfAttention<T>`, `ReverseGradOperations.GeluExact`, `ReverseGradOperations.Add`, `ReverseGradOperations.Softmax`, `ReverseGradOperations.MatMul`.

> **GELU note:** BERT-family models (MiniLM, DistilBERT) use the exact erf GELU (`GeluExact`). The tanh approximation (`ReverseGradOperations.Gelu`) matches HF `gelu_new`/GPT-2 and is retained for GPT-style `TransformerBlock`.

### Weight loading

Each model defines a static `LoadWeights()` factory that maps HuggingFace tensor names to Nivara module parameters. No reflection or generic deserialization — explicit, type-safe loading with full compile-time checking.

- **MobileNetV2**: 262 tensors mapped to 262 module parameters (Conv2d weight/bias, BatchNorm running mean/var/weight/bias, Linear weight/bias)
- **ResNet-18**: 102 tensors mapped to 102 module parameters
- **MiniLM**: 96 tensors mapped from HuggingFace keys like `encoder.layers.N.attention.self.query.weight` to Nivara `Linear<T>` weight/bias fields
- **DistilBERT**: 105 tensors mapped via `DistilBertLoader.LoadEncoderWeights` from `distilbert.embeddings.*` and `distilbert.transformer.layer.{0-5}.*` keys

## SafeTensors loader

The sample includes a custom zero-dependency `SafeTensorsLoader` that parses the HuggingFace SafeTensors binary format directly:

- **Memory-mapped header parsing** via `System.Text.Json` — reads the JSON header from the first 8 bytes + offset table
- **Zero-copy tensor extraction** using `MemoryMarshal.Cast<byte, float>` — the weight data is reinterpret-cast directly from the memory-mapped file buffer
- **Format validation** — throws `NotSupportedException` for non-F32 tensors (F16, BF16) with clear guidance to the user

## Performance benchmarks

Measured on the same machine (CPU-only, no GPU). Nivara measured in Release mode. PyTorch uses MKL-optimized kernels. Both use batch size 1 with 3-pass warmup + 10 timed passes.

| Model | Input | PyTorch (CPU) | Nivara (.NET 10) | Slowdown |
|-------|-------|---------------|-------------------|----------|
| **MobileNetV2** | 1×3×224×224 | 115 ms | 2,471 ms | **~21×** |
| **ResNet-18** | 1×3×224×224 | 68 ms | 667 ms | **~10×** |
| **MiniLM-L6** | 128 tokens | 58 ms | 429 ms | **~7×** |
| **DistilBERT** | 128 tokens | 105 ms | 1,484 ms | **~14×** |

AutoDiff graph nodes are only created inside `GradientUtils.Grad()` scopes (used by `TrainingLoop` and manual training code). Inference passes outside `Grad()` produce leaf tensors with no computation graph overhead. The vision model gap is dominated by convolution kernels (especially depthwise convolutions in MobileNetV2), which use naive nested loops. ResNet-18 benefits from fewer depthwise layers. MiniLM is closest to parity since its attention operations map well to `TensorPrimitives` and `TensorsHelper` span-based kernels. DistilBERT (~14×) is slower than MiniLM (~7×) at the same 128-token input because it is a larger model (67.0M vs 22.7M params, 768-dim vs 384-dim) whose per-head `Slice`/`Transpose`/`MatMul`/`Softmax` attention loop and mask building dominate the forward pass — the same hot path identified for the follow-up fused multi-head attention optimization.

## Sample data

| File | Purpose |
|------|---------|
| `samples/data/mobilenet_v2/model.safetensors` | MobileNetV2 weights (~13.5 MB) |
| `samples/data/resnet18/model.safetensors` | ResNet-18 weights (~44.6 MB) |
| `samples/data/minilm/model.safetensors` | MiniLM weights (~91 MB) |
| `samples/data/minilm/config.json` | MiniLM BERT config |
| `samples/data/minilm/vocab.txt` | MiniLM wordpiece vocabulary |
| `samples/data/distilbert/model.safetensors` | DistilBERT weights (~255.5 MB, 105 tensors) |
| `samples/data/distilbert/config.json` | DistilBERT config |
| `samples/data/distilbert/vocab.txt` | DistilBERT wordpiece vocabulary |
| `samples/data/distilbert/last_hidden_state_py.bin` | PyTorch reference hidden states (generated by `Python/distilbert_compare.py`) |
| `samples/data/compare_input.bin` | Shared `[1,3,224,224]` input for compare modes (generated by `Python/generate_input.py`) |
| `samples/data/images/` | Synthetic test images at various resolutions (created by `Python/create_images.py`) |

## Nivara capabilities exercised

### Vision models

| Capability | Where exercised |
|---|---|
| `Conv2d<T>` with asymmetric padding, grouped convs, 1×1 fast path | All conv layers in both models |
| `BatchNorm2d<T>` with running statistics | Every conv → BN block |
| `MaxPool2d<T>` with argmax | ResNet-18 stem |
| `AdaptiveAvgPool2d<T>` with gradient broadcast | Both model heads |
| `Linear<T>` with MatMul + bias | Classifier heads |
| `Module<T>` tree with `LoadStateDict` | Full model construction |
| Depthwise separable convolutions (groups = channels) | MobileNetV2 3×3 blocks |

### MiniLM (text)

| Capability | Where exercised |
|---|---|
| `Embedding<T>` Gather-based lookup | Token/position/segment embeddings |
| `LayerNorm<T>` with affine parameters | After embedding, after each attention and FFN |
| `MultiheadAttention<T>` bidirectional mode, padding mask | 6 attention layers |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `ReverseGradOperations.Add` (residual) | Every residual connection |
| `Module<T>.Eval()` | Inference mode (disables dropout) |
| `Microsoft.ML.Tokenizers` integration | BERT WordPiece tokenizer |

### DistilBERT (text)

| Capability | Where exercised |
|---|---|
| `Embedding<T>` without token-type embeddings | `includeTokenTypeEmbedding: false` |
| `BertSelfAttention<T>` padding-mask path | 6 attention layers (768-dim, 12 heads) |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `DistilBertLoader.LoadEncoderWeights` | `distilbert.*` SafeTensors weight mapping |
