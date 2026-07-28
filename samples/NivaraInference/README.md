# Nivara HuggingFace Inference Sample

Demonstrates loading pre-trained HuggingFace vision models (MobileNetV2, ResNet-18) into Nivara's AutoDiff engine and running forward inference — all without any third-party ML framework dependencies.

Includes Python (PyTorch) reference implementations for direct performance comparison.

## What this sample does

1. **Loads SafeTensors weights** using a custom zero-dependency reader (`SafeTensorsLoader.cs`)
2. **Builds model architecture** using Nivara's `Module<T>`, `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`
3. **Runs forward inference** with synthetic random input and real images, reports top-k predictions and timings

## Supported models

| Model | Weight size | Tensors | Parameters | Output |
|-------|-------------|---------|------------|--------|
| MobileNetV2 | 13.5 MB | 262 | 3.4M | 1001 classes |
| ResNet-18 | 44.6 MB | 102 | 11.7M | 1000 classes |
| MiniLM (L6-v2) | 91 MB | 96 | 22.7M | 384-dim embedding |

Models are downloaded to `samples/data/` via HuggingFace CLI:
```bash
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download timm/resnet18.augreg_in1k --local-dir samples/data/resnet18
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir samples/data/minilm
```

## Usage

### C# (Nivara)

```bash
# Vision model: random-data inference
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18

# Vision model: benchmark 10 passes on synthetic data + real images
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- resnet18 benchmark

# Vision model: compare with Python reference
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare
dotnet run --project samples/NivaraInference -- resnet18 compare

# Vision model: step-by-step diagnostics
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare_diag
dotnet run --project samples/NivaraInference -- resnet18 compare_diag

# Vision model: single image inference
dotnet run --project samples/NivaraInference -- mobilenet_v2 path/to/image.jpg
dotnet run --project samples/NivaraInference -- resnet18 path/to/image.jpg

# MiniLM: tokenize + embed a sentence
dotnet run --project samples/NivaraInference -- minilm

# MiniLM: benchmark (10 passes)
dotnet run --project samples/NivaraInference -- minilm benchmark

# MiniLM: pairwise cosine similarity demo
dotnet run --project samples/NivaraInference -- minilm similarity
```

### Python (PyTorch reference)

```bash
cd samples/NivaraInference/Python
pip install -r requirements.txt

# Inference
python mobilenet.py
python resnet18.py

# Compare: forward pass on shared input, outputs logits for C# comparison
python mobilenet_compare.py
python resnet18_compare.py

# Diagnostics: step-by-step intermediates to samples/data/diag/
python mobilenet_diag.py
python resnet18_diag.py

# Regenerate shared comparison input
python generate_input.py
```

## Benchmark comparison

Measured on the same machine (CPU-only, no GPU acceleration):

| | PyTorch (CPU) | Nivara (.NET 10) | Slowdown |
|---|---|---|---|
| **MobileNetV2** | 38.5 ms | 3,278 ms | **~85x** |
| **ResNet-18** | 46.4 ms | 814 ms | **~17x** |

Both use batch size 1, 224×224 input, ImageNet normalization. PyTorch uses MKL-optimized BLAS kernels; Nivara uses managed .NET tensor operations.

The ResNet-18 gap is smaller because it has fewer depthwise-separable convolutions (which require groups×spatial tensor manipulation in Nivara's current implementation). MobileNetV2 is dominated by depthwise convolutions — the primary bottleneck.

## Architecture notes

### SafeTensorsLoader

Custom zero-dependency reader that parses the SafeTensors binary format directly:
- Memory-mapped header parsing via `System.Text.Json`
- Zero-copy tensor extraction using `MemoryMarshal.Cast<byte, float>`
- Throws `NotSupportedException` for non-F32 tensors (e.g., F16, BF16) with clear guidance

### MobileNetV2

- 16 inverted residual blocks with expansion/depthwise/project pattern
- Depthwise separable convolutions (groups = channels for 3×3 layers)
- ReLU6 activation via `Clip(Relu(x), 0, 6)`
- Residual skip connections only when `stride == 1 && inChannels == outChannels`

### ResNet-18

- Standard BasicBlock with 3×3 convolutions and identity/1×1 shortcut
- Stem: 7×7 conv → BN → ReLU → MaxPool2d
- 4 stages with channel progression: 64 → 128 → 256 → 512
- Global average pooling → linear classifier

### MiniLM (sentence-transformers/all-MiniLM-L6-v2)

- 6-layer **pre-norm BERT** encoder with GELU activation (not ReLU²)
- **Embedding lookup** via `ReverseGradOperations.Gather` — direct row indexing, no one-hot+MatMul waste
- **Bidirectional self-attention** with optional padding mask support
- **[CLS] token pooling** — extracts first token embedding from the output sequence
- **L2 normalization** — output embedding normalized to unit length for cosine similarity
- **384-dimensional output** sentence embeddings suitable for semantic similarity
- Tokenization via **Microsoft.ML.Tokenizers.BertTokenizer** (sample-only dependency)
- Exercises: `Embedding<T>` (Gather path), `LayerNorm<T>`, `GELU`, `MultiheadAttention<T>` (with padding mask), `Linear<T>`

### Weight loading

Each model defines a static `LoadWeights()` factory that maps HuggingFace tensor names to Nivara module parameters. No reflection or generic deserialization — explicit, type-safe loading.

**MiniLM weight mapping** (96 tensors): HuggingFace safetensors keys like `encoder.layers.N.attention.self.query.weight` are mapped to Nivara `Linear<T>` weight/bias fields via explicit lookup in `MiniLMDistilled.LoadWeights`.

## Sample images

Test images are in `samples/data/images/` — synthetic patterns at various resolutions used for benchmarking. Created by `Python/create_images.py`.

## Comparison data

Shared comparison fixtures live in `samples/data/`:

| File | Created by | Purpose |
|---|---|---|
| `compare_input.bin` | `python Python/generate_input.py` | Deterministic `[1,3,224,224]` input for `compare`/`compare_diag` modes |
| `mobilenet_v2/model.safetensors` | `hf download google/mobilenet_v2_1.0_224` | MobileNetV2 weights |
| `resnet18/model.safetensors` | `hf download timm/resnet18.augreg_in1k` | ResNet-18 weights |

## Nivara core gaps exercised (vision models)

- **Conv2d** with asymmetric padding, grouped convolutions, 1×1 fast path
- **BatchNorm2d** with running statistics for inference mode
- **MaxPool2d** with argmax tracking for backward pass
- **AdaptiveAvgPool2d** with gradient broadcast
- **Linear** with matrix multiply + bias broadcast
- **Module<T>** tree with `LoadStateDict` for parameter loading

## Nivara core gaps exercised (MiniLM)

- **Embedding<T>** Gather-based lookup (no one-hot+MatMul)
- **LayerNorm<T>** with affine parameters and configurable epsilon
- **GELU activation** — tanh approximation via `Activation.Gelu`
- **MultiheadAttention<T>** bidirectional mode with optional padding mask
- **Pre-norm transformer** architecture (LayerNorm before attention/FFN, not after)
- **[CLS] token pooling** and **L2 normalization** for sentence embeddings
- **`Microsoft.ML.Tokenizers`** integration for BERT tokenization
- **`Module<T>.Eval()`** for inference mode (disables dropout, though MiniLM has none)
