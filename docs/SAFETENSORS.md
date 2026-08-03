# SafeTensors Inference with Nivara

**Status:** Research complete, implementation planned
**Date:** 2026-07-26
**Target:** Load pre-trained HuggingFace models into Nivara and run inference

---

## Overview

Hugging Face has become the de facto hub for sharing pre-trained machine learning models. The ecosystem hosts millions of models spanning vision, language, audio, and multimodal tasks — all stored in the **SafeTensors** format by default since 2023.

**Goal:** Enable Nivara to load SafeTensors models and run inference using its existing AutoDiff engine, without training. This is purely an inference path — no gradient computation, no backpropagation, no optimizer state.

**Why this matters:**
- Validates Nivara's module system against real-world architectures
- Provides compelling samples showcasing production-like workflows
- Positions Nivara as a viable .NET-native inference runtime for small models
- Zero-dependency inference (pure managed C# via Lokad.SafeTensors + Nivara)

---

## SafeTensors Format

SafeTensors is a simple, secure binary format for storing tensor data. It was designed to replace PyTorch's pickle-based serialization with a format that is safe (no arbitrary code execution), fast (lazy-loading capable), and portable across languages.

### Binary Layout

```
┌─────────────────────────────────────────┐
│ 8 bytes: N (uint64 LE)                 │  ← Header size in bytes
├─────────────────────────────────────────┤
│ N bytes: JSON UTF-8 header              │  ← Tensor metadata
├─────────────────────────────────────────┤
│ Remaining bytes: raw tensor data        │  ← Contiguous byte buffer
└─────────────────────────────────────────┘
```

### Header Structure

The JSON header is a dictionary mapping tensor names to their metadata:

```json
{
  "__metadata__": { "format": "pt" },
  "features.0.0.weight": {
    "dtype": "F32",
    "shape": [32, 3, 3, 3],
    "data_offsets": [0, 3456]
  },
  "features.0.1.weight": {
    "dtype": "F32",
    "shape": [32],
    "data_offsets": [3456, 3488]
  }
}
```

Key properties:
- **Tensor name** — hierarchical naming convention: `layer.sublayer.parameter_type` (e.g., `features.0.0.weight`)
- **dtype** — data type tag (`F32`, `F16`, `BF16`, `I32`, `I64`, `U8`, `BOOL`, etc.)
- **shape** — dimension sizes as integer array
- **data_offsets** — `[begin, end)` byte offsets relative to the start of the byte buffer
- **`__metadata__`** — optional string-to-string map (e.g., `{"format": "pt"}` for PyTorch origin)

### Supported Data Types

| Tag | Type | Bytes | Notes |
|-----|------|-------|-------|
| F32 | Single-precision float | 4 | **Most common for small models** |
| F16 | Half-precision float | 2 | Common for medium models |
| BF16 | Brain float 16 | 2 | **Dominant for large models (LLMs)** |
| F64 | Double-precision float | 8 | Rare in ML |
| I32 | Signed 32-bit int | 4 | Token IDs, labels |
| I64 | Signed 64-bit int | 8 | Some tokenizer metadata |
| I16 | Signed 16-bit int | 2 | Rare |
| I8 | Signed 8-bit int | 1 | Quantized weights |
| U8 | Unsigned 8-bit int | 1 | Quantized weights |
| BOOL | Boolean | 1 | Attention masks |
| F8_E4M3 | FP8 (4-bit exp) | 1 | Quantized (modern) |
| F8_E5M2 | FP8 (5-bit exp) | 1 | Quantized (modern) |

### Sharded Models

Large models split weights across multiple files:

```
model-00001-of-00072.safetensors
model-00002-of-00072.safetensors
...
model-00072-of-00072.safetensors
model.safetensors.index.json          ← weight_map: tensor_name → filename
```

The `model.safetensors.index.json` contains:

```json
{
  "metadata": { "total_size": 352494542848 },
  "weight_map": {
    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00072.safetensors",
    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00072.safetensors",
    ...
  }
}
```

### Format Properties

- **Endianness:** Little-endian
- **Memory order:** Row-major (C-order)
- **Zero-copy:** Tensor data can be memory-mapped directly from disk
- **Security:** No executable code — pure data serialization
- **Validation:** Byte buffer must be fully indexed (no holes)

---

## .NET Ecosystem

### SafeTensors Readers

| Package | Version | Dependencies | Target | Notes |
|---------|---------|--------------|--------|-------|
| **Lokad.SafeTensors** | 0.1.0 | None | .NET 8, .NET 10 | **Recommended** — pure managed, zero deps |
| Cortex.SafeTensors | 1.1.0 | Cortex framework | — | Part of Cortex ML framework |
| Onnxify.Safetensors | 0.3.9 | Onnxify ecosystem | .NET 8, .NET 10 | Part of ONNX bridge |

**Recommendation:** Use `Lokad.SafeTensors`. It is standalone, has zero external dependencies, targets .NET 10, and provides a clean API:

```csharp
using Lokad.SafeTensors;

// Header-only inspection (no tensor data loaded)
var header = SafeTensorSerializer.ReadHeader("model.safetensors");
foreach (var tensor in header.Tensors)
    Console.WriteLine($"{tensor.Name} {tensor.Dtype} [{string.Join(", ", tensor.Shape)}]");

// Full file load
var file = SafeTensorSerializer.ReadFile("model.safetensors");
var weights = file.GetTensor("features.0.0.weight");
ReadOnlyMemory<byte> data = weights.GetBytes();

// In-memory deserialization
var file2 = SafeTensorSerializer.Deserialize(byteArray);
```

### HuggingFace .NET Ecosystem

The broader .NET ML ecosystem offers several approaches:

| Approach | Pros | Cons |
|----------|------|------|
| **ONNX Runtime** (Microsoft.ML.OnnxRuntime) | Mature, GPU support, broad model coverage | Requires ONNX conversion, native deps |
| **TorchSharp** | PyTorch-compatible API, SafeTensors via PyBridge | Requires native libtorch |
| **SentenceTransformers-Sharp** | Pure managed inference for specific models | Limited to pre-packaged models |
| **Nivara + Lokad.SafeTensors** | Zero native deps, full control, columnar storage | Requires manual architecture reconstruction |

Nivara's approach is unique: reconstruct the model architecture using Nivara's module system, load weights from SafeTensors, and run inference on the AutoDiff engine. This gives full control over the computation graph and enables custom inference pipelines.

---

## BF16 Strategy

### Current State (.NET 10)

- `System.Numerics.Half` (F16) exists and is fully supported
- `System.Numerics.BFloat16` does **not** exist in .NET 10
- SafeTensors files containing BF16 weights can be read (raw bytes accessible), but no native conversion

### .NET 11 (Preview, ~2026)

- `System.Numerics.BFloat16` merged into the runtime (PR #98643, Oct 2025)
- Implements `IBinaryFloatingPointIeee754` — full math operations, conversion operators
- Explicit conversion: `(BFloat16)float_value` and `(float)bfloat16_value`
- AVX-512 BF16 intrinsics also coming (PR #129326) for hardware-accelerated operations

### Nivara's BF16 Migration Plan

| Phase | .NET Version | BF16 Support | Action |
|-------|-------------|--------------|--------|
| **Now** | .NET 10 | None | Target F32 models only; reject BF16 with helpful error |
| **Phase 2** | .NET 10 | Manual | Add manual BF16→F32 conversion (reinterpret bits, ~20 lines) for testing |
| **Phase 3** | .NET 11 | Native | Migrate to `System.Numerics.BFloat16`; auto-convert BF16→F32 at load time |
| **Phase 4** | .NET 11 | Native | Add `BFloat16` to `TypeValidator`, `ColumnStorageFactory`, `ColumnStorage<BFloat16>` |

**Why BF16 matters:** Most modern HuggingFace models (LLaMA, Mistral, Gemma, Bloom) ship weights in BF16. F32-only limits us to smaller/older models initially. Native BF16 on .NET 11 unlocks the entire HuggingFace model catalog.

---

## Model Feasibility Analysis

### What Nivara Already Has

| Operation | Module | Status |
|-----------|--------|--------|
| 2D Convolution | `Conv2d<T>` | ✓ Implemented with im2col kernel |
| 1D Convolution | `Conv1d<T>` | ✓ Implemented |
| Depthwise Separable Conv | `DepthwiseSeparableConv2d<T>` | ✓ MobileNet-style |
| Batch Normalization | `BatchNorm1d<T>`, `BatchNorm2d<T>` | ✓ Fused span kernel |
| Layer Normalization | `LayerNorm<T>` | ✓ With affine support |
| RMS Normalization | `RMSNorm` ops | ✓ Per-row and flat |
| Linear (Dense) | `Linear<T>` | ✓ Kaiming init |
| Embedding | `Embedding<T>` | ✓ Lookup embedding |
| Dropout | `Dropout<T>` | ✓ Inverted dropout |
| ReLU | `Activation.Relu` | ✓ With gradient |
| Leaky ReLU | `Activation.LeakyRelu` | ✓ Default slope 0.01 |
| Sigmoid | `Activation.Sigmoid` | ✓ With gradient |
| Tanh | `Activation.Tanh` | ✓ With gradient |
| Softmax | `Softmax<T>` | ✓ Last-dim aware |
| Matrix Multiply | `MatMul` ops | ✓ TensorPrimitives.Dot |
| Sequential | `Sequential<T>` | ✓ Ordered module chain |
| Module System | `Module<T>` | ✓ Parameters, StateDict, Train/Eval |
| Weight Loading | `LoadStateDict()` | ✓ Shape-validated loading |

### Tier 1 — Ready Now (Zero Missing Ops)

Models whose architectures map directly to existing Nivara modules.

#### MobileNetV2 (Recommended First Target)

| Property | Value |
|----------|-------|
| HuggingFace model | `google/mobilenet_v2_1.0_224` |
| Parameters | ~3.4M |
| Weight size (F32) | ~13.4MB |
| Input | 224×224 RGB image (CHW, normalized) |
| Output | 1001 ImageNet classes |
| Architecture | Conv2d → BN → ReLU6 → InvertedResidual blocks → AdaptiveAvgPool → Linear |

**Architecture breakdown:**

```
Stem:
  Conv2d(3, 32, 3×3, stride=2) + BatchNorm2d + ReLU6

16 InvertedResidual Blocks (each):
  1×1 Conv2d (expand) + BN + ReLU6
  3×3 DepthwiseConv2d + BN + ReLU6
  1×1 Conv2d (project) + BN  ← linear bottleneck (no activation)

Head:
  Conv2d(320, 1280, 1×1) + BN + ReLU6
  AdaptiveAvgPool2d(1×1)
  Dropout(0.8) + Linear(1280, 1001)
```

**Why MobileNetV2 is ideal:**
- Uses `DepthwiseSeparableConv2d` which Nivara already implements
- ReLU6 = `Clip(Relu(x), 0, 6)` using existing ops
- Small model (3.4MB) — loads instantly, fast inference
- Single-file SafeTensors (no sharding)
- F32 weights available (no BF16 dependency)
- Compelling visual demo: image in → top-5 predictions out

#### MobileNetV3-Small

| Property | Value |
|----------|-------|
| HuggingFace model | `google/mobilenet_v3_small_1.0_224` |
| Parameters | ~2.5M |
| Weight size (F32) | ~10MB |
| Architecture | Similar to V2 + HardSwish, Squeeze-and-Excitation |

HardSwish can be faked: `x × ReLU6(x+3) / 6` using existing ops. Squeeze-and-Excitation uses AdaptiveAvgPool + Linear + Sigmoid — all exist.

#### ResNet-18

| Property | Value |
|----------|-------|
| HuggingFace model | `microsoft/resnet-18` |
| Parameters | ~11.7M |
| Weight size (F32) | ~44MB |
| Input | 224×224 RGB image (CHW, normalized) |
| Output | 1000 ImageNet classes |
| Architecture | Conv2d → BN → ReLU → MaxPool → BasicBlocks (Conv2d+BN+ReLU+skip) → AdaptiveAvgPool → Linear |

**Architecture breakdown:**

```
Stem:
  Conv2d(3→64, 7×7, stride=2, padding=3) + BatchNorm2d + ReLU
  MaxPool2d(3×3, stride=2, padding=1)

4 Residual Layers (each with 2 BasicBlocks):
  Layer1: 2× BasicBlock(64→64, stride=1)     — no downsample
  Layer2: 2× BasicBlock(64→128, stride=2)    — 1×1Conv downsample on first block
  Layer3: 2× BasicBlock(128→256, stride=2)   — 1×1Conv downsample on first block
  Layer4: 2× BasicBlock(256→512, stride=2)   — 1×1Conv downsample on first block

Head:
  AdaptiveAvgPool2d(1×1)                      — global average pooling
  Linear(512→1000)
```

**Missing core operations:**
- **MaxPool2d** — used in stem for 3×3 stride=2 pooling. Forward: sliding window max. Backward: argmax routing.
- **AdaptiveAvgPool2d** — used for global average pooling. Forward: mean over spatial dims. Backward: gradient broadcast.

Both are small, self-contained additions (~150 and ~120 lines respectively) following existing `Conv2d<T>` patterns.

**Why ResNet-18 is important:**
- Showcases Conv2d + BatchNorm2d + ReLU — the core v1.1 NN layer additions
- Residual connections demonstrate skip-connection patterns
- Canonical vision architecture recognized by the ML community
- Complements MobileNetV2 as a "standard vs efficient" comparison
- Single-file SafeTensors, F32 weights, no BF16 dependency
- ~44MB model is practical for demo purposes

### Tier 2 — Moderate Work (1-3 New Ops Required)

#### DistilBERT (Base Uncased)

| Property | Value |
|----------|-------|
| HuggingFace model | `distilbert/distilbert-base-uncased` |
| Parameters | ~67M |
| Weight size (F32) | ~260MB |
| Architecture | Embedding → 6 TransformerBlocks → Linear |

**Missing operations:**
- **GELU activation** — used in transformer FFN layers. Formula: `0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`
- **Sinusoidal positional encoding** — added to token embeddings. Formula based on sin/cos of position frequencies.

**Missing modules:**
- **Positional encoding module** — not in Nivara's module system
- **Multi-head attention with causal mask** — exists in `TransformerBlock`, but current implementation doesn't support encoder-style (non-causal) attention

**Effort estimate:** ~2-3 days for GELU + positional encoding + DistilBERT reconstruction.

#### BERT-base

| Property | Value |
|----------|-------|
| HuggingFace model | `bert-base-uncased` |
| Parameters | ~110M |
| Weight size (F32) | ~440MB |
| Architecture | Embedding → 12 TransformerBlocks (with cross-attention) → Linear |

Same gaps as DistilBERT plus cross-attention support in TransformerBlock.

#### GPT-2 (Small)

| Property | Value |
|----------|-------|
| HuggingFace model | `openai-community/gpt2` |
| Parameters | ~124M |
| Weight size (F32) | ~500MB |
| Architecture | Embedding → 12 CausalTransformerBlocks → Linear |

**Missing operations:**
- **GELU** — same as DistilBERT
- **Learned positional encoding** — simpler than sinusoidal, just an embedding lookup

GPT-2 uses the existing `TransformerBlock` (causal self-attention), so the architecture is closer to what Nivara already supports.

### Tier 3 — Significant Work

#### LLaMA / Mistral / Gemma

| Property | Value |
|----------|-------|
| Parameters | 7B-70B |
| Weight size (F32) | 28GB-280GB |
| Architecture | Embedding → RoPE attention → SwiGLU FFN → RMSNorm → Linear |

**Missing operations:**
- **SwiGLU** — gated linear unit FFN: `SiLU(W1·x) ⊙ (W3·x)` — dominant in modern LLMs
- **RoPE (Rotary Position Embedding)** — position-dependent rotation of Q/K vectors
- **KV-cache** — essential for efficient autoregressive decoding
- **BF16** — all modern LLMs ship BF16 weights (28GB+ in F32 is impractical)

**Effort estimate:** ~2-3 weeks. Not recommended until BF16 support lands in .NET 11.

---

## Gap Analysis

### Missing Operations (Prioritized)

| Operation | Needed By | Priority | Effort |
|-----------|-----------|----------|--------|
| **MaxPool2d** | ResNet-18 stem pooling | **High (v1.1)** | 1 day |
| **AdaptiveAvgPool2d** | ResNet-18, MobileNetV2 global pooling | **High (v1.1)** | 0.5 day |
| **GELU** | BERT, GPT-2, T5, DistilBERT | High | 0.5 day |
| **SiLU/Swish** | LLaMA, Mistral, Gemma | Medium | 0.5 day |
| **SwiGLU** (fused gate) | Modern LLM FFN layers | Medium | 1 day |
| **Sinusoidal positional encoding** | BERT, DistilBERT | High | 0.5 day |
| **Learned positional encoding** | GPT-2 | High | 0.25 day |
| **RoPE** | LLaMA, Mistral, Gemma | Low | 1 day |
| **KV-cache** | Autoregressive decoding | Low | 2 days |
| **Softmax dim parameter** | General transformer inference | Low | 0.5 day |

**Note:** MaxPool2d and AdaptiveAvgPool2d are required for ResNet-18 inference and are aligned with the Nivara v1.1 vision of comprehensive CNN/NN layer support. Both follow existing `Conv2d<T>` patterns and are small, self-contained additions.

### Existing Modules (Sufficient)

All CNN operations are fully implemented:
- `Conv2d`, `Conv1d`, `DepthwiseSeparableConv2d`
- `BatchNorm1d`, `BatchNorm2d`
- `LayerNorm`, `RMSNorm`
- `Linear`, `Embedding`, `Sequential`
- `Dropout`, all core activations

**New for v1.1 (planned):**
- `MaxPool2d` — for ResNet-18 stem pooling
- `AdaptiveAvgPool2d` — for global average pooling in both ResNet-18 and MobileNetV2

### Module System Gaps

| Gap | Detail | Priority |
|-----|--------|----------|
| **No MaxPool2d module** | Needed for ResNet-18 stem. Forward: sliding window max. Backward: argmax routing. | **High (v1.1)** |
| **No AdaptiveAvgPool2d module** | Needed for global average pooling. Forward: mean over spatial dims. Backward: gradient broadcast. | **High (v1.1)** |
| No positional encoding module | Sinusoidal and learned PE not implemented | High |
| TransformerBlock only supports causal attention | No encoder-style (non-causal) attention | Medium |
| TransformerBlock hardcodes affine=false for LayerNorm | No learnable LN params in the block | Medium |
| No GELU/SiLU activation wrapper | Need to add to `Activation<T>` | High |

---

## HuggingFace Weight Name Conventions

Understanding the mapping between PyTorch state dict keys and Nivara module paths is critical for weight loading.

### PyTorch Naming Patterns

| Model Type | Pattern | Example |
|------------|---------|---------|
| MobileNetV2 | `features.{block}.{layer}.{param}` | `features.0.0.weight` |
| BERT | `bert.embeddings.{name}.{param}` | `bert.embeddings.word_embeddings.weight` |
| BERT | `bert.encoder.layer.{i}.{name}.{param}` | `bert.encoder.layer.0.attention.self.query.weight` |
| GPT-2 | `transformer.h.{i}.{name}.{param}` | `transformer.h.0.attn.c_attn.weight` |
| DistilBERT | `distilbert.transformer.layer.{i}.{name}.{param}` | `distilbert.transformer.layer.0.attention.q_lin.weight` |

### Parameter Types

| Suffix | Meaning | Shape Convention |
|--------|---------|-----------------|
| `.weight` | Weight matrix (learnable) | `[outFeatures, inFeatures]` for Linear; `[outChannels, inChannels, kH, kW]` for Conv2d |
| `.bias` | Bias vector (learnable) | `[outFeatures]` or `[outChannels]` |
| `.running_mean` | BatchNorm running mean | `[numFeatures]` |
| `.running_var` | BatchNorm running variance | `[numFeatures]` |
| `.num_batches_tracked` | BatchNorm batch counter | scalar (I64) |
| `.weight` (LayerNorm) | Scale parameter | `[normalizedShape]` |
| `.bias` (LayerNorm) | Shift parameter | `[normalizedShape]` |

### Key Transposition

**Important:** PyTorch stores Linear weights as `[outFeatures, inFeatures]` (row-major), while Nivara's `Linear<T>` expects the same convention. The `Forward` method transposes internally: `y = x @ Wᵀ + b`. Weight loading should **not** transpose — just copy directly.

---

## Implementation Strategy

### Phase 0: Core Library — MaxPool2d + AdaptiveAvgPool2d

**Priority:** High (required for ResNet-18, aligned with Nivara v1.1)

- Implement `MaxPool2d<T>` with configurable kernel, stride, padding
  - Forward: sliding window max over each kernel position
  - Backward: route gradient to argmax position in each window
  - ~150 lines including forward/backward kernels
- Implement `AdaptiveAvgPool2d<T>` with target output size
  - Forward: mean over spatial dimensions
  - Backward: gradient broadcast to all input positions
  - ~120 lines, simpler than MaxPool2d
- Add unit tests for both (shape validation, gradient correctness)
- Document in `docs/AUTODIFF.md`

### Phase 1: SafeTensors Reader (Sample Level)

- Add `Lokad.SafeTensors` NuGet package to the sample project
- Implement `SafeTensorsLoader` class:
  - `Read(string path)` → `Dictionary<string, float[]>`
  - Validate all tensors are F32 (reject BF16 with helpful message)
  - Convert raw bytes to `float[]` via `BitConverter` or `MemoryMarshal`
  - Support single-file models (no sharding initially)

### Phase 2: MobileNetV2 Reconstruction

- Implement `MobileNetV2<T>` as `Module<T>` using existing Nivara modules
- Map HuggingFace weight names to Nivara module tree
- Load weights via `LoadStateDict()` pattern
- Validate forward pass produces correct output shape

### Phase 3: ResNet-18 Reconstruction

- Implement `ResNet18<T>` as `Module<T>` using new MaxPool2d + AdaptiveAvgPool2d
- Implement `BasicBlock` with residual connections and optional 1×1Conv downsample
- Map HuggingFace weight names (layer1-4 naming convention)
- Load weights via `LoadStateDict()` pattern
- Validate forward pass produces correct output shape

### Phase 4: Image Preprocessing

- Use **SkiaSharp** for image loading, resize, and format conversion
- Implement ImageNet normalization (mean/std per channel)
- Convert HWC → CHW tensor layout
- Resize to 224×224 with bicubic interpolation

### Phase 5: Demo & Validation

- Support model selection via CLI (`--model mobilenet|resnet18`)
- Load pre-trained weights from HuggingFace Hub
- Run inference on sample images (cat, dog, car, etc.)
- Display top-5 predictions with confidence scores
- Compare with HuggingFace Python output for validation

### Future Phases

- **Phase 6:** Add GELU + positional encoding → DistilBERT inference
- **Phase 7:** Add SiLU + SwiGLU → GPT-2 inference
- **Phase 8:** Migrate to .NET 11 for BF16 support → LLaMA-class models

---

## References

### SafeTensors
- [SafeTensors format specification](https://github.com/safetensors/safetensors)
- [SafeTensors metadata parsing (HuggingFace docs)](https://huggingface.co/docs/safetensors/en/metadata_parsing)
- [Lokad.SafeTensors (NuGet)](https://www.nuget.org/packages/Lokad.SafeTensors)

### HuggingFace Models
- [MobileNetV2 (HuggingFace docs)](https://huggingface.co/docs/transformers/en/model_doc/mobilenet_v2)
- [MobileNetV2 config (PyTorch source)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilenet_v2/configuration_mobilenet_v2.py)
- [MobileNetV2 model (PyTorch source)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilenet_v2/modeling_mobilenet_v2.py)
- [ResNet-18 on HuggingFace](https://huggingface.co/microsoft/resnet-18)
- [ResNet docs (HuggingFace)](https://huggingface.co/docs/transformers/en/model_doc/resnet)

### .NET BFloat16
- [BFloat16 PR #98643 (merged into .NET 11)](https://github.com/dotnet/runtime/pull/98643)
- [BFloat16 API reference (.NET 11)](https://learn.microsoft.com/en-us/dotnet/api/system.numerics.bfloat16?view=net-11.0)
- [AVX-512 BF16 intrinsics PR #129326](https://github.com/dotnet/runtime/pull/129326)

### Nivara AutoDiff
- [Nivara AutoDiff documentation](./AUTODIFF.md)
- [NivaraTensorExtensions](../src/Nivara/Tensors/NivaraTensorExtensions.cs)
- [TransformerBlock module](../src/Nivara/AutoDiff/Nn/TransformerBlock.cs)
- [DepthwiseSeparableConv2d module](../src/Nivara/AutoDiff/Nn/DepthwiseSeparableConv2d.cs)
