# NivaraInference — HuggingFace Vision Model Inference

A sample project demonstrating Nivara's ability to load pre-trained HuggingFace vision models (MobileNetV2, ResNet-18) via SafeTensors and run image classification inference. This is an **inference-only** sample — no training, no gradient computation, no backpropagation. It validates that Nivara's module system can represent real-world architectures and produce correct outputs.

**Target audience:** .NET developers evaluating Nivara for production inference workloads, ML practitioners migrating from PyTorch/HuggingFace to .NET-native inference.

## What it does

NivaraInference loads pre-trained vision models from HuggingFace Hub, reads their SafeTensors weights, reconstructs the architecture in Nivara modules, and runs inference on images to produce top-5 ImageNet predictions. It showcases:

- **SafeTensors loading** — zero-dependency binary format reader with `MemoryMarshal.Cast<byte, float>` for zero-copy F32 tensor deserialization
- **Real-world architecture reconstruction** — MobileNetV2 (inverted residuals, depthwise separable conv) and ResNet-18 (basic blocks, skip connections) built entirely from Nivara `Module<T>` primitives
- **New core operations** — `MaxPool2d<T>` and `AdaptiveAvgPool2d<T>` added to the core library
- **TF-style asymmetric padding** — `Conv2d<T>` extended with `paddingTop/paddingLeft` support for TensorFlow-weight compatibility
- **Image preprocessing pipeline** — SkiaSharp-based loading, resize, ImageNet normalization, HWC→CHW layout conversion
- **`Module<T>.LoadStateDict()`** — weight loading from SafeTensors into Nivara module trees

## Quick start

```bash
# Interactive wizard (no args)
dotnet run --project samples/NivaraInference

# MobileNetV2 inference (default)
dotnet run --project samples/NivaraInference -- --image test.jpg

# ResNet-18 inference
dotnet run --project samples/NivaraInference -- --model resnet18 --image test.jpg

# Specify model file directly
dotnet run --project samples/NivaraInference -- --model mobilenet --model-path model.safetensors --image test.jpg

# Show top-K predictions
dotnet run --project samples/NivaraInference -- --model resnet18 --image test.jpg --top-k 10

# Download model from HuggingFace
dotnet run --project samples/NivaraInference -- --download --model resnet18 --image test.jpg
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--model <name>` | `mobilenet` | Model architecture: `mobilenet` or `resnet18` |
| `--model-path <path>` | — | Path to .safetensors model file (overrides --model download) |
| `--image <path>` | — | Path to input image |
| `--top-k <int>` | 5 | Number of top predictions to show |
| `--download` | — | Download model from HuggingFace Hub |
| `--labels <path>` | — | Path to ImageNet labels file |
| `--help`, `-h` | — | Show CLI help |

## Architecture

```
Input: 224×224 RGB image (JPEG/PNG/BMP)

ImagePreprocessor (SkiaSharp)
  ├── Load → Resize 224×224 → Normalize (ImageNet mean/std) → HWC→CHW
  └── Output: float[3, 224, 224]

SafeTensorsLoader
  ├── Read .safetensors → Dictionary<string, (float[] Data, long[] Shape)>
  └── Zero-copy F32 via MemoryMarshal.Cast<byte, float>

Model Forward Pass
  ├── MobileNetV2: Stem → 16 InvertedResidual blocks → Head → Classifier → [1, 1001]
  └── ResNet-18:   Stem → 4 ResNetLayers (8 BasicBlocks) → AvgPool → FC → [1, 1000]

Output
  └── Softmax → Top-K predictions with confidence scores
```

### MobileNetV2

```
Stem:
  Conv2d(3→32, kernel=3×3, stride=2, padding=1) + BatchNorm2d + ReLU6

16 InvertedResidual Blocks:
  Block 0:  expand 3→16,  depthwise 16,  project 16→16,  stride=1
  Block 1:  expand 16→24, depthwise 24,  project 24→24,  stride=2
  ...
  Block 15: expand 160→160, depthwise 160, project 160→160, stride=1

Head:
  Conv2d(320→1280, kernel=1×1) + BatchNorm2d + ReLU6
  AdaptiveAvgPool2d(1×1)
  Dropout(0.8)
  Linear(1280→1001)
```

### ResNet-18

```
Stem:
  Conv2d(3→64, kernel=7×7, stride=2, padding=3) + BatchNorm2d + ReLU
  MaxPool2d(kernel=3×3, stride=2, padding=1)

Layer1 (2 BasicBlocks, 64ch, stride=1)
Layer2 (2 BasicBlocks, 128ch, stride=2, first block has 1×1 downsample)
Layer3 (2 BasicBlocks, 256ch, stride=2, first block has 1×1 downsample)
Layer4 (2 BasicBlocks, 512ch, stride=2, first block has 1×1 downsample)

Head:
  AdaptiveAvgPool2d(1×1)
  Linear(512→1000)
```

## What this exercises vs. other samples

| Feature | NivaraVAE | NivaraChat | **NivaraInference** |
|---|---|---|---|
| **Architecture type** | Encoder-decoder | Classifier | **Pre-trained vision (inference)** |
| **Module\<T\> inheritance** | Yes | Yes | **Yes** |
| **Conv2d\<T\>** | Yes (mode=conv) | No | **Yes (core of both models)** |
| **BatchNorm2d\<T\>** | No | No | **Yes (used throughout)** |
| **MaxPool2d\<T\>** | No | No | **Yes (new core op)** |
| **AdaptiveAvgPool2d\<T\>** | No | No | **Yes (new core op)** |
| **DepthwiseSeparableConv2d\<T\>** | No | No | **Yes (MobileNetV2)** |
| **Dropout\<T\>** | Yes | Yes | **Yes (MobileNetV2 head)** |
| **LoadStateDict** | No | Yes | **Yes (SafeTensors → Module)** |
| **TrainingLoop\<T\>** | No (manual) | Yes | **No (inference only)** |
| **DataLoader\<T\>** | No | Yes | **No (single image)** |
| **Loss function** | BCE+KL | CrossEntropy | **None (inference only)** |
| **Optimizer** | Adam | Adam | **None (inference only)** |
| **ModelSerializer** | Yes | Yes | **No (SafeTensors format instead)** |
| **Image I/O** | No | No | **Yes (SkiaSharp)** |
| **External data source** | Synthetic | External | **HuggingFace Hub** |
| **TF-style padding** | No | No | **Yes (MobileNetV2)** |

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `Module<T>` | `MobileNetV2.cs`, `ResNet18.cs` | Model base class with parameter registration |
| `Conv2d<T>` | Both models | Core convolution (with TF asymmetric padding for MobileNetV2) |
| `BatchNorm2d<T>` | Both models | Batch normalization (eval mode, uses running stats) |
| `MaxPool2d<T>` | `ResNet18.cs` | Max pooling (new core op) |
| `AdaptiveAvgPool2d<T>` | Both models | Global average pooling (new core op) |
| `DepthwiseSeparableConv2d<T>` | `MobileNetV2.cs` | Depthwise separable convolution |
| `Linear<T>` | Both models | Classification head |
| `Dropout<T>` | `MobileNetV2.cs` | Head regularization (eval mode, disabled) |
| `Activation.Relu<T>` | `ResNet18.cs` | Non-linearity |
| `Activation.Clip` (for ReLU6) | `MobileNetV2.cs` | Clipped activation: `Clip(Relu(x), 0, 6)` |
| `Module.LoadStateDict()` | `Program.cs` | Load SafeTensors weights into module tree |
| `SafeTensorsLoader` | `SafeTensorsLoader.cs` | Binary format reader (custom, zero-dependency) |
| `TensorPrimitives` | `SafeTensorsLoader.cs`, `ImagePreprocessor.cs` | SIMD-accelerated normalization |

## Files

```
samples/NivaraInference/
├── README.md                    # This file
├── NivaraInference.csproj       # Console app, net10.0
├── Program.cs                   # Entry point, CLI parsing, inference pipeline
├── SafeTensorsLoader.cs         # SafeTensors binary format reader
├── MobileNetV2.cs               # MobileNetV2 Module<float>
├── InvertedResidualBlock.cs     # InvertedResidualBlock helper
├── ResNet18.cs                  # ResNet18 Module<float>
├── BasicBlock.cs                # ResNet BasicBlock helper
├── ImagePreprocessor.cs         # SkiaSharp image loading + normalization
└── imagenet_labels.txt          # ImageNet class labels (1001 classes)
```

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- SkiaSharp (image I/O)
- No SafeTensors external dependency — custom zero-dependency reader

## Library gaps this example exposed and resolved

| Gap | Problem | Resolution |
|-----|---------|------------|
| **No `MaxPool2d<T>` module** | ResNet-18 stem uses `MaxPool2d(3×3, stride=2, padding=1)`. No pooling module existed in core. | Implemented `MaxPool2d<T>` with configurable kernel, stride, padding. Sliding-window max forward with argmax-tracked backward. File: `src/Nivara/AutoDiff/Nn/MaxPool2d.cs`. |
| **No `AdaptiveAvgPool2d<T>` module** | Both models use global average pooling (`AdaptiveAvgPool2d(1×1)`) before the classification head. No adaptive pooling existed. | Implemented `AdaptiveAvgPool2d<T>` with target output size. Forward: mean over spatial dims. Backward: gradient broadcast. File: `src/Nivara/AutoDiff/Nn/AdaptiveAvgPool2d.cs`. |
| **TF-style asymmetric padding** | MobileNetV2 HuggingFace weights use TensorFlow-style asymmetric padding (e.g., `pad_left=1, pad_right=0` for stride=2). `Conv2d<T>` only supported symmetric padding. | Extended `Conv2d<T>` to accept `paddingTop/paddingLeft` parameters (or keep symmetric via single `padding`). Updated Im2Col to handle asymmetric padding boundaries. File: `src/Nivara/AutoDiff/Nn/Conv2d.cs`. |
| **No ReLU6 activation** | MobileNetV2 uses ReLU6 (`min(max(x, 0), 6)`) throughout. No named ReLU6 existed. | Composed from existing ops: `Activation.Clip(Relu(x), 0, 6)`. Could optionally add `Activation.ReLU6<T>` convenience method. File: `src/Nivara/AutoDiff/Nn/Activation.cs`. |
| **ADR-001: null handling cleanup** | `ApplyDropout`, `ApplyPow`, `ApplyRMSNorm`, `Slice` (forward+backward), `ForwardGradOperations` mirror methods, and `GradientUtils` clipping/norm methods still contain ~350-450 lines of dead null-branching code (AutoDiff is non-nullable per ADR-001). | Removed dual-path null branching from all AutoDiff helpers. Single fast `TryGetSpan` path only. Files: `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs`, `ForwardGradOperations.cs`, `GradientUtils.cs`. |

### Core library additions from this example

| New API | Location | Purpose |
|---------|----------|---------|
| `MaxPool2d<T>` | `src/Nivara/AutoDiff/Nn/MaxPool2d.cs` | Max pooling with configurable kernel/stride/padding |
| `AdaptiveAvgPool2d<T>` | `src/Nivara/AutoDiff/Nn/AdaptiveAvgPool2d.cs` | Adaptive pooling to target output size (global avg pool when size=1) |
| `Conv2d` asymmetric padding | `src/Nivara/AutoDiff/Nn/Conv2d.cs` | TF-compatible `paddingTop/paddingLeft` parameters |

### Core library performance fixes driven by this example

| Fix | What changed | Impact |
|-----|-------------|--------|
| **ADR-001 null cleanup (inference-irrelevant paths)** | Removed ~350 lines of dead null-branching from `Dropout`, `Pow`, `RMSNorm` (forward+gradient), `Slice`, `ForwardGradOperations` mirror methods, and `GradientUtils` clipping/norm methods. | Eliminates branch mispredictions on hot paths. All AutoDiff operations now have single (null-free) SIMD-accelerated paths. |
| **TensorPrimitives in pooling kernels** | `AdaptiveAvgPool2d` forward uses `TensorPrimitives.Average` over spatial dims. `MaxPool2d` uses `Vector<T>` for small kernel sizes where vectorized max is beneficial. | SIMD-vectorized on AVX2/AVX512 hardware. |
| **SafeTensors zero-copy loading** | `MemoryMarshal.Cast<byte, float>` for F32 tensor data avoids array copies during weight loading. | Zero-copy deserialization for the dominant F32 weight format. |

## Expected output

```
=== Nivara HuggingFace Inference ===

Model: ResNet-18 (microsoft/resnet-18)
Loading weights from model.safetensors (44.2 MB, 62 tensors)...
Loading image from cat.jpg...
Preprocessing: resize 224×224, normalize, CHW layout
Running inference...

Top-5 predictions:
  1. Egyptian cat:     87.23%
  2. tabby:             5.41%
  3. tiger cat:         3.12%
  4. Persian cat:       1.87%
  5. siamese cat:       0.98%

Inference time: 65ms
```

## Performance baseline

| Metric | MobileNetV2 | ResNet-18 | Notes |
|--------|-------------|-----------|-------|
| Model load time | <500ms | <800ms | 13.4MB vs 44MB weights |
| Image preprocessing | <100ms | <100ms | Same pipeline |
| Forward pass | <50ms | <80ms | More FLOPs in ResNet |
| Total pipeline | <700ms | <1000ms | Load + preprocess + inference |
| Memory usage | ~60MB | ~180MB | Model weights + activations |

## Future work

1. **Batch inference** — process multiple images at once
2. **GELU activation** → unlock DistilBERT/transformer inference
3. **Positional encoding module** → enable BERT/GPT-2 inference
4. **Model comparison** → side-by-side accuracy/speed between MobileNetV2 and ResNet-18
5. **ONNX comparison** → validate against ONNX Runtime output
6. **Export to ONNX** → save Nivara-trained models in ONNX format
7. **BF16 support** — when .NET 11 ships with native BFloat16, update SafeTensors loader and AutoDiff kernels

## References

- [SafeTensors format specification](https://github.com/safetensors/safetensors)
- [MobileNetV2 on HuggingFace](https://huggingface.co/google/mobilenet_v2_1.0_224)
- [ResNet-18 on HuggingFace](https://huggingface.co/microsoft/resnet-18)
- [ResNet-18 config](https://huggingface.co/docs/transformers/en/model_doc/resnet)
- [Nivara AutoDiff docs](../../docs/AUTODIFF.md)
- [SafeTensors research](../../docs/SAFETENSORS.md)
