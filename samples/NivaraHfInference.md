# NivaraHfInference — Implementation Plan

**Status:** Planned
**Created:** 2026-07-26
**Goal:** Load pre-trained HuggingFace models (MobileNetV2, ResNet-18) and run image classification inference using Nivara's AutoDiff engine.

> This is a temporary implementation plan. It will be removed once the sample is complete.

---

## Overview

Load pre-trained vision models from HuggingFace Hub, read their SafeTensors weights, reconstruct the architecture in Nivara modules, and run inference on images to produce top-5 ImageNet predictions.

**Supported models:**

| Model | HuggingFace ID | Parameters | Weight Size | Core Ops Needed |
|-------|---------------|------------|-------------|-----------------|
| **MobileNetV2** | `google/mobilenet_v2_1.0_224` | ~3.4M | ~13.4MB | All exist ✓ |
| **ResNet-18** | `microsoft/resnet-18` | ~11.7M | ~44MB | MaxPool2d, AdaptiveAvgPool2d (new) |

This is an **inference-only** sample — no training, no gradient computation, no backpropagation. It validates that Nivara's module system can represent real-world architectures and produce correct outputs.

---

## Model 1: MobileNetV2

### Architecture Mapping

| HuggingFace Component | Nivara Module | Status |
|----------------------|---------------|--------|
| `nn.Conv2d` | `Conv2d<T>` | ✓ Exists |
| `nn.BatchNorm2d` | `BatchNorm2d<T>` | ✓ Exists |
| `nn.ReLU6` | `Clip(Relu(x), 0, 6)` | ✓ Composable from existing ops |
| `nn.AdaptiveAvgPool2d(1)` | Global mean pool (manual) | ✓ Composable |
| `nn.Linear` | `Linear<T>` | ✓ Exists |
| `nn.Dropout` | `Dropout<T>` | ✓ Exists |
| Depthwise Separable Conv | `DepthwiseSeparableConv2d<T>` | ✓ Exists |

**No new core operations required.** All building blocks exist.

### Architecture

```
Stem:
  Conv2d(3→32, kernel=3×3, stride=2, padding=1) + BatchNorm2d + ReLU6

16 InvertedResidual Blocks:
  Block 0:  expand 3→16,  depthwise 16,  project 16→16,  stride=1
  Block 1:  expand 16→24, depthwise 24,  project 24→24,  stride=2
  Block 2:  expand 24→24, depthwise 24,  project 24→24,  stride=1
  Block 3:  expand 24→32, depthwise 32,  project 32→32,  stride=2
  Block 4:  expand 32→32, depthwise 32,  project 32→32,  stride=1
  Block 5:  expand 32→32, depthwise 32,  project 32→32,  stride=1
  Block 6:  expand 32→64, depthwise 64,  project 64→64,  stride=2
  Block 7:  expand 64→64, depthwise 64,  project 64→64,  stride=1
  Block 8:  expand 64→64, depthwise 64,  project 64→64,  stride=1
  Block 9:  expand 64→64, depthwise 64,  project 64→64,  stride=1
  Block 10: expand 64→96,  depthwise 96,  project 96→96,  stride=1
  Block 11: expand 96→96,  depthwise 96,  project 96→96,  stride=1
  Block 12: expand 96→96,  depthwise 96,  project 96→96,  stride=1
  Block 13: expand 96→160, depthwise 160, project 160→160, stride=2
  Block 14: expand 160→160, depthwise 160, project 160→160, stride=1
  Block 15: expand 160→160, depthwise 160, project 160→160, stride=1

Head:
  Conv2d(320→1280, kernel=1×1) + BatchNorm2d + ReLU6
  AdaptiveAvgPool2d(1×1)
  Dropout(0.8)
  Linear(1280→1001)
```

### InvertedResidual Block

```
Input ────────────────────────────────┐
                                      │ (skip connection if stride=1 and inChannels==outChannels)
                                      │
1×1 Conv2d(in→expand, stride=1)       │
BatchNorm2d(expand)                   │
ReLU6                                 │
                                      │
3×3 DepthwiseConv2d(expand, stride)   │
BatchNorm2d(expand)                   │
ReLU6                                 │
                                      │
1×1 Conv2d(expand→out, stride=1)      │  ← linear bottleneck (no activation)
BatchNorm2d(out)                      │
                                      │
              + ──────────────────────┘
              v
           Output
```

### Weight Name Mapping (MobileNetV2)

| PyTorch Key Pattern | Nivara Module Path |
|--------------------|--------------------|
| `features.0.0.weight` | `Stem.Conv.Weight` |
| `features.0.1.weight` | `Stem.Norm.Weight` |
| `features.0.1.bias` | `Stem.Norm.Bias` |
| `features.0.1.running_mean` | `Stem.Norm.RunningMean` |
| `features.0.1.running_var` | `Stem.Norm.RunningVar` |
| `features.{i}.conv.0.weight` | `Block_{i}.ExpandConv.Weight` |
| `features.{i}.conv.1.weight` | `Block_{i}.ExpandNorm.Weight` |
| `features.{i}.conv.3.weight` | `Block_{i}.DepthwiseConv.Weight` |
| `features.{i}.conv.4.weight` | `Block_{i}.DepthwiseNorm.Weight` |
| `features.{i}.conv.5.weight` | `Block_{i}.PointwiseConv.Weight` |
| `features.{i}.conv.6.weight` | `Block_{i}.PointwiseNorm.Weight` |
| `features.16.0.weight` | `Head.Conv.Weight` |
| `features.16.1.weight` | `Head.Norm.Weight` |
| `classifier.1.weight` | `Classifier.Weight` |
| `classifier.1.bias` | `Classifier.Bias` |

---

## Model 2: ResNet-18

### Architecture Mapping

| HuggingFace Component | Nivara Module | Status |
|----------------------|---------------|--------|
| `nn.Conv2d` | `Conv2d<T>` | ✓ Exists |
| `nn.BatchNorm2d` | `BatchNorm2d<T>` | ✓ Exists |
| `nn.ReLU` | `Activation.Relu` | ✓ Exists |
| `nn.MaxPool2d` | `MaxPool2d<T>` | **⚠ New — must add to core** |
| `nn.AdaptiveAvgPool2d` | `AdaptiveAvgPool2d<T>` | **⚠ New — must add to core** |
| `nn.Linear` | `Linear<T>` | ✓ Exists |

### Architecture

```
Input: 224×224 RGB image

Stem:
  Conv2d(3→64, kernel=7×7, stride=2, padding=3) + BatchNorm2d + ReLU
  MaxPool2d(kernel=3×3, stride=2, padding=1)

Layer1 (2 BasicBlocks, 64 channels, stride=1):
  BasicBlock(64→64, stride=1): Conv3×3→BN→ReLU→Conv3×3→BN + skip→ReLU

Layer2 (2 BasicBlocks, 128 channels, stride=2):
  BasicBlock(64→128, stride=2): Conv3×3(stride=2)→BN→ReLU→Conv3×3→BN + 1×1Conv(downsample)→BN→ReLU
  BasicBlock(128→128, stride=1): Conv3×3→BN→ReLU→Conv3×3→BN + skip→ReLU

Layer3 (2 BasicBlocks, 256 channels, stride=2):
  BasicBlock(128→256, stride=2): Conv3×3(stride=2)→BN→ReLU→Conv3×3→BN + 1×1Conv(downsample)→BN→ReLU
  BasicBlock(256→256, stride=1): Conv3×3→BN→ReLU→Conv3×3→BN + skip→ReLU

Layer4 (2 BasicBlocks, 512 channels, stride=2):
  BasicBlock(256→512, stride=2): Conv3×3(stride=2)→BN→ReLU→Conv3×3→BN + 1×1Conv(downsample)→BN→ReLU
  BasicBlock(512→512, stride=1): Conv3×3→BN→ReLU→Conv3×3→BN + skip→ReLU

Head:
  AdaptiveAvgPool2d(1×1)          ← global average pooling
  Linear(512→1000)
```

### BasicBlock

```
Input ────────────────────────────────┐
                                      │
Conv2d(in→out, 3×3, stride)          │
BatchNorm2d(out)                      │
ReLU                                  │
                                      │
Conv2d(out→out, 3×3, stride=1)       │
BatchNorm2d(out)                      │
                                      │
              + ──────────────────────┘  (skip: identity or 1×1Conv if dims change)
              │
              v
              ReLU
              v
           Output
```

### Weight Name Mapping (ResNet-18)

| PyTorch Key Pattern | Nivara Module Path |
|--------------------|--------------------|
| `conv1.weight` | `Stem.Conv.Weight` |
| `bn1.weight` | `Stem.Norm.Weight` |
| `bn1.bias` | `Stem.Norm.Bias` |
| `bn1.running_mean` | `Stem.Norm.RunningMean` |
| `bn1.running_var` | `Stem.Norm.RunningVar` |
| `layer{i}.{j}.conv1.weight` | `Layer{i}.Block{j}.Conv1.Weight` |
| `layer{i}.{j}.bn1.weight` | `Layer{i}.Block{j}.Norm1.Weight` |
| `layer{i}.{j}.bn1.bias` | `Layer{i}.Block{j}.Norm1.Bias` |
| `layer{i}.{j}.bn1.running_mean` | `Layer{i}.Block{j}.Norm1.RunningMean` |
| `layer{i}.{j}.bn1.running_var` | `Layer{i}.Block{j}.Norm1.RunningVar` |
| `layer{i}.{j}.conv2.weight` | `Layer{i}.Block{j}.Conv2.Weight` |
| `layer{i}.{j}.bn2.weight` | `Layer{i}.Block{j}.Norm2.Weight` |
| `layer{i}.{j}.bn2.bias` | `Layer{i}.Block{j}.Norm2.Bias` |
| `layer{i}.{j}.bn2.running_mean` | `Layer{i}.Block{j}.Norm2.RunningMean` |
| `layer{i}.{j}.bn2.running_var` | `Layer{i}.Block{j}.Norm2.RunningVar` |
| `layer{i}.{j}.downsample.0.weight` | `Layer{i}.Block{j}.Downsample.Conv.Weight` |
| `layer{i}.{j}.downsample.1.weight` | `Layer{i}.Block{j}.Downsample.Norm.Weight` |
| `layer{i}.{j}.downsample.1.bias` | `Layer{i}.Block{j}.Downsample.Norm.Bias` |
| `layer{i}.{j}.downsample.1.running_mean` | `Layer{i}.Block{j}.Downsample.Norm.RunningMean` |
| `layer{i}.{j}.downsample.1.running_var` | `Layer{i}.Block{j}.Downsample.Norm.RunningVar` |
| `fc.weight` | `Classifier.Weight` |
| `fc.bias` | `Classifier.Bias` |

**ResNet-18 block counts:** Layer1=2 blocks, Layer2=2 blocks, Layer3=2 blocks, Layer4=2 blocks (total 8 BasicBlocks).

**Downsample layers:** Only the first block of Layer2, Layer3, and Layer4 has a downsample path (1×1 Conv + BN) to match channel dimensions. Layer1 has no downsample (64→64).

---

## Core Library Changes Required

ResNet-18 requires two new operations in the core Nivara library. These are small, self-contained additions.

### 1. MaxPool2d\<T\>

**File:** `src/Nivara/AutoDiff/Nn/MaxPool2d.cs`

```csharp
public sealed class MaxPool2d<T> : Module<T> where T : struct, INumber<T>
{
    // Parameters
    int KernelSize;     // e.g., 3
    int Stride;         // e.g., 2 (defaults to KernelSize if not specified)
    int Padding;        // e.g., 1
    
    // Forward: input [N, C, H, W] → output [N, C, oH, oW]
    // where oH = (H + 2*padding - kernelSize) / stride + 1
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input);
}
```

**Forward kernel:** Sliding window max over each kernel position. Track argmax positions for backward pass.

**Backward kernel:** Route gradient only to the argmax position in each window (sparse gradient assignment).

**Estimated size:** ~150 lines including forward/backward kernels.

### 2. AdaptiveAvgPool2d\<T\>

**File:** `src/Nivara/AutoDiff/Nn/AdaptiveAvgPool2d.cs`

```csharp
public sealed class AdaptiveAvgPool2d<T> : Module<T> where T : struct, INumber<T>
{
    // Parameters
    int OutputSize;     // target spatial size (e.g., 1 for global average pooling)
    
    // Forward: input [N, C, H, W] → output [N, C, OutputSize, OutputSize]
    // When OutputSize=1: global average pooling → [N, C, 1, 1]
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input);
}
```

**Forward kernel:** For each output position, compute the average over the corresponding input region. When `OutputSize=1`, this is simply the mean of all spatial positions per channel.

**Backward kernel:** Distribute gradient equally to all input positions that contributed to the output average. Gradient is `gradOutput / (H * W)` spread across all spatial positions.

**Estimated size:** ~120 lines. Simpler than MaxPool2d (no argmax tracking).

### Why These Are Small

- Both operations are standard CNN building blocks with well-known forward/backward rules
- They follow the same patterns as existing `Conv2d<T>` (2D spatial ops with tiled execution)
- The backward pass for MaxPool2d is just sparse gradient assignment (no computation)
- The backward pass for AdaptiveAvgPool2d is gradient broadcasting (simple)

---

## Implementation Steps

### Phase 1: Core Library — MaxPool2d + AdaptiveAvgPool2d

**Priority:** High (required for ResNet-18)

1. Implement `MaxPool2d<T>` with configurable kernel, stride, padding
2. Implement `AdaptiveAvgPool2d<T>` with target output size
3. Add unit tests for both (shape validation, gradient correctness, edge cases)
4. Document in `docs/AUTODIFF.md`

### Phase 2: SafeTensors Loader

**File:** `SafeTensorsLoader.cs`

```csharp
public static class SafeTensorsLoader
{
    // Read a .safetensors file and return all tensors as float arrays
    public static Dictionary<string, (float[] Data, long[] Shape)> Read(string path);
    
    // Read from byte array
    public static Dictionary<string, (float[] Data, long[] Shape)> Read(byte[] bytes);
    
    // Validate dtype is F32
    private static void ValidateDtype(string dtype, string tensorName);
    
    // Convert raw bytes to float array
    private static float[] BytesToFloats(ReadOnlySpan<byte> bytes);
}
```

Key decisions:
- **Reject non-F32 tensors** with helpful message: "Tensor '{name}' has dtype '{dtype}'. Nivara currently supports F32 only. BF16 support is planned for .NET 11."
- **Use `MemoryMarshal.Cast<byte, float>`** for zero-copy conversion when possible

### Phase 3: MobileNetV2 Module

**File:** `MobileNetV2.cs`

Define MobileNetV2 as `Module<float>` using existing Nivara modules. ReLU6 composed from `Clip(Relu(x), 0, 6)`.

### Phase 4: ResNet-18 Module

**File:** `ResNet18.cs`

Define ResNet-18 as `Module<float>` using existing modules plus the new `MaxPool2d<T>` and `AdaptiveAvgPool2d<T>`.

```csharp
public sealed class ResNet18 : Module<float>
{
    Conv2d<float> Conv1;
    BatchNorm2d<float> Norm1;
    MaxPool2d<float> MaxPool;
    
    ResNetLayer Layer1;  // 2 BasicBlocks, 64 channels
    ResNetLayer Layer2;  // 2 BasicBlocks, 128 channels, downsample
    ResNetLayer Layer3;  // 2 BasicBlocks, 256 channels, downsample
    ResNetLayer Layer4;  // 2 BasicBlocks, 512 channels, downsample
    
    AdaptiveAvgPool2d<float> AvgPool;
    Linear<float> Classifier;
    
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        var x = Relu(BatchNorm2dForward(Conv1.Forward(input), Norm1));
        x = MaxPool.Forward(x);
        
        x = Layer1.Forward(x);
        x = Layer2.Forward(x);
        x = Layer3.Forward(x);
        x = Layer4.Forward(x);
        
        x = AvgPool.Forward(x);       // [N, 512, 1, 1]
        x = Flatten(x);               // [N, 512]
        return Classifier.Forward(x); // [N, 1000]
    }
}
```

### Phase 5: Image Preprocessing

**File:** `ImagePreprocessor.cs`

Use SkiaSharp for image loading, resize, normalization. Both models use the same ImageNet normalization:
- mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Resize to 224×224, convert HWC→CHW

### Phase 6: Inference & Output

**File:** `Program.cs`

```csharp
// Model selection
var model = options.ModelName switch
{
    "mobilenet" => new MobileNetV2(),
    "resnet18" => new ResNet18(),
    _ => throw new ArgumentException($"Unknown model: {options.ModelName}")
};

// Load weights, preprocess image, run inference, display top-5
```

### Phase 7: ImageNet Labels

**File:** `imagenet_labels.txt`

1001 classes for MobileNetV2, 1000 classes for ResNet-18 (same ImageNet classes, slightly different indexing).

---

## Project Structure

```
samples/NivaraHfInference/
├── README.md                    # Sample documentation
├── NivaraHfInference.csproj     # Console app, net10.0
├── Program.cs                   # Entry point, CLI parsing, inference pipeline
├── SafeTensorsLoader.cs         # SafeTensors file reader
├── MobileNetV2.cs               # MobileNetV2 Module<float>
├── InvertedResidualBlock.cs     # InvertedResidualBlock helper
├── ResNet18.cs                  # ResNet18 Module<float>
├── BasicBlock.cs                # ResNet BasicBlock helper
├── ImagePreprocessor.cs         # SkiaSharp image loading + normalization
└── imagenet_labels.txt          # ImageNet class labels
```

---

## Dependencies

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\..\src\Nivara\Nivara.csproj" />
  </ItemGroup>

  <ItemGroup>
    <PackageReference Include="Lokad.SafeTensors" Version="0.1.0" />
    <PackageReference Include="SkiaSharp" Version="2.88.*" />
  </ItemGroup>
</Project>
```

| Package | Purpose | Why external |
|---------|---------|--------------|
| `Lokad.SafeTensors` | Read .safetensors files | No SafeTensors support in core |
| `SkiaSharp` | Image loading, resize, format conversion | No image I/O in core |

---

## CLI Options

```bash
# Interactive wizard (no args)
dotnet run --project samples/NivaraHfInference

# MobileNetV2 inference (default)
dotnet run --project samples/NivaraHfInference -- --image test.jpg

# ResNet-18 inference
dotnet run --project samples/NivaraHfInference -- --model resnet18 --image test.jpg

# Specify model file directly
dotnet run --project samples/NivaraHfInference -- --model mobilenet --model-path model.safetensors --image test.jpg

# Show top-K predictions
dotnet run --project samples/NivaraHfInference -- --model resnet18 --image test.jpg --top-k 10

# Download model from HuggingFace
dotnet run --project samples/NivaraHfInference -- --download --model resnet18 --image test.jpg
```

| Option | Default | Description |
|--------|---------|-------------|
| `--model <name>` | `mobilenet` | Model architecture: `mobilenet` or `resnet18` |
| `--model-path <path>` | — | Path to .safetensors model file (overrides --model download) |
| `--image <path>` | — | Path to input image |
| `--top-k <int>` | 5 | Number of top predictions to show |
| `--download` | — | Download model from HuggingFace Hub |
| `--labels <path>` | — | Path to ImageNet labels file |
| `--help`, `-h` | — | Show CLI help |

---

## Verification Plan

### Correctness Validation

1. **Weight loading test:** Verify tensor count and shapes match expected architecture for both models
2. **Forward pass shape test:** Input `[1, 3, 224, 224]` → MobileNetV2: `[1, 1001]`, ResNet-18: `[1, 1000]`
3. **Deterministic output:** Same input image produces identical predictions across runs
4. **Python comparison:** Run same image through HuggingFace Python, compare top-5 predictions

### Expected Output

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

### Performance Baseline

| Metric | MobileNetV2 | ResNet-18 | Notes |
|--------|-------------|-----------|-------|
| Model load time | <500ms | <800ms | 13.4MB vs 44MB weights |
| Image preprocessing | <100ms | <100ms | Same pipeline |
| Forward pass | <50ms | <80ms | More FLOPs in ResNet |
| Total pipeline | <700ms | <1000ms | Load + preprocess + inference |
| Memory usage | ~60MB | ~180MB | Model weights + activations |

---

## Risk: TensorFlow Padding (MobileNetV2)

The default HuggingFace MobileNetV2 (`google/mobilenet_v2_1.0_224`) uses `tf_padding=True`, which means convolution layers use **TensorFlow-style asymmetric padding**. Nivara's `Conv2d<T>` uses symmetric padding only.

**Mitigations:**
1. Use a model variant with `tf_padding=False` if available
2. Implement asymmetric padding at the sample level
3. Add asymmetric padding support to `Conv2d` (core library change)

**ResNet-18 has no such issue** — it uses standard symmetric padding.

---

## Nivara v1.1 Alignment

This sample directly showcases Nivara v1.1 capabilities:

| v1.1 Feature | How It's Demonstrated |
|-------------|----------------------|
| Conv2d | Core of both architectures (inverted residuals, basic blocks) |
| BatchNorm2d | Used throughout both architectures |
| DepthwiseSeparableConv2d | MobileNetV2's core building block |
| Linear | Classification heads in both models |
| Module<T> | Both models are `Module<T>` subclasses with `LoadStateDict()` |
| MaxPool2d (new) | ResNet-18 stem pooling |
| AdaptiveAvgPool2d (new) | Global average pooling in both architectures |

---

## Future Work

1. **GELU activation** → unlock DistilBERT inference
2. **Positional encoding module** → enable BERT/GPT-2 inference
3. **Batch inference** → process multiple images at once
4. **Model comparison** → side-by-side accuracy/speed between MobileNetV2 and ResNet-18
5. **ONNX comparison** → validate against ONNX Runtime output
6. **Export to ONNX** → save Nivara-trained models in ONNX format

---

## References

- [SafeTensors format specification](https://github.com/safetensors/safetensors)
- [MobileNetV2 on HuggingFace](https://huggingface.co/google/mobilenet_v2_1.0_224)
- [ResNet-18 on HuggingFace](https://huggingface.co/microsoft/resnet-18)
- [ResNet-18 config](https://huggingface.co/docs/transformers/en/model_doc/resnet)
- [Lokad.SafeTensors NuGet](https://www.nuget.org/packages/Lokad.SafeTensors)
- [SkiaSharp NuGet](https://www.nuget.org/packages/SkiaSharp)
- [Nivara AutoDiff docs](../../docs/AUTODIFF.md)
- [SafeTensors research](../../docs/SAFETENSORS.md)
