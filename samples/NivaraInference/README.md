# Nivara HuggingFace Inference Sample

Demonstrates loading pre-trained HuggingFace vision models (MobileNetV2, ResNet-18) into Nivara's AutoDiff engine and running forward inference — all without any third-party ML framework dependencies.

## What this sample does

1. **Loads SafeTensors weights** using a custom zero-dependency reader (`SafeTensorsLoader.cs`)
2. **Builds model architecture** using Nivara's `Module<T>`, `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`
3. **Runs forward inference** with a synthetic random input and reports top-5 predictions

## Supported models

| Model | Size | Tensors | Parameters | Output classes |
|-------|------|---------|------------|----------------|
| MobileNetV2 | 13.5 MB | 262 | 3.4M | 1001 |
| ResNet-18 | 44.6 MB | 102 | 11.7M | 1000 |

Models are downloaded to `samples/data/` via HuggingFace CLI:
```bash
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download timm/resnet18.augreg_in1k --local-dir samples/data/resnet18
```

## Usage

```bash
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18
```

## Architecture notes

### SafeTensorsLoader

Custom zero-dependency reader that parses the SafeTensors binary format directly:
- Memory-mapped header parsing via `System.Text.Json`
- Zero-copy tensor extraction using `MemoryMarshal.Cast<byte, float>`
- Skips non-F32 tensors silently (e.g., I64 `num_batches_tracked`)
- Throws `NotSupportedException` with guidance for BF16 tensors

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

### Weight loading

Each model defines a static `LoadWeights()` factory that maps HuggingFace tensor names to Nivara module parameters. No reflection or generic deserialization — explicit, type-safe loading.

## Nivara core gaps exercised

- **Conv2d** with asymmetric padding, grouped convolutions, 1×1 fast path
- **BatchNorm2d** with running statistics for inference mode
- **MaxPool2d** with argmax tracking for backward pass
- **AdaptiveAvgPool2d** with gradient broadcast
- **Linear** with matrix multiply + bias broadcast
- **Module<T>** tree with `LoadStateDict` for parameter loading
