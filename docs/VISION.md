# Vision models — MobileNetV2, ResNet-18

## Model overview

Two vision classification models:

| Model | Type | Weight size | Tensors | Parameters | Output |
|-------|------|-------------|---------|------------|--------|
| MobileNetV2 | Vision (classification) | 13.5 MB | 262 | 3.4M | 1001 classes |
| ResNet-18 | Vision (classification) | 44.6 MB | 102 | 11.7M | 1000 classes |

**MobileNetV2** — a lightweight classification network built from inverted residual blocks:

- **Stem**: 3×3 conv → BatchNorm → ReLU6
- **16 inverted residual blocks** with expansion/depthwise/project phases
- **Depthwise separable convolutions** (groups = input channels) for 3×3 layers
- **ReLU6** activation via `Clip(Relu(x), 0, 6)`
- **Residual shortcuts** only when `stride == 1 && inChannels == outChannels`
- **Head**: 1×1 conv → global avg pool → 1001-class linear classifier

**ResNet-18** — a standard 18-layer residual network:

- **Stem**: 7×7 conv → BatchNorm → ReLU → 3×3 MaxPool
- **4 stages** with channel progression: 64 → 128 → 256 → 512
- **BasicBlock**: two 3×3 convs with BatchNorm + ReLU, identity shortcut (or 1×1 conv when dimensions change)
- **Head**: global average pooling → 1000-class linear classifier
- **Downsampling** at stage boundaries via strided convolution in the shortcut path

## How this model improved the library

The vision models were the first in the sample. They exercised:

- **`Conv2d<T>`** with asymmetric padding, grouped convs, and the 1×1 fast path — the first real convolution workload.
- **`BatchNorm2d<T>`** with running statistics — every conv → BN block.
- **`MaxPool2d<T>`** with argmax — ResNet-18 stem.
- **`AdaptiveAvgPool2d<T>`** with gradient broadcast — both model heads.
- **Depthwise separable convolutions** (groups = channels) — MobileNetV2 3×3 blocks.

The vision gap is now tracked as issue **#457** (SIMD + row parallelism for the grouped/depthwise conv path). The current `Conv2d.cs` grouped/depthwise path is a 7-deep nested loop (`n → ic → oc → oh → ow → kh → kw`, lines 625–655) with no `Vector<>` and no `Parallel.For`. MobileNetV2 is depthwise-dominated, which is also why ResNet-18's gap is much smaller.

## Library features used

| Capability | Where exercised |
|---|---|
| `Conv2d<T>` with asymmetric padding, grouped convs, 1×1 fast path | All conv layers in both models |
| `BatchNorm2d<T>` with running statistics | Every conv → BN block |
| `MaxPool2d<T>` with argmax | ResNet-18 stem |
| `AdaptiveAvgPool2d<T>` with gradient broadcast | Both model heads |
| `Linear<T>` with MatMul + bias | Classifier heads |
| `Module<T>` tree with `LoadStateDict` | Full model construction |
| Depthwise separable convolutions (groups = channels) | MobileNetV2 3×3 blocks |

## Verification

| Model | Gate | Result |
|-------|------|--------|
| MobileNetV2 | `compare` vs PyTorch | output match |
| ResNet-18 | `compare` vs PyTorch | output match |

## Performance

| Model | PyTorch | Nivara | Slowdown |
|-------|---------|--------|----------|
| MobileNetV2 | 22.4 ms | 731.7 ms | ~33× |
| ResNet-18 | 14.1 ms | 249.4 ms | ~18× |

The vision gap is dominated by convolution kernels (especially depthwise convolutions in MobileNetV2), which use naive nested loops. PyTorch vision is multi-threaded MKL; Nivara's conv kernels are single-threaded naive loops. See the [NivaraInference README](../samples/NivaraInference/README.md) for full benchmark tables.
