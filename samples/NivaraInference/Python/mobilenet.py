"""MobileNetV2 inference using HuggingFace SafeTensors weights (same as C# sample)."""
import time
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import load_safetensors, IMAGES_DIR, MODELS_DIR, print_top_k

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BLOCK_CONFIGS = [
    (96, 24, 1),
    (144, 24, 1),
    (144, 32, 2),
    (192, 32, 1),
    (192, 32, 1),
    (192, 64, 2),
    (384, 64, 1),
    (384, 64, 1),
    (384, 64, 1),
    (384, 96, 1),
    (576, 96, 1),
    (576, 96, 2),
    (576, 160, 1),
    (960, 160, 1),
    (960, 160, 1),
    (960, 320, 1),
]


def relu6(x):
    return torch.clamp(torch.relu(x), 0, 6)


class StemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.first_bn = nn.BatchNorm2d(32)
        self.dw_conv = nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False)
        self.dw_bn = nn.BatchNorm2d(32)
        self.pw_conv = nn.Conv2d(32, 16, 1, bias=False)
        self.pw_bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = relu6(self.first_bn(self.first_conv(x)))
        x = relu6(self.dw_bn(self.dw_conv(x)))
        x = relu6(self.pw_bn(self.pw_conv(x)))
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, expand_ch, out_ch, stride):
        super().__init__()
        self.has_expansion = in_ch != expand_ch
        self.use_residual = (stride == 1 and in_ch == out_ch)

        if self.has_expansion:
            self.expand_conv = nn.Conv2d(in_ch, expand_ch, 1, bias=False)
            self.expand_bn = nn.BatchNorm2d(expand_ch)

        self.dw_conv = nn.Conv2d(expand_ch, expand_ch, 3, stride=stride, padding=1, groups=expand_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(expand_ch)
        self.project_conv = nn.Conv2d(expand_ch, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        if self.has_expansion:
            out = relu6(self.expand_bn(self.expand_conv(x)))
        else:
            out = x
        out = relu6(self.dw_bn(self.dw_conv(out)))
        out = self.project_bn(self.project_conv(out))
        if self.use_residual:
            out = out + identity
        return out


class MobileNetV2HF(nn.Module):
    def __init__(self, num_classes=1001):
        super().__init__()
        self.stem = StemBlock()
        self.blocks = nn.ModuleList()
        in_ch = 16
        for expand, out_ch, stride in BLOCK_CONFIGS:
            self.blocks.append(InvertedResidual(in_ch, expand, out_ch, stride))
            in_ch = out_ch
        self.head_conv = nn.Conv2d(320, 1280, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = relu6(self.head_bn(self.head_conv(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_weights(model, tensors):
    """Load weights from HuggingFace safetensors into the PyTorch model."""
    sd = model.state_dict()
    mapping = {}

    # Stem
    mapping["mobilenet_v2.conv_stem.first_conv.convolution.weight"] = "stem.first_conv.weight"
    mapping["mobilenet_v2.conv_stem.first_conv.normalization.weight"] = "stem.first_bn.weight"
    mapping["mobilenet_v2.conv_stem.first_conv.normalization.bias"] = "stem.first_bn.bias"
    mapping["mobilenet_v2.conv_stem.first_conv.normalization.running_mean"] = "stem.first_bn.running_mean"
    mapping["mobilenet_v2.conv_stem.first_conv.normalization.running_var"] = "stem.first_bn.running_var"
    mapping["mobilenet_v2.conv_stem.conv_3x3.convolution.weight"] = "stem.dw_conv.weight"
    mapping["mobilenet_v2.conv_stem.conv_3x3.normalization.weight"] = "stem.dw_bn.weight"
    mapping["mobilenet_v2.conv_stem.conv_3x3.normalization.bias"] = "stem.dw_bn.bias"
    mapping["mobilenet_v2.conv_stem.conv_3x3.normalization.running_mean"] = "stem.dw_bn.running_mean"
    mapping["mobilenet_v2.conv_stem.conv_3x3.normalization.running_var"] = "stem.dw_bn.running_var"
    mapping["mobilenet_v2.conv_stem.reduce_1x1.convolution.weight"] = "stem.pw_conv.weight"
    mapping["mobilenet_v2.conv_stem.reduce_1x1.normalization.weight"] = "stem.pw_bn.weight"
    mapping["mobilenet_v2.conv_stem.reduce_1x1.normalization.bias"] = "stem.pw_bn.bias"
    mapping["mobilenet_v2.conv_stem.reduce_1x1.normalization.running_mean"] = "stem.pw_bn.running_mean"
    mapping["mobilenet_v2.conv_stem.reduce_1x1.normalization.running_var"] = "stem.pw_bn.running_var"

    # Blocks
    for i, (expand, out_ch, stride) in enumerate(BLOCK_CONFIGS):
        prefix = f"mobilenet_v2.layer.{i}"
        in_ch = 16 if i == 0 else BLOCK_CONFIGS[i - 1][1]
        has_expansion = in_ch != expand

        if has_expansion:
            mapping[f"{prefix}.expand_1x1.convolution.weight"] = f"blocks.{i}.expand_conv.weight"
            mapping[f"{prefix}.expand_1x1.normalization.weight"] = f"blocks.{i}.expand_bn.weight"
            mapping[f"{prefix}.expand_1x1.normalization.bias"] = f"blocks.{i}.expand_bn.bias"
            mapping[f"{prefix}.expand_1x1.normalization.running_mean"] = f"blocks.{i}.expand_bn.running_mean"
            mapping[f"{prefix}.expand_1x1.normalization.running_var"] = f"blocks.{i}.expand_bn.running_var"

        mapping[f"{prefix}.conv_3x3.convolution.weight"] = f"blocks.{i}.dw_conv.weight"
        mapping[f"{prefix}.conv_3x3.normalization.weight"] = f"blocks.{i}.dw_bn.weight"
        mapping[f"{prefix}.conv_3x3.normalization.bias"] = f"blocks.{i}.dw_bn.bias"
        mapping[f"{prefix}.conv_3x3.normalization.running_mean"] = f"blocks.{i}.dw_bn.running_mean"
        mapping[f"{prefix}.conv_3x3.normalization.running_var"] = f"blocks.{i}.dw_bn.running_var"
        mapping[f"{prefix}.reduce_1x1.convolution.weight"] = f"blocks.{i}.project_conv.weight"
        mapping[f"{prefix}.reduce_1x1.normalization.weight"] = f"blocks.{i}.project_bn.weight"
        mapping[f"{prefix}.reduce_1x1.normalization.bias"] = f"blocks.{i}.project_bn.bias"
        mapping[f"{prefix}.reduce_1x1.normalization.running_mean"] = f"blocks.{i}.project_bn.running_mean"
        mapping[f"{prefix}.reduce_1x1.normalization.running_var"] = f"blocks.{i}.project_bn.running_var"

    # Head
    mapping["mobilenet_v2.conv_1x1.convolution.weight"] = "head_conv.weight"
    mapping["mobilenet_v2.conv_1x1.normalization.weight"] = "head_bn.weight"
    mapping["mobilenet_v2.conv_1x1.normalization.bias"] = "head_bn.bias"
    mapping["mobilenet_v2.conv_1x1.normalization.running_mean"] = "head_bn.running_mean"
    mapping["mobilenet_v2.conv_1x1.normalization.running_var"] = "head_bn.running_var"

    # Classifier
    mapping["classifier.weight"] = "classifier.weight"
    mapping["classifier.bias"] = "classifier.bias"

    loaded = 0
    skipped = []
    for hf_key, pt_key in mapping.items():
        if hf_key in tensors and pt_key in sd:
            if tensors[hf_key].shape == sd[pt_key].shape:
                sd[pt_key] = tensors[hf_key]
                loaded += 1
            else:
                skipped.append(f"{hf_key}: {tensors[hf_key].shape} vs {sd[pt_key].shape}")
        elif hf_key not in tensors:
            skipped.append(f"{hf_key}: not in tensors")
        else:
            skipped.append(f"{pt_key}: not in model")

    model.load_state_dict(sd)
    print(f"  Loaded {loaded}/{len(mapping)} parameters")
    if skipped:
        print(f"  Skipped: {skipped}")


def preprocess_image(path, size=224):
    from PIL import Image
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)


def main():
    print("=== HuggingFace MobileNetV2 Inference (same weights as C#) ===")
    print()

    model_dir = os.path.join(MODELS_DIR, "mobilenet_v2")
    print(f"Loading weights from {model_dir}/model.safetensors...")
    tensors = load_safetensors(model_dir)
    print(f"  Loaded {len(tensors)} tensors")

    print("Building MobileNetV2 model...")
    model = MobileNetV2HF(num_classes=1001)
    model.eval()
    load_weights(model, tensors)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print()

    # Warmup
    print("Warmup (3 passes)...")
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    print()

    # Benchmark synthetic
    print("Benchmark: synthetic 224x224 input (10 passes)...")
    times = []
    for i in range(10):
        inp = torch.randn(1, 3, 224, 224)
        start = time.perf_counter()
        with torch.no_grad():
            out = model(inp)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        print(f"  Run {i+1:2d}: {elapsed:.1f} ms")

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.1f} ms  (min={min(times):.1f}, max={max(times):.1f})")
    print()
    print_top_k(out)
    print()

    # Real images
    if os.path.exists(IMAGES_DIR):
        image_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
        if image_files:
            print("Benchmark: real images...")
            for fname in image_files:
                path = os.path.join(IMAGES_DIR, fname)
                from PIL import Image
                img = Image.open(path)
                inp = preprocess_image(path)
                start = time.perf_counter()
                with torch.no_grad():
                    out = model(inp)
                elapsed = (time.perf_counter() - start) * 1000
                print(f"  {fname} ({img.size[0]}x{img.size[1]}): {elapsed:.1f} ms")
                print_top_k(out, k=3)
                print()


if __name__ == "__main__":
    main()
