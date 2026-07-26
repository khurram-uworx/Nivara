"""ResNet-18 inference using HuggingFace SafeTensors weights (same as C# sample)."""
import time
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import load_safetensors, IMAGES_DIR, MODELS_DIR, print_top_k

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.has_downsample = in_ch != out_ch or stride != 1
        if self.has_downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18HF(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.stage0_layer0 = BasicBlock(64, 64)
        self.stage0_layer1 = BasicBlock(64, 64)
        self.stage1_layer0 = BasicBlock(64, 128, stride=2)
        self.stage1_layer1 = BasicBlock(128, 128)
        self.stage2_layer0 = BasicBlock(128, 256, stride=2)
        self.stage2_layer1 = BasicBlock(256, 256)
        self.stage3_layer0 = BasicBlock(256, 512, stride=2)
        self.stage3_layer1 = BasicBlock(512, 512)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.relu(self.stem_bn(self.stem_conv(x)))
        x = self.stem_pool(x)
        x = self.stage0_layer0(x)
        x = self.stage0_layer1(x)
        x = self.stage1_layer0(x)
        x = self.stage1_layer1(x)
        x = self.stage2_layer0(x)
        x = self.stage2_layer1(x)
        x = self.stage3_layer0(x)
        x = self.stage3_layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def load_weights(model, tensors):
    """Load weights from HuggingFace safetensors into the PyTorch model."""
    sd = model.state_dict()

    # Mapping: HF key -> PyTorch key
    mapping = {
        "resnet.embedder.embedder.convolution.weight": "stem_conv.weight",
        "resnet.embedder.embedder.normalization.weight": "stem_bn.weight",
        "resnet.embedder.embedder.normalization.bias": "stem_bn.bias",
        "resnet.embedder.embedder.normalization.running_mean": "stem_bn.running_mean",
        "resnet.embedder.embedder.normalization.running_var": "stem_bn.running_var",
        "resnet.embedder.embedder.normalization.num_batches_tracked": "stem_bn.num_batches_tracked",
    }

    for stage in range(4):
        for layer in range(2):
            for k in range(2):
                prefix = f"resnet.encoder.stages.{stage}.layers.{layer}"
                py_layer = f"stage{stage}_layer{layer}"
                mapping[f"{prefix}.layer.{k}.convolution.weight"] = f"{py_layer}.conv{k+1}.weight"
                mapping[f"{prefix}.layer.{k}.normalization.weight"] = f"{py_layer}.bn{k+1}.weight"
                mapping[f"{prefix}.layer.{k}.normalization.bias"] = f"{py_layer}.bn{k+1}.bias"
                mapping[f"{prefix}.layer.{k}.normalization.running_mean"] = f"{py_layer}.bn{k+1}.running_mean"
                mapping[f"{prefix}.layer.{k}.normalization.running_var"] = f"{py_layer}.bn{k+1}.running_var"
                mapping[f"{prefix}.layer.{k}.normalization.num_batches_tracked"] = f"{py_layer}.bn{k+1}.num_batches_tracked"

            # shortcut
            prefix = f"resnet.encoder.stages.{stage}.layers.{layer}"
            py_layer = f"stage{stage}_layer{layer}"
            mapping[f"{prefix}.shortcut.convolution.weight"] = f"{py_layer}.downsample.0.weight"
            mapping[f"{prefix}.shortcut.normalization.weight"] = f"{py_layer}.downsample.1.weight"
            mapping[f"{prefix}.shortcut.normalization.bias"] = f"{py_layer}.downsample.1.bias"
            mapping[f"{prefix}.shortcut.normalization.running_mean"] = f"{py_layer}.downsample.1.running_mean"
            mapping[f"{prefix}.shortcut.normalization.running_var"] = f"{py_layer}.downsample.1.running_var"
            mapping[f"{prefix}.shortcut.normalization.num_batches_tracked"] = f"{py_layer}.downsample.1.num_batches_tracked"

    mapping["classifier.1.weight"] = "fc.weight"
    mapping["classifier.1.bias"] = "fc.bias"

    loaded = 0
    for hf_key, pt_key in mapping.items():
        if hf_key in tensors and pt_key in sd:
            sd[pt_key] = tensors[hf_key]
            loaded += 1

    model.load_state_dict(sd)
    print(f"  Loaded {loaded}/{len(mapping)} parameters")


def preprocess_image(path, size=224):
    """Load image, resize to size×size, apply ImageNet normalization."""
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
    print("=== HuggingFace ResNet-18 Inference (same weights as C#) ===")
    print()

    model_dir = os.path.join(MODELS_DIR, "resnet18")
    print(f"Loading weights from {model_dir}/model.safetensors...")
    tensors = load_safetensors(model_dir)
    print(f"  Loaded {len(tensors)} tensors")

    print("Building ResNet-18 model...")
    model = ResNet18HF(num_classes=1000)
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
