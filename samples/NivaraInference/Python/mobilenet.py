"""MobileNetV2 inference using PyTorch (CPU)."""
import time
import sys
import os
import torch
from torchvision import models

sys.path.insert(0, os.path.dirname(__file__))
from utils import timer, load_image, get_transform, print_top_k


def main():
    print("=== PyTorch MobileNetV2 Inference (CPU) ===")
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print()

    # Load model
    print("Loading MobileNetV2 weights...")
    with timer("Model load"):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print()

    transform = get_transform()

    # Warmup
    print("Warmup (random input)...")
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    print()

    # Benchmark with synthetic input
    print("Benchmark: synthetic 224x224 input...")
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

    # Benchmark with image
    image_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "images")
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

    if image_files:
        print("Benchmark: real images...")
        for fname in image_files:
            path = os.path.join(image_dir, fname)
            img = load_image(path)
            inp = transform(img).unsqueeze(0)

            start = time.perf_counter()
            with torch.no_grad():
                out = model(inp)
            elapsed = (time.perf_counter() - start) * 1000

            print(f"  {fname} ({img.size[0]}x{img.size[1]}): {elapsed:.1f} ms")
            print_top_k(out, k=3)
            print()


if __name__ == "__main__":
    main()
