"""Shared utilities for inference benchmarking."""
import time
import sys
import os
from contextlib import contextmanager
from PIL import Image
import torchvision.transforms as T


@contextmanager
def timer(label: str):
    """Context manager that prints elapsed time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"  {label}: {elapsed * 1000:.1f} ms")
    return elapsed


def load_image(path: str, size: int = 224) -> Image.Image:
    """Load and resize an image to the given size."""
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def get_transform():
    """Standard ImageNet preprocessing for inference."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def print_top_k(logits, k=5, label_offset=1):
    """Print top-k predictions from a logits tensor."""
    probs = logits.softmax(dim=-1)
    topk = probs.topk(k)
    values = topk.values[0].tolist()
    indices = topk.indices[0].tolist()
    print(f"  Top-{k} predictions:")
    for i, (idx, val) in enumerate(zip(indices, values)):
        print(f"    #{i+1}: class {idx:5d}  score={val:.6f}")
    return indices, values
