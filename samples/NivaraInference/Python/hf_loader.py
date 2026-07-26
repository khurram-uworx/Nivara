"""Shared utilities for loading HuggingFace SafeTensors weights into PyTorch."""
import os
import sys
import struct
import json
import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "samples", "data")
IMAGES_DIR = os.path.join(REPO_ROOT, "samples", "data", "images")


def load_safetensors(model_dir: str) -> dict[str, torch.Tensor]:
    """Load all F32 tensors from a safetensors file into a name→tensor dict."""
    path = os.path.join(model_dir, "model.safetensors")
    result = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            result[key] = f.get_tensor(key)
    return result


def bn_from_hf(prefix: str, tensors: dict, num_features: int) -> dict:
    """Extract BatchNorm params from HuggingFace tensor naming."""
    params = {}
    for suffix, shape in [("weight", [num_features]), ("bias", [num_features]),
                          ("running_mean", [num_features]), ("running_var", [num_features])]:
        key = f"{prefix}.normalization.{suffix}"
        if key in tensors:
            params[suffix] = tensors[key]
        else:
            params[suffix] = torch.zeros(shape) if suffix in ("running_mean", "running_var") else torch.ones(shape) if suffix == "weight" else torch.zeros(shape)
    return params


def load_conv_from_hf(tensors: dict, hf_key: str) -> torch.Tensor:
    """Load a convolution weight tensor from HuggingFace key."""
    return tensors[hf_key]


def print_top_k(logits: torch.Tensor, k: int = 5):
    """Print top-k predictions."""
    probs = logits.softmax(dim=-1)
    topk = probs.topk(k)
    values = topk.values[0].tolist()
    indices = topk.indices[0].tolist()
    print(f"  Top-{k} predictions:")
    for i, (idx, val) in enumerate(zip(indices, values)):
        print(f"    #{i+1}: class {idx:5d}  score={val:.6f}")
    return indices, values
