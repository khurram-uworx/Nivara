"""Generate a fixed random input and save it for C# comparison."""
import os
import sys
import struct
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def main():
    rng = torch.Generator()
    rng.manual_seed(42)
    inp = torch.randn(1, 3, 224, 224, generator=rng)

    # Save as raw float32 little-endian binary (NCHW)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    path = os.path.join(out_dir, "compare_input.bin")
    inp.numpy().astype(np.float32).tofile(path)
    print(f"Saved input tensor: {inp.shape}, {inp.numel()} floats -> {path}")
    print(f"  mean={inp.mean():.6f}, std={inp.std():.6f}")
    print(f"  first 5 values: {inp.flatten()[:5].tolist()}")

    # Also save as text for easy inspection
    txt_path = os.path.join(out_dir, "compare_input.txt")
    with open(txt_path, "w") as f:
        vals = inp.flatten().tolist()
        f.write(" ".join(f"{v:.8f}" for v in vals))
    print(f"  Text copy -> {txt_path}")


if __name__ == "__main__":
    main()
