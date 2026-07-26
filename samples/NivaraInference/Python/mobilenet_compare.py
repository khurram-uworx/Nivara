"""Compare MobileNetV2 Python vs C# outputs using the same input + weights."""
import os, sys, struct
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import load_safetensors, MODELS_DIR
from mobilenet import MobileNetV2HF, load_weights


def main():
    model_dir = os.path.join(MODELS_DIR, "mobilenet_v2")
    tensors = load_safetensors(model_dir)

    model = MobileNetV2HF(num_classes=1001)
    model.eval()
    load_weights(model, tensors)

    inp_bin = os.path.join(MODELS_DIR, "compare_input.bin")
    inp_np = np.fromfile(inp_bin, dtype=np.float32).reshape(1, 3, 224, 224)
    inp = torch.from_numpy(inp_np)
    print(f"Input: shape={list(inp.shape)}, mean={inp.mean():.6f}")

    with torch.no_grad():
        logits = model(inp)
        print(f"\nLogits (first 10): {logits[0,:10].tolist()}")
        print(f"Logits stats: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}")

        topk = torch.topk(logits[0], 5)
        for i in range(5):
            print(f"  #{i+1}: class {topk.indices[i]:4d}  score={topk.values[i]:.6f}")

        logits.numpy().astype('float32').tofile(os.path.join(MODELS_DIR, "compare_mobilenet_logits.bin"))
        print(f"\nSaved Python logits to {MODELS_DIR}/compare_mobilenet_logits.bin")


if __name__ == "__main__":
    main()
