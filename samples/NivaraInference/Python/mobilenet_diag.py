"""Diagnostic: save intermediate tensors at each major stage of MobileNetV2."""
import os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import load_safetensors, MODELS_DIR

BLOCK_CONFIGS = [
    (96, 24, 1), (144, 24, 1), (144, 32, 2), (192, 32, 1),
    (192, 32, 1), (192, 64, 2), (384, 64, 1), (384, 64, 1),
    (384, 64, 1), (384, 96, 1), (576, 96, 1), (576, 96, 2),
    (576, 160, 1), (960, 160, 1), (960, 160, 1), (960, 320, 1),
]


def relu6(x):
    return torch.clamp(torch.relu(x), 0, 6)


def main():
    model_dir = os.path.join(MODELS_DIR, "mobilenet_v2")
    tensors = load_safetensors(model_dir)

    from mobilenet import MobileNetV2HF, load_weights
    model = MobileNetV2HF(num_classes=1001)
    model.eval()
    load_weights(model, tensors)

    inp_bin = os.path.join(MODELS_DIR, "compare_input.bin")
    inp_np = np.fromfile(inp_bin, dtype=np.float32).reshape(1, 3, 224, 224)
    inp = torch.from_numpy(inp_np)

    diag_dir = os.path.join(MODELS_DIR, "diag")
    os.makedirs(diag_dir, exist_ok=True)

    def save(name, tensor):
        arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
        arr.tofile(os.path.join(diag_dir, f"{name}.bin"))
        print(f"  {name}: shape={list(arr.shape)}, mean={arr.mean():.6f}, first3={arr.flatten()[:3].tolist()}")
        return arr

    with torch.no_grad():
        x = inp
        save("mn_input", x)

        x = model.stem(x)
        save("mn_stem", x)

        for i, block in enumerate(model.blocks):
            x = block(x)
            if i % 4 == 3:
                save(f"mn_after_block{i}", x)

        x = model.head_conv(x)
        x = model.head_bn(x)
        x = relu6(x)
        save("mn_head", x)

        x = model.avgpool(x)
        x_flat = torch.flatten(x, 1)
        save("mn_avgpool", x_flat)

        logits = model.classifier(x_flat)
        save("mn_logits", logits)


if __name__ == "__main__":
    main()
