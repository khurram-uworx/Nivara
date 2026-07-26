"""Diagnostic: compare each step of ResNet-18 between Python and C# outputs."""
import os, sys, struct
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import load_safetensors, MODELS_DIR

def main():
    model_dir = os.path.join(MODELS_DIR, "resnet18")
    tensors = load_safetensors(model_dir)

    from resnet18 import ResNet18HF, load_weights
    model = ResNet18HF(num_classes=1000)
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
        # Step 1: stem conv only
        conv_w = tensors["resnet.embedder.embedder.convolution.weight"]
        conv_only = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        conv_only.weight.data.copy_(conv_w)
        stem_conv = conv_only(inp)
        save("step1_stem_conv", stem_conv)

        # Step 2: stem BN (eval with running stats)
        bn_w = tensors["resnet.embedder.embedder.normalization.weight"]
        bn_b = tensors["resnet.embedder.embedder.normalization.bias"]
        bn_rm = tensors["resnet.embedder.embedder.normalization.running_mean"]
        bn_rv = tensors["resnet.embedder.embedder.normalization.running_var"]
        bn = nn.BatchNorm2d(64)
        bn.weight.data.copy_(bn_w)
        bn.bias.data.copy_(bn_b)
        bn.running_mean.copy_(bn_rm)
        bn.running_var.copy_(bn_rv)
        bn.eval()
        stem_bn = bn(stem_conv)
        save("step2_stem_bn", stem_bn)

        # Step 3: relu + pool
        stem_relu = torch.relu(stem_bn)
        save("step3_stem_relu", stem_relu)
        stem_pool = model.stem_pool(stem_relu)
        save("step4_stem_pool", stem_pool)

        # Step 5: stage0
        s0 = model.stage0_layer1(model.stage0_layer0(stem_pool))
        save("step5_stage0", s0)

        # Step 6: all stages
        x = s0
        x = model.stage1_layer1(model.stage1_layer0(x))
        save("step6_stage1", x)
        x = model.stage2_layer1(model.stage2_layer0(x))
        save("step7_stage2", x)
        x = model.stage3_layer1(model.stage3_layer0(x))
        save("step8_stage3", x)

        # Step 9: avgpool + fc
        x = model.avgpool(x)
        x_flat = torch.flatten(x, 1)
        save("step9_avgpool", x_flat)
        logits = model.fc(x_flat)
        save("step10_logits", logits)

        # Also save the expected final output
        full_out = model(inp)
        save("final_logits", full_out)

if __name__ == "__main__":
    main()
