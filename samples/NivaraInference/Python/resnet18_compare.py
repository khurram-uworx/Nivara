"""Compare Python and C# outputs layer-by-layer using the SAME input + weights."""
import os, sys, struct, time
import torch, torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import load_safetensors, MODELS_DIR


def main():
    model_dir = os.path.join(MODELS_DIR, "resnet18")
    tensors = load_safetensors(model_dir)

    # Build ResNet18
    from resnet18 import ResNet18HF, load_weights
    model = ResNet18HF(num_classes=1000)
    model.eval()
    load_weights(model, tensors)

    # Load the same input C# will use
    inp_bin = os.path.join(MODELS_DIR, "compare_input.bin")
    inp_np = np.fromfile(inp_bin, dtype=np.float32).reshape(1, 3, 224, 224)
    inp = torch.from_numpy(inp_np)
    print(f"Input: shape={list(inp.shape)}, mean={inp.mean():.6f}")

    # Save input back to be sure it matches
    inp.numpy().tofile(inp_bin)

    with torch.no_grad():
        # Step 1: stem conv only (no BN)
        conv_w = tensors["resnet.embedder.embedder.convolution.weight"]
        conv_only = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        conv_only.weight.data.copy_(conv_w)
        conv_out = conv_only(inp)
        print(f"\nStep 1 - stem conv only:")
        print(f"  shape={list(conv_out.shape)}, mean={conv_out.mean():.6f}")
        print(f"  [0,0,:3,:3] = {conv_out[0,0,:3,:3].flatten().tolist()}")

        # Save conv output for C# comparison
        conv_out.numpy().astype('float32').tofile(os.path.join(MODELS_DIR, "compare_stem_conv.bin"))
        print(f"  Saved to {MODELS_DIR}/compare_stem_conv.bin")

        # Step 2: stem conv + BN
        bn_w = tensors["resnet.embedder.embedder.normalization.weight"]
        bn_b = tensors["resnet.embedder.embedder.normalization.bias"]
        bn_m = tensors["resnet.embedder.embedder.normalization.running_mean"]
        bn_v = tensors["resnet.embedder.embedder.normalization.running_var"]
        bn = nn.BatchNorm2d(64)
        bn.weight.data.copy_(bn_w)
        bn.bias.data.copy_(bn_b)
        bn.running_mean.copy_(bn_m)
        bn.running_var.copy_(bn_v)
        bn.eval()
        stem_out = bn(conv_out)
        print(f"\nStep 2 - stem conv+bn:")
        print(f"  shape={list(stem_out.shape)}, mean={stem_out.mean():.6f}")
        print(f"  [0,0,:3,:3] = {stem_out[0,0,:3,:3].flatten().tolist()}")

        # Step 3: relu + pool
        pool_out = model.stem_pool(torch.relu(stem_out))
        print(f"\nStep 3 - relu+pool:")
        print(f"  shape={list(pool_out.shape)}, mean={pool_out.mean():.6f}")
        print(f"  [0,0,:3,:3] = {pool_out[0,0,:3,:3].flatten().tolist()}")

        # Step 4: stage0
        s0 = model.stage0_layer1(model.stage0_layer0(pool_out))
        print(f"\nStep 4 - after stage0:")
        print(f"  shape={list(s0.shape)}, mean={s0.mean():.6f}")
        print(f"  [0,0,:3,:3] = {s0[0,0,:3,:3].flatten().tolist()}")

        # Step 5: all stages
        x = s0
        x = model.stage1_layer1(model.stage1_layer0(x))
        x = model.stage2_layer1(model.stage2_layer0(x))
        x = model.stage3_layer1(model.stage3_layer0(x))
        print(f"\nStep 5 - after all stages:")
        print(f"  shape={list(x.shape)}, mean={x.mean():.6f}")
        print(f"  [0,0,:3,:3] = {x[0,0,:3,:3].flatten().tolist()}")

        # Step 6: avgpool + fc
        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        print(f"\nStep 6 - after avgpool:")
        print(f"  shape={list(x.shape)}, values = {x[0,:5].tolist()}")

        logits = model.fc(x)
        print(f"\nStep 7 - final logits (first 10):")
        print(f"  {logits[0,:10].tolist()}")

        # Save logits for C# comparison
        logits.numpy().astype('float32').tofile(os.path.join(MODELS_DIR, "compare_logits.bin"))
        print(f"  Saved to {MODELS_DIR}/compare_logits.bin")


if __name__ == "__main__":
    main()
