"""Generate reference tensors for each NN layer type using PyTorch.

Saves raw float32 binary + JSON manifest for C# unit tests to compare against.
Test fixtures go to samples/data/torch-comparison/.

Usage: python gen_reference.py
"""
import os, json, struct
import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TEST_DIR = os.path.join(REPO_ROOT, "samples", "data", "torch-comparison")


def save_tensor(path, tensor):
    """Save a torch tensor as raw float32 binary."""
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    arr.tofile(path)
    return arr.shape


def run():
    os.makedirs(TEST_DIR, exist_ok=True)
    manifest = {}

    # Deterministic RNG
    rng = torch.Generator()
    rng.manual_seed(42)

    # =========================================================================
    # Conv2d tests
    # =========================================================================
    cases = [
        # (name, in_ch, out_ch, k, s, p, groups, input_shape)
        ("conv2d_3x3_s1_p1",    3,  16, 3, 1, 1, 1,  (1, 3, 7, 7)),
        ("conv2d_1x1_s1_p0",    3,  32, 1, 1, 0, 1,  (1, 3, 7, 7)),
        ("conv2d_depthwise",   16,  16, 3, 1, 1, 16, (1, 16, 5, 5)),
        ("conv2d_stride2",      3,  32, 3, 2, 1, 1,  (1, 3, 14, 14)),
        ("conv2d_with_bias",    3,   8, 3, 1, 1, 1,  (1, 3, 4, 4)),
    ]

    for name, in_ch, out_ch, k, s, p, groups, inp_shape in cases:
        conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, groups=groups, bias=True)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = conv(inp)

        inp_np = inp.numpy().astype(np.float32)
        w_np = conv.weight.data.cpu().numpy().astype(np.float32)
        b_np = conv.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        w_np.tofile(os.path.join(TEST_DIR, f"{name}_weight.bin"))
        b_np.tofile(os.path.join(TEST_DIR, f"{name}_bias.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Conv2d",
            "input_shape": list(inp_shape),
            "weight_shape": list(w_np.shape),
            "bias_shape": list(b_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"in_channels": in_ch, "out_channels": out_ch,
                       "kernel_size": k, "stride": s, "padding": p, "groups": groups},
        }
        print(f"  {name}: input={inp_shape} weight={w_np.shape} output={out_np.shape}")

    # =========================================================================
    # BatchNorm2d tests (eval mode with running stats)
    # =========================================================================
    bn_cases = [
        # (name, num_features, input_shape)
        ("bn2d_16ch",   16, (1, 16, 5, 5)),
        ("bn2d_3ch",     3, (1, 3, 7, 7)),
        ("bn2d_batch4", 16, (4, 16, 8, 8)),  # batch > 1 to verify running stats != batch stats
    ]

    for name, nf, inp_shape in bn_cases:
        bn = nn.BatchNorm2d(nf)
        inp = torch.randn(inp_shape, generator=rng)

        # Set known running stats
        bn.running_mean.copy_(torch.randn(nf, generator=rng) * 0.5)
        bn.running_var.copy_(torch.rand(nf, generator=rng) + 0.5)
        bn.weight.data.copy_(torch.randn(nf, generator=rng) * 0.1 + 1.0)
        bn.bias.data.copy_(torch.randn(nf, generator=rng) * 0.1)

        bn.eval()

        with torch.no_grad():
            out = bn(inp)

        inp_np = inp.numpy().astype(np.float32)
        gamma_np = bn.weight.data.cpu().numpy().astype(np.float32)
        beta_np = bn.bias.data.cpu().numpy().astype(np.float32)
        rm_np = bn.running_mean.cpu().numpy().astype(np.float32)
        rv_np = bn.running_var.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        gamma_np.tofile(os.path.join(TEST_DIR, f"{name}_gamma.bin"))
        beta_np.tofile(os.path.join(TEST_DIR, f"{name}_beta.bin"))
        rm_np.tofile(os.path.join(TEST_DIR, f"{name}_running_mean.bin"))
        rv_np.tofile(os.path.join(TEST_DIR, f"{name}_running_var.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "BatchNorm2d",
            "input_shape": list(inp_shape),
            "gamma_shape": list(gamma_np.shape),
            "beta_shape": list(beta_np.shape),
            "running_mean_shape": list(rm_np.shape),
            "running_var_shape": list(rv_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"num_features": nf, "eps": 1e-5},
        }
        print(f"  {name}: input={inp_shape} running_mean={rm_np.shape} output={out_np.shape}")

    # Also save what the batch-stats-only result would be (the bug case)
    for name, nf, inp_shape in bn_cases:
        bug_name = name + "_batch_stats"
        bn = nn.BatchNorm2d(nf)
        inp = torch.randn(inp_shape, generator=rng)

        # Set known running stats but also compute in TRAIN mode to show difference
        bn.running_mean.copy_(torch.randn(nf, generator=rng) * 0.5)
        bn.running_var.copy_(torch.rand(nf, generator=rng) + 0.5)
        bn.weight.data.copy_(torch.randn(nf, generator=rng) * 0.1 + 1.0)
        bn.bias.data.copy_(torch.randn(nf, generator=rng) * 0.1)

        # Use train mode -> batch stats (this is what Nivara currently does wrong)
        bn.train()

        with torch.no_grad():
            out_bug = bn(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_bug_np = out_bug.numpy().astype(np.float32)
        inp_np.tofile(os.path.join(TEST_DIR, f"{bug_name}_input.bin"))
        out_bug_np.tofile(os.path.join(TEST_DIR, f"{bug_name}_output.bin"))

        manifest[bug_name] = {
            "layer": "BatchNorm2d",
            "note": "batch_stats_only (no running stats) - the bug case",
            "input_shape": list(inp_shape),
            "output_shape": list(out_bug_np.shape),
        }
        print(f"  {bug_name}: input={inp_shape} output={out_bug_np.shape}")

    # =========================================================================
    # ReLU / ReLU6 tests
    # =========================================================================
    relu_cases = [
        ("relu_1d",  (32,)),
        ("relu_4d",  (1, 16, 8, 8)),
    ]

    for name, inp_shape in relu_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out_relu = torch.relu(inp)
        out_relu6 = torch.nn.functional.relu6(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_relu_np = out_relu.numpy().astype(np.float32)
        out_relu6_np = out_relu6.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_relu_np.tofile(os.path.join(TEST_DIR, f"{name}_relu_output.bin"))
        out_relu6_np.tofile(os.path.join(TEST_DIR, f"{name}_relu6_output.bin"))

        manifest[name] = {
            "layer": "ReLU/ReLU6",
            "input_shape": list(inp_shape),
            "relu_output_shape": list(out_relu_np.shape),
            "relu6_output_shape": list(out_relu6_np.shape),
        }
        print(f"  {name}: input={inp_shape}")

    # =========================================================================
    # MaxPool2d tests
    # =========================================================================
    pool_cases = [
        ("maxpool_3x3_s2_p1",  (1, 16, 14, 14), 3, 2, 1),
        ("maxpool_2x2_s2_p0",  (1, 32, 28, 28), 2, 2, 0),
    ]

    for name, inp_shape, k, s, p in pool_cases:
        pool = nn.MaxPool2d(k, stride=s, padding=p)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = pool(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "MaxPool2d",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"kernel_size": k, "stride": s, "padding": p},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # AdaptiveAvgPool2d tests
    # =========================================================================
    aap_cases = [
        ("adaptiveavgpool_1x1", (1, 512, 7, 7), 1),
        ("adaptiveavgpool_1x1_sm", (1, 32, 14, 14), 1),
    ]

    for name, inp_shape, out_size in aap_cases:
        pool = nn.AdaptiveAvgPool2d(out_size)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = pool(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "AdaptiveAvgPool2d",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"output_size": out_size},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Linear tests
    # =========================================================================
    linear_cases = [
        ("linear_128_64",     128, 64),
        ("linear_512_1000",   512, 1000),
    ]

    for name, in_f, out_f in linear_cases:
        lin = nn.Linear(in_f, out_f, bias=True)
        inp = torch.randn(1, in_f, generator=rng)

        with torch.no_grad():
            out = lin(inp)

        inp_np = inp.numpy().astype(np.float32)
        w_np = lin.weight.data.cpu().numpy().astype(np.float32)
        b_np = lin.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        w_np.tofile(os.path.join(TEST_DIR, f"{name}_weight.bin"))
        b_np.tofile(os.path.join(TEST_DIR, f"{name}_bias.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Linear",
            "input_shape": list(inp.shape),
            "weight_shape": list(w_np.shape),
            "bias_shape": list(b_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"in_features": in_f, "out_features": out_f},
        }
        print(f"  {name}: input={inp.shape} weight={w_np.shape} output={out_np.shape}")

    # =========================================================================
    # Write manifest
    # =========================================================================
    manifest_path = os.path.join(TEST_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Total test cases: {len(manifest)}")


if __name__ == "__main__":
    run()
