"""Generate reference tensors for each NN layer type using PyTorch.

Saves raw float32 binary + JSON manifest for C# unit tests to compare against.
Test fixtures go to samples/data/torch-comparison/.

Reproducibility:
  - Verified with Python 3.12, torch 2.13.0+cpu, numpy 1.26.
  - Fixtures are RNG-derived. A single shared torch.Generator is seeded with
    42 (see run()) and every random draw uses generator=rng. Weights are
    created by nn.* modules, which consume from the global torch RNG, so
    CPU-only execution is required for bit-stable output. Add a new draw to
    the end of a case, never insert one mid-stream, or every subsequent
    fixture changes and the C# manifest must be regenerated together.
  - Regenerate after upgrading torch/numpy: run `python gen_reference.py` and
    commit the full samples/data/torch-comparison/ tree as one unit.

Usage: python gen_reference.py
"""
import os, json, struct, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEST_DIR = os.path.join(REPO_ROOT, "samples", "data", "torch-comparison")


def save_tensor(path, tensor):
    """Save a torch tensor as raw float32 binary."""
    arr = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    arr.tofile(path)
    return arr.shape


def run():
    os.makedirs(TEST_DIR, exist_ok=True)
    manifest = {}

    print(f"torch {torch.__version__} | numpy {np.__version__} | python {sys.version.split()[0]}")
    print(f"RNG seed: 42 (single shared torch.Generator)\n")

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
    # Conv1d tests
    # =========================================================================
    conv1d_cases = [
        # (name, in_ch, out_ch, k, s, p, input_shape)
        ("conv1d_k3",    8,  8, 3, 1, 1, (1, 8, 16)),
        ("conv1d_k5",    8, 16, 5, 1, 2, (1, 8, 16)),
        ("conv1d_k7",    4,  8, 7, 1, 3, (1, 4, 32)),
        ("conv1d_s2",    8, 16, 3, 2, 1, (1, 8, 16)),
    ]

    for name, in_ch, out_ch, k, s, p, inp_shape in conv1d_cases:
        conv = nn.Conv1d(in_ch, out_ch, k, stride=s, padding=p, bias=True)
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
            "layer": "Conv1d",
            "input_shape": list(inp_shape),
            "weight_shape": list(w_np.shape),
            "bias_shape": list(b_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"in_channels": in_ch, "out_channels": out_ch,
                       "kernel_size": k, "stride": s, "padding": p},
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
    # BatchNorm1d tests (eval mode with running stats)
    # =========================================================================
    bn1d_cases = [
        # (name, num_features, input_shape)
        ("bn1d_2d",  16, (4, 16)),
        ("bn1d_3d",   8, (2, 8, 20)),
    ]

    for name, nf, inp_shape in bn1d_cases:
        bn = nn.BatchNorm1d(nf)
        inp = torch.randn(inp_shape, generator=rng)

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
            "layer": "BatchNorm1d",
            "input_shape": list(inp_shape),
            "gamma_shape": list(gamma_np.shape),
            "beta_shape": list(beta_np.shape),
            "running_mean_shape": list(rm_np.shape),
            "running_var_shape": list(rv_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"num_features": nf, "eps": 1e-5},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

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
    # LeakyReLU tests
    # =========================================================================
    leaky_cases = [
        ("leaky_relu_1d", (32,)),
        ("leaky_relu_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in leaky_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = F.leaky_relu(inp, negative_slope=0.01)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "LeakyReLU",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"negative_slope": 0.01},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Sigmoid tests
    # =========================================================================
    sigmoid_cases = [
        ("sigmoid_1d", (32,)),
        ("sigmoid_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in sigmoid_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = torch.sigmoid(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Sigmoid",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Tanh tests
    # =========================================================================
    tanh_cases = [
        ("tanh_1d", (32,)),
        ("tanh_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in tanh_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = torch.tanh(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "Tanh",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # GELU tests
    # "gelu_*" fixtures use the tanh approximation (PyTorch F.gelu approximate="tanh"),
    # "gelu_exact_*" fixtures use the exact erf-based GELU (F.gelu default).
    # The exact cases use a dedicated RNG so the main rng stream (and therefore all
    # other fixtures) is unaffected.
    # =========================================================================
    gelu_cases = [
        ("gelu_1d", (32,)),
        ("gelu_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in gelu_cases:
        inp = torch.randn(inp_shape, generator=rng)
        out = F.gelu(inp, approximate="tanh")

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "GELU (tanh)",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    gelu_exact_rng = torch.Generator().manual_seed(101)
    gelu_exact_cases = [
        ("gelu_exact_1d", (32,)),
        ("gelu_exact_4d", (1, 8, 4, 4)),
    ]

    for name, inp_shape in gelu_exact_cases:
        inp = torch.randn(inp_shape, generator=gelu_exact_rng)
        out = F.gelu(inp)

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "GELU (exact)",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

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
    # Embedding tests
    # =========================================================================
    emb_vocab = 100
    emb_dim = 16
    emb_weight = torch.randn(emb_vocab, emb_dim, generator=rng)

    # Single token lookup
    single_idx = torch.tensor([42])
    single_out = emb_weight[42]

    single_idx.numpy().astype(np.int32).tofile(os.path.join(TEST_DIR, "emb_single_input.bin"))
    emb_weight.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_single_weight.bin"))
    single_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_single_output.bin"))

    manifest["emb_single"] = {
        "layer": "Embedding",
        "input_shape": [1],
        "weight_shape": list(emb_weight.shape),
        "output_shape": list(single_out.shape),
        "params": {"num_embeddings": emb_vocab, "embedding_dim": emb_dim},
    }
    print(f"  emb_single: input=[1] weight={emb_weight.shape} output={single_out.shape}")

    # Batch lookup
    batch_idx = torch.tensor([0, 13, 42, 99])
    batch_out = emb_weight[batch_idx]

    batch_idx.numpy().astype(np.int32).tofile(os.path.join(TEST_DIR, "emb_batch_input.bin"))
    emb_weight.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_batch_weight.bin"))
    batch_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "emb_batch_output.bin"))

    manifest["emb_batch"] = {
        "layer": "Embedding",
        "input_shape": [4],
        "weight_shape": list(emb_weight.shape),
        "output_shape": list(batch_out.shape),
        "params": {"num_embeddings": emb_vocab, "embedding_dim": emb_dim},
    }
    print(f"  emb_batch: input=[4] weight={emb_weight.shape} output={batch_out.shape}")

    # =========================================================================
    # Dropout tests (eval mode = passthrough)
    # =========================================================================
    drop_inp = torch.randn(4, 32, generator=rng)
    drop_out = F.dropout(drop_inp, p=0.5, training=False)

    drop_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "dropout_eval_input.bin"))
    drop_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "dropout_eval_output.bin"))

    manifest["dropout_eval"] = {
        "layer": "Dropout",
        "input_shape": list(drop_inp.shape),
        "output_shape": list(drop_out.shape),
        "params": {"p": 0.5, "training": False},
    }
    print(f"  dropout_eval: input={drop_inp.shape} output={drop_out.shape}")

    # =========================================================================
    # RMSNorm tests (no learnable params)
    # =========================================================================
    rms_cases = [
        ("rmsnorm_2d", (4, 32)),
        ("rmsnorm_3d", (2, 4, 32)),
    ]

    for name, inp_shape in rms_cases:
        inp = torch.randn(inp_shape, generator=rng)
        # PyTorch RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        # But Nivara's RMSNorm has no learnable weight — it's just normalization
        eps = 1e-5
        rms = torch.sqrt(torch.mean(inp ** 2, dim=-1, keepdim=True) + eps)
        out = inp / rms

        inp_np = inp.numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "RMSNorm",
            "input_shape": list(inp_shape),
            "output_shape": list(out_np.shape),
            "params": {"eps": eps},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # LayerNorm tests (with learnable params)
    # =========================================================================
    ln_cases = [
        ("layernorm_2d", (4, 32), 32),
        ("layernorm_3d", (2, 4, 32), 32),
    ]

    for name, inp_shape, norm_shape in ln_cases:
        ln = nn.LayerNorm(norm_shape)
        inp = torch.randn(inp_shape, generator=rng)

        with torch.no_grad():
            out = ln(inp)

        inp_np = inp.numpy().astype(np.float32)
        gamma_np = ln.weight.data.cpu().numpy().astype(np.float32)
        beta_np = ln.bias.data.cpu().numpy().astype(np.float32)
        out_np = out.numpy().astype(np.float32)

        inp_np.tofile(os.path.join(TEST_DIR, f"{name}_input.bin"))
        gamma_np.tofile(os.path.join(TEST_DIR, f"{name}_gamma.bin"))
        beta_np.tofile(os.path.join(TEST_DIR, f"{name}_beta.bin"))
        out_np.tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))

        manifest[name] = {
            "layer": "LayerNorm",
            "input_shape": list(inp_shape),
            "gamma_shape": list(gamma_np.shape),
            "beta_shape": list(beta_np.shape),
            "output_shape": list(out_np.shape),
            "params": {"normalized_shape": norm_shape, "eps": 1e-5},
        }
        print(f"  {name}: input={inp_shape} output={out_np.shape}")

    # =========================================================================
    # Softmax tests
    # =========================================================================
    softmax_inp = torch.randn(4, 10, generator=rng)
    softmax_out = F.softmax(softmax_inp, dim=1)

    softmax_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "softmax_input.bin"))
    softmax_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "softmax_output.bin"))

    manifest["softmax"] = {
        "layer": "Softmax",
        "input_shape": list(softmax_inp.shape),
        "output_shape": list(softmax_out.shape),
        "params": {"dim": 1},
    }
    print(f"  softmax: input={softmax_inp.shape} output={softmax_out.shape}")

    # =========================================================================
    # LogSoftmax tests
    # =========================================================================
    logsoftmax_inp = torch.randn(4, 10, generator=rng)
    logsoftmax_out = F.log_softmax(logsoftmax_inp, dim=1)

    logsoftmax_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "log_softmax_input.bin"))
    logsoftmax_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "log_softmax_output.bin"))

    manifest["log_softmax"] = {
        "layer": "LogSoftmax",
        "input_shape": list(logsoftmax_inp.shape),
        "output_shape": list(logsoftmax_out.shape),
        "params": {"dim": 1},
    }
    print(f"  log_softmax: input={logsoftmax_inp.shape} output={logsoftmax_out.shape}")

    # =========================================================================
    # MatMul tests
    # =========================================================================
    matmul_a = torch.randn(4, 8, generator=rng)
    matmul_b = torch.randn(8, 16, generator=rng)
    matmul_out = torch.matmul(matmul_a, matmul_b)

    matmul_a.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_a.bin"))
    matmul_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_b.bin"))
    matmul_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_output.bin"))

    manifest["matmul"] = {
        "layer": "MatMul",
        "a_shape": list(matmul_a.shape),
        "b_shape": list(matmul_b.shape),
        "output_shape": list(matmul_out.shape),
    }
    print(f"  matmul: a={matmul_a.shape} b={matmul_b.shape} output={matmul_out.shape}")

    # =========================================================================
    # BCEWithLogitsLoss tests
    # =========================================================================
    bce_inp = torch.randn(4, 10, generator=rng)
    bce_target = torch.rand(4, 10, generator=rng)

    bce_sum = F.binary_cross_entropy_with_logits(bce_inp, bce_target, reduction='sum')
    bce_mean = F.binary_cross_entropy_with_logits(bce_inp, bce_target, reduction='mean')

    bce_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_input.bin"))
    bce_target.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_target.bin"))
    torch.tensor([bce_sum.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_sum_output.bin"))
    torch.tensor([bce_mean.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "bce_with_logits_mean_output.bin"))

    manifest["bce_with_logits"] = {
        "layer": "BCEWithLogitsLoss",
        "input_shape": list(bce_inp.shape),
        "target_shape": list(bce_target.shape),
        "sum_output_shape": [1],
        "mean_output_shape": [1],
    }
    print(f"  bce_with_logits: input={bce_inp.shape} sum={bce_sum.item():.6f} mean={bce_mean.item():.6f}")

    # =========================================================================
    # CrossEntropyLoss tests
    # =========================================================================
    ce_inp = torch.randn(4, 10, generator=rng)
    ce_target = torch.tensor([0, 3, 7, 2])

    ce_out = F.cross_entropy(ce_inp, ce_target)

    ce_inp.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "cross_entropy_input.bin"))
    ce_target.numpy().astype(np.int64).tofile(os.path.join(TEST_DIR, "cross_entropy_target.bin"))
    torch.tensor([ce_out.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "cross_entropy_output.bin"))

    manifest["cross_entropy"] = {
        "layer": "CrossEntropyLoss",
        "input_shape": list(ce_inp.shape),
        "target_shape": list(ce_target.shape),
        "output_shape": [1],
    }
    print(f"  cross_entropy: input={ce_inp.shape} target={ce_target.shape} loss={ce_out.item():.6f}")

    # =========================================================================
    # MSELoss tests
    # =========================================================================
    mse_pred = torch.randn(4, 10, generator=rng)
    mse_target = torch.randn(4, 10, generator=rng)

    mse_sum = F.mse_loss(mse_pred, mse_target, reduction='sum')
    mse_mean = F.mse_loss(mse_pred, mse_target, reduction='mean')

    mse_pred.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_pred.bin"))
    mse_target.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_target.bin"))
    torch.tensor([mse_sum.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_sum_output.bin"))
    torch.tensor([mse_mean.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "mse_loss_mean_output.bin"))

    manifest["mse_loss"] = {
        "layer": "MSELoss",
        "pred_shape": list(mse_pred.shape),
        "target_shape": list(mse_target.shape),
        "sum_output_shape": [1],
        "mean_output_shape": [1],
    }
    print(f"  mse_loss: pred={mse_pred.shape} sum={mse_sum.item():.6f} mean={mse_mean.item():.6f}")

    # =========================================================================
    # L1Loss tests
    # =========================================================================
    l1_pred = torch.randn(4, 10, generator=rng)
    l1_target = torch.randn(4, 10, generator=rng)

    l1_out = F.l1_loss(l1_pred, l1_target, reduction='sum')

    l1_pred.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_pred.bin"))
    l1_target.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_target.bin"))
    torch.tensor([l1_out.item()]).numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "l1_loss_output.bin"))

    manifest["l1_loss"] = {
        "layer": "L1Loss",
        "pred_shape": list(l1_pred.shape),
        "target_shape": list(l1_target.shape),
        "output_shape": [1],
    }
    print(f"  l1_loss: pred={l1_pred.shape} sum={l1_out.item():.6f}")

    # =========================================================================
    # AddBias tests (row-broadcast bias addition, linear bias op)
    # Uses a dedicated RNG so the main stream (and every other fixture) is
    # bit-stable. Computes a + b where b is broadcast across rows.
    # =========================================================================
    ops_rng = torch.Generator().manual_seed(202)

    add_bias_a = torch.randn(4, 16, generator=ops_rng)
    add_bias_b = torch.randn(16, generator=ops_rng)
    add_bias_out = add_bias_a + add_bias_b

    add_bias_a.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "add_bias_a.bin"))
    add_bias_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "add_bias_b.bin"))
    add_bias_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "add_bias_output.bin"))

    manifest["add_bias"] = {
        "layer": "AddBias",
        "a_shape": list(add_bias_a.shape),
        "bias_shape": list(add_bias_b.shape),
        "output_shape": list(add_bias_out.shape),
    }
    print(f"  add_bias: a={add_bias_a.shape} bias={add_bias_b.shape} output={add_bias_out.shape}")

    # =========================================================================
    # MatMulTransposedB tests (inference a @ b^T, linear weight layout)
    # b is saved in [N, K] row-major — the raw nn.Linear weight layout the
    # kernel consumes without a transpose. Same dedicated RNG.
    # =========================================================================
    mmtb_a = torch.randn(4, 8, generator=ops_rng)
    mmtb_b = torch.randn(16, 8, generator=ops_rng)
    mmtb_out = torch.matmul(mmtb_a, mmtb_b.t())

    mmtb_a.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_transposed_b_a.bin"))
    mmtb_b.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_transposed_b_b.bin"))
    mmtb_out.numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, "matmul_transposed_b_output.bin"))

    manifest["matmul_transposed_b"] = {
        "layer": "MatMulTransposedB",
        "a_shape": list(mmtb_a.shape),
        "b_shape": list(mmtb_b.shape),
        "output_shape": list(mmtb_out.shape),
    }
    print(f"  matmul_transposed_b: a={mmtb_a.shape} b={mmtb_b.shape} output={mmtb_out.shape}")

    # =========================================================================
    # Fused multi-head attention tests (ReverseGradOperations.MultiHeadAttention)
    # Dedicated RNG keeps the main stream bit-stable. scale = 1/sqrt(headDim).
    # =========================================================================
    attn_rng = torch.Generator().manual_seed(303)

    def save_attn_case(name, q, k, v, scale, mask, dout, num_heads=4):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        d = q.shape[1]
        head_dim = d // num_heads
        heads = []
        for h in range(num_heads):
            qh = q[:, h * head_dim:(h + 1) * head_dim]
            kh = k[:, h * head_dim:(h + 1) * head_dim]
            vh = v[:, h * head_dim:(h + 1) * head_dim]
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(p, vh))
        out = torch.cat(heads, dim=-1)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)

        q.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_q.bin"))
        k.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_k.bin"))
        v.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_v.bin"))
        out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))
        if mask is not None:
            mask.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_mask.bin"))
        dout.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dout.bin"))
        dq.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dq.bin"))
        dk.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dk.bin"))
        dv.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dv.bin"))
        manifest[name] = {
            "layer": "MultiHeadAttention",
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "num_heads": num_heads,
            "scale": float(scale),
            "masked": mask is not None,
            "output_shape": list(out.shape),
        }
        print(f"  {name}: q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} output={list(out.shape)}")

    # Self-attention with a causal additive mask (qLen == kvLen == 4, D == 16).
    attn_q = torch.randn(4, 16, generator=attn_rng)
    attn_k = torch.randn(4, 16, generator=attn_rng)
    attn_v = torch.randn(4, 16, generator=attn_rng)
    attn_scale = 1.0 / math.sqrt(4)  # headDim = 16 / 4
    attn_mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)
    attn_dout = torch.randn(4, 16, generator=attn_rng)
    save_attn_case("attn_self_causal", attn_q, attn_k, attn_v, attn_scale, attn_mask, attn_dout)

    # Self-attention without a mask.
    save_attn_case("attn_self", attn_q, attn_k, attn_v, attn_scale, None, attn_dout)

    # Cross-attention (qLen != kvLen), last key/value is padding.
    attn_cq = torch.randn(3, 8, generator=attn_rng)
    attn_ck = torch.randn(5, 8, generator=attn_rng)
    attn_cv = torch.randn(5, 8, generator=attn_rng)
    attn_cscale = 1.0 / math.sqrt(4)  # headDim = 8 / 2
    attn_cmask = torch.zeros(3, 5)
    attn_cmask[:, 4] = float("-inf")
    attn_cdout = torch.randn(3, 8, generator=attn_rng)
    save_attn_case("attn_cross", attn_cq, attn_ck, attn_cv, attn_cscale, attn_cmask, attn_cdout, num_heads=2)

    # =========================================================================
    # Batched fused multi-head attention tests
    # (ReverseGradOperations.BatchedMultiHeadAttention, inputs are [B, L, D])
    # Same semantics as the single-sequence cases above but with a leading batch
    # dimension and a per-batch-element [B, qLen, kvLen] additive mask.
    # Appended at the END of the generation stream so the shared attn_rng stream
    # is untouched and every existing fixture stays bit-identical.
    # =========================================================================
    def save_batched_attn_case(name, q, k, v, scale, mask, dout, num_heads=4):
        q = q.detach().requires_grad_(True)
        k = k.detach().requires_grad_(True)
        v = v.detach().requires_grad_(True)
        d = q.shape[2]
        head_dim = d // num_heads
        heads = []
        for h in range(num_heads):
            qh = q[:, :, h * head_dim:(h + 1) * head_dim]
            kh = k[:, :, h * head_dim:(h + 1) * head_dim]
            vh = v[:, :, h * head_dim:(h + 1) * head_dim]
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
            if mask is not None:
                scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            heads.append(torch.matmul(p, vh))
        out = torch.cat(heads, dim=-1)
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), grad_outputs=dout)

        q.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_q.bin"))
        k.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_k.bin"))
        v.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_v.bin"))
        out.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_output.bin"))
        if mask is not None:
            mask.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_mask.bin"))
        dout.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dout.bin"))
        dq.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dq.bin"))
        dk.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dk.bin"))
        dv.detach().numpy().astype(np.float32).tofile(os.path.join(TEST_DIR, f"{name}_dv.bin"))
        manifest[name] = {
            "layer": "BatchedMultiHeadAttention",
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "num_heads": num_heads,
            "scale": float(scale),
            "masked": mask is not None,
            "output_shape": list(out.shape),
        }
        print(f"  {name}: q={list(q.shape)} k={list(k.shape)} v={list(v.shape)} output={list(out.shape)}")

    # Batched self-attention with a per-batch causal mask (B=2, L=4, D=16, H=4).
    bat_q = torch.randn(2, 4, 16, generator=attn_rng)
    bat_k = torch.randn(2, 4, 16, generator=attn_rng)
    bat_v = torch.randn(2, 4, 16, generator=attn_rng)
    bat_scale = 1.0 / math.sqrt(4)  # headDim = 16 / 4
    bat_mask = torch.triu(torch.full((1, 4, 4), float("-inf")), diagonal=1).repeat(2, 1, 1)
    bat_dout = torch.randn(2, 4, 16, generator=attn_rng)
    save_batched_attn_case("batched_attn_causal", bat_q, bat_k, bat_v, bat_scale, bat_mask, bat_dout)

    # Batched cross-attention (B=2, qLen=3, kvLen=5, D=8, H=2), last key padded.
    bat_cq = torch.randn(2, 3, 8, generator=attn_rng)
    bat_ck = torch.randn(2, 5, 8, generator=attn_rng)
    bat_cv = torch.randn(2, 5, 8, generator=attn_rng)
    bat_cscale = 1.0 / math.sqrt(4)  # headDim = 8 / 2
    bat_cmask = torch.zeros(2, 3, 5)
    bat_cmask[:, :, 4] = float("-inf")
    bat_cdout = torch.randn(2, 3, 8, generator=attn_rng)
    save_batched_attn_case("batched_attn_cross", bat_cq, bat_ck, bat_cv, bat_cscale, bat_cmask, bat_cdout, num_heads=2)

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
