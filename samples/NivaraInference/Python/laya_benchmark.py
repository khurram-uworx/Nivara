"""Laya (PyTorch/wheel) inference benchmark, same methodology as the C# `laya benchmark` mode.

Mirrors `Laya.RunBenchmark` deliberately: the same four fixture questions, the same
padding to the config's `max_len`, one untimed warmup pass, then 3 timed passes of the
whole model (encoder **and** head) reported as median/min/max. Only the same-row
ratio is meaningful, and only if both sides run in the same session.

The questions, the state, the wheel, and the model construction are imported from
`laya_compare` rather than restated, so the two scripts cannot drift apart: a change
to the fixture set moves the correctness gate and this timing in the same commit, or
not at all. Nothing here writes fixtures -- it reads the checkpoint and times it.

The C# side is single-threaded per op, while PyTorch defaults to one thread per
physical core, so the printed thread count is part of the result, not a detail.

Deliberately NOT pinned: do not set ``torch.set_num_threads(1)`` and do not set
``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` to 1 here. The Nivara CPU path is
single-threaded per op, so pinning PyTorch to one thread would hand us a several-fold
"win" that is really just PyTorch's matmul parallelism being switched off -- the
comparison would flatter the port and mean nothing. The number to record is the
default, all-cores PyTorch result, against which the iGPU figure is honestly
awkward. Report the thread count next to the timing, always.
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))

# Imports the wheel loader, the four questions, the shared STATE, and the
# `repo_root` helper. Re-executes its stdout wrapper, which is harmless.
from laya_compare import QUESTIONS, STATE, download_wheel, load_common, repo_root

WARMUP_PASSES = 1
TIMED_PASSES = 3


def main():
    import tempfile

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_dir = os.path.join(repo_root(), "samples", "data", "laya")
    if not os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        raise SystemExit(f"No checkpoint at {model_dir}; nothing to time.")

    print("=== Laya PyTorch (laya 0.3.20 wheel) Inference ===")
    print()
    print(f"  torch {torch.__version__}")
    cores = os.cpu_count() or 1
    threads = torch.get_num_threads()
    print(f"  threads {threads} (interop {torch.get_num_interop_threads()}), "
          f"ProcessorCount {cores}")
    if threads == 1 and cores > 1:
        print("  WARNING: PyTorch is pinned to one thread while the machine has "
              f"{cores}. Nivara's CPU path is single-threaded per op, so this "
              "configuration measures PyTorch with its matmul parallelism disabled "
              "and is not a usable comparison. Unset OMP_NUM_THREADS / "
              "MKL_NUM_THREADS and re-run.")

    cache = os.path.join(tempfile.gettempdir(), "opencode", "laya_wheel_cache")
    try:
        wheel_path = download_wheel(cache)
    except Exception as e:
        raise SystemExit(f"Could not obtain the wheel ({e}).") from e
    common = load_common(wheel_path)
    print(f"  Loaded {common.__file__}")

    import json

    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_dir, "tokenizer"), local_files_only=True)

    # Timed separately so it is comparable with the C# `Load weights` line.
    build_start = time.perf_counter()
    encoder_dir = os.path.join(model_dir, "encoder")
    model = common.build_model(cfg, encoder_dir=encoder_dir, pretrained=True)
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.eval()
    build_ms = (time.perf_counter() - build_start) * 1000
    print(f"  Loaded checkpoint ({len(weights)} tensors), "
          f"attn={model.encoder.config._attn_implementation}")
    print(f"  Model build: {build_ms:.0f} ms")
    print()

    max_len = cfg["max_len"]
    head_max_len = cfg["head_max_len"]
    print(f"Encoder: {model.encoder.config.num_hidden_layers} layers, "
          f"d={model.encoder.config.hidden_size}, "
          f"heads={model.encoder.config.num_attention_heads}")
    print(f"Head max_len: {head_max_len}, model max_len: {max_len}")
    print()
    # The pad id comes from the encoder config, as the C# `PadTo` call site does --
    # the agent config does not carry it.
    with open(os.path.join(encoder_dir, "config.json"), encoding="utf-8") as f:
        pad_id = json.load(f)["pad_token_id"]

    print(f"  Pad token id: {pad_id}")
    print()
    print(f"{WARMUP_PASSES} warmup + {TIMED_PASSES} timed passes at max_len {max_len}, "
          "median reported.")
    print()
    print(f"  {'question':<10} {'valid':>5} {'markers':>7} {'median':>10} {'min':>8} {'max':>8}")

    rows = []

    for qid, q in QUESTIONS:
        ids, markers = common.build_sequence(
            tokenizer, STATE, q, max_len=max_len, head_max_len=head_max_len)
        valid_length = len(ids)

        # Same as C# `PadTo`: pad to max_len, carry valid_length so the mask
        # covers the pad tail rather than the tokens attending to it.
        padded = list(ids) + [pad_id] * (max_len - valid_length)
        ids_t = torch.tensor([padded], dtype=torch.long)
        attn = torch.zeros(1, max_len, dtype=torch.long)
        attn[0, :valid_length] = 1
        mpos = torch.tensor([markers], dtype=torch.long)
        mmask = torch.ones(1, len(markers), dtype=torch.bool)
        qtype = torch.tensor([common.QTYPES[q["t"]]], dtype=torch.long)

        args = (ids_t, attn, mpos, mmask, qtype)

        # The C# warmup runs the encoder only; mirrored here so the two protocols
        # are describable by the same sentence.
        for _ in range(WARMUP_PASSES):
            with torch.no_grad():
                model.encoder(input_ids=ids_t, attention_mask=attn)

        timings = []
        for _ in range(TIMED_PASSES):
            start = time.perf_counter()
            with torch.no_grad():
                model(*args)
            timings.append((time.perf_counter() - start) * 1000)

        samples = sorted(timings)
        median = statistics.median(timings)
        rows.append(median)
        print(f"  {qid:<10} {valid_length:>5} {len(markers):>7} "
              f"{median:>8.0f} ms {min(timings):>6.0f} ms {max(timings):>6.0f} ms")

    print()
    print(f"  mean of medians: {statistics.mean(rows):.0f} ms")
    print("Encoder plus head, one pass each, no KV cache -- cost scales with the "
          "padded sequence length, not with tokens generated.")


if __name__ == "__main__":
    main()
