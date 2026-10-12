"""ModernBERT-large inference benchmark (PyTorch side of the C# `modernbert benchmark` comparison).

Mirrors `ModernBert.RunBenchmark` deliberately: the same fixture sentence, the same two padded
lengths by default (128 and 256), then 3 timed passes reported as median and min. `--seq N[,N...]`
times a chosen subset of those lengths, exactly like the C# `--seq`; the default reproduces the
(128, 256) rows unchanged. Only the same-row PyTorch-vs-Nivara ratio is meaningful, and only if
both sides are run in the same session.

A length above 2048 has no CPU partner: the C# CPU path caps `--seq` at its dense `[L, L]` mask
limit, so the matching C# number for such a row is `modernbert --ilgpu benchmark --seq N` instead.

The two sides do not warm up identically: this script runs 3 untimed passes (`WARMUP_PASSES`),
the C# mode 1. The docstring here used to claim "one untimed pass ... deliberately", which
contradicted the constant two lines below it. Pass `--warmup 3` to the C# mode to match, or run
this with `WARMUP_PASSES = 1` to match the other way.

The sentence is `SampleSentences[0]` in the C# mode and `TEXT` in `modernbert_compare.py`, so the
26 valid tokens are identical on both sides.
"""
import argparse
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModel, AutoTokenizer

MAX_LENGTHS = (128, 256)
WARMUP_PASSES = 3
TEXT = "Nivara runs ModernBERT-large on CPU in 2026, with 28 layers and sliding-window attention."

_SEQ_ERROR = ('--seq expects positive integers, one value or a comma-separated list; '
              'got \'{token}\' in "{raw}".')


def _parse_seq_lengths(raw):
    """Parse a `--seq` value — one length or a comma-separated list — ascending and de-duplicated,
    so the printed table stays ordered however the list was typed. A token must be a positive
    integer, which also settles the `--seq 1,000` trap: the comma is the list separator, so that is
    the tokens `1` and `000`, and `000` is not a length. Mirrors the C# `Program.TryParseSeqLengths`.
    """
    lengths = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        # Mirror the C# accept set exactly: an optional leading sign, then ASCII digits. Python's
        # `int()` alone would also accept `1_28`, full-width `１２８`, and other Unicode digits that
        # `int.TryParse` with NumberStyles.Integer rejects — a silent divergence the issue's "same
        # surface" goal rules out.
        digits = token[1:] if token[:1] in ("+", "-") else token
        if not (digits.isascii() and digits.isdigit()):
            raise argparse.ArgumentTypeError(_SEQ_ERROR.format(token=token, raw=raw))
        length = int(token)
        if length <= 0:
            raise argparse.ArgumentTypeError(_SEQ_ERROR.format(token=token, raw=raw))
        lengths.append(length)

    if not lengths:
        raise argparse.ArgumentTypeError(
            f"--seq needs at least one sequence length; got \"{raw}\".")
    return sorted(set(lengths))


def main():
    ap = argparse.ArgumentParser(
        description="ModernBERT-large PyTorch inference benchmark "
                    "(the C# `modernbert benchmark` partner).")
    ap.add_argument(
        "--seq",
        type=_parse_seq_lengths,
        default=None,
        metavar="N[,N...]",
        help="Sequence lengths to time, one value or a comma-separated list, ascending and "
             "de-duplicated (mirrors the C# --seq). Defaults to "
             + ",".join(str(length) for length in MAX_LENGTHS)
             + " so the published rows are unchanged.")
    args = ap.parse_args()
    lengths = args.seq if args.seq is not None else list(MAX_LENGTHS)

    print("=== HuggingFace ModernBERT-large Inference (same weights as C#) ===")
    print()
    print(f"Sequence lengths: {', '.join(str(length) for length in lengths)}")
    print()

    model_dir = os.path.join(MODELS_DIR, "modernbert")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    num_layers = model.config.num_hidden_layers
    print(f"  Parameters: {param_count:,}")
    print(f"  attn implementation: {model.config._attn_implementation}")
    print(f"  Input text: \"{TEXT}\"")
    print()

    outputs = None
    for max_length in lengths:
        encoded = tokenizer(
            TEXT,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        valid_len = int(encoded["attention_mask"].sum().item())

        for _ in range(WARMUP_PASSES):
            with torch.no_grad():
                model(**encoded)

        timings = []
        for _ in range(3):
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(**encoded)
            timings.append((time.perf_counter() - start) * 1000)

        median = statistics.median(timings)
        print(
            f"seq={max_length:4d} (valid {valid_len:3d})  median {median:8.1f} ms  "
            f"min {min(timings):8.1f} ms  {valid_len / (median / 1000.0):9.1f} tok/s  "
            f"~{median / num_layers:.2f} ms/layer"
        )

    print()
    print("Each pass encodes the full padded sequence — this model has no KV cache, "
          "since an encoder re-reads the whole sequence every time.")

    hidden = outputs.last_hidden_state[0].numpy()
    valid_len = int(encoded["attention_mask"].sum().item())
    print(f"Output shape: {hidden.shape}")
    print(
        f"Stats over valid positions: min={hidden[:valid_len].min():.6f}, "
        f"max={hidden[:valid_len].max():.6f}, mean={hidden[:valid_len].mean():.6f}, "
        f"std={hidden[:valid_len].std():.6f}"
    )
    print(f"Non-finite values within the valid region: {int((~np.isfinite(hidden[:valid_len])).sum())}")


if __name__ == "__main__":
    main()
