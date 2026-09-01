"""Generate a SmolLM-135M-Instruct reference for the Nivara C# generation loop.

Run-once fixture for the Phase 3 A/B correctness diff against the Nivara BF16
causal-LM sample. Loads the locally-downloaded BF16 checkpoint into PyTorch,
greedily decodes a fixed prompt, and saves:

    samples/data/compare_smollm_py.bin   -- int32: prompt token ids followed by
                                           every greedily generated token id
    samples/data/compare_smollm_logits_py.bin -- float32: logits at the FINAL
                                           generated position (length = vocab_size)

The token-id stream is the argmax-agreement check; the final-position logits are
the numeric precision diff (a single forward step, so it doesn't compound).

The prompt and max_new_tokens must match what the C# sample uses so the two
sides diff the identical sequence.
"""
import os
import sys
import struct
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(MODELS_DIR, "smollm-135m")
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def main():
    ap = argparse.ArgumentParser(
        description="SmolLM-135M greedy reference + timing (bfloat16 default).")
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16",
                    help="Compute dtype: float32 (widened) or bfloat16 (native on disk). Default bfloat16.")
    args = ap.parse_args()
    dtype = _dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=dtype
    )
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    bytes_per = 2 if dtype == torch.bfloat16 else 4
    print(f"Parameters: {params:,}   weights (~{args.dtype}): "
          f"{params * bytes_per / 1024 / 1024:.1f} MB")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ids = tokenizer(PROMPT, return_tensors="pt")
    input_len = ids.input_ids.shape[1]
    print(f"Prompt: {PROMPT!r}")
    print(f"Input token ids ({input_len}): {ids.input_ids[0].tolist()}")

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model.generate(
            **ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,   # greedy -> deterministic argmax stream
        )
        gen_sec = time.perf_counter() - t0
        generated = out[0, input_len:].tolist()

    print(f"Generated {len(generated)} tokens in {gen_sec * 1000:.0f} ms "
          f"({gen_sec * 1000 / max(1, len(generated)):.0f} ms/token) "
          f"[PyTorch {args.dtype} CPU]")
    print(f"Generated token ids ({len(generated)}): {generated}")
    print(f"Decoded: {tokenizer.decode(out[0], skip_special_tokens=True)!r}")

    # Final-position logits for the numeric precision diff: feed input + all but
    # the last generated token so the model predicts the last generated token.
    full_prefix = out[0].tolist()
    prefix = torch.tensor([full_prefix[:-1]])
    with torch.no_grad():
        logits = model(prefix).logits  # [1, len, vocab]
    last_logits = logits[0, -1, :].float().numpy()  # [vocab]

    save_path = os.path.join(MODELS_DIR, "compare_smollm_py.bin")
    with open(save_path, "wb") as f:
        f.write(struct.pack("<i", input_len))
        np.asarray(full_prefix, dtype=np.int32).tofile(f)
    print(f"Saved token-id stream ({input_len} prompt + {len(generated)} generated) to {save_path}")

    logits_path = os.path.join(MODELS_DIR, "compare_smollm_logits_py.bin")
    with open(logits_path, "wb") as f:
        last_logits.tofile(f)
    print(f"Saved final-position logits ({last_logits.shape[0]} floats) to {logits_path}")
    print(f"Logits argmax: {int(np.argmax(last_logits))}  top-5: "
          f"{np.argsort(last_logits)[-5:][::-1].tolist()}")


if __name__ == "__main__":
    main()
