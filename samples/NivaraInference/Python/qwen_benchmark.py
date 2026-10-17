"""Qwen2.5-0.5B-Instruct greedy-decode benchmark (PyTorch side of the C# `qwen benchmark` comparison).

Mirrors `NivaraInference -- qwen benchmark` / `-- qwen benchmark --plain` deliberately: the
same (rendered, chat-templated) prompt, greedy decoding, and the same timing discipline
(one warm-up followed by 3 measured passes, reported as median milliseconds and tok/s).
Only the same-protocol PyTorch-vs-Nivara ratio is meaningful, and only if both sides are
run in the same session without contending for the CPU/GPU.

PyTorch/HuggingFace uses its own highly optimized attention and KV cache internally, so the
PyTorch numbers here mark the distance to the library's hand-written paths; they are not a gate.

This script prefers the plain (no-tools) prompt surface so the PyTorch reference can use the
standard `generate()` API without an interleaved tool-call turn.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR_NAME = "qwen2.5-0.5b-instruct"
PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 32
WARMUP_PASSES = 1
TIMED_PASSES = 3


def render_prompt(tokenizer):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": PROMPT},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    except Exception:
        return PROMPT


def main():
    ap = argparse.ArgumentParser(
        description="Qwen2.5-0.5B-Instruct PyTorch greedy-decode benchmark "
                    "(partner to the C# `qwen benchmark`).")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                    help="number of greedy tokens to generate per run (default 32)")
    ap.add_argument("--warmup", type=int, default=WARMUP_PASSES,
                    help="number of untimed warm-up passes (default 1)")
    ap.add_argument("--runs", type=int, default=TIMED_PASSES,
                    help="number of timed passes (default 3)")
    args = ap.parse_args()

    model_dir = os.path.join(MODELS_DIR, MODEL_DIR_NAME)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True,
                                                trust_remote_code=True)
    model.eval()

    prompt = render_prompt(tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[1])

    param_count = sum(p.numel() for p in model.parameters())
    print("=== Qwen2.5-0.5B-Instruct greedy decode (PyTorch / HuggingFace) ===")
    print(f"Device: CPU  Precision: float32  Strategy: HuggingFace generate (KV cache)")
    print(f"Model:  {model_dir}")
    print(f"Parameters: {param_count:,}")
    print(f"Prompt: \"{PROMPT}\" ({input_len} rendered tokens)")
    print(f"Generating {args.max_new_tokens} new tokens per run")
    print()

    with torch.no_grad():
        for _ in range(args.warmup):
            model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                           do_sample=False, use_cache=True)

    timings_ms = []
    sequences = []
    with torch.no_grad():
        for _ in range(args.runs):
            start = time.perf_counter()
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, use_cache=True)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)
            sequences.append(out[0][input_len:].tolist())

    print(f"Warmup: {args.warmup} pass(es)")
    for i, ms in enumerate(timings_ms, 1):
        toks = len(sequences[i - 1])
        print(f"Run {i}: {ms:7.0f} ms ({toks} tokens, "
              f"{toks * 1000.0 / max(ms, 1e-9):.2f} tok/s)")
    median_ms = sorted(timings_ms)[len(timings_ms) // 2]
    med_toks = len(sequences[sorted(range(len(timings_ms)), key=lambda i: timings_ms[i])[len(timings_ms) // 2]])
    print()
    print(f"Median: {median_ms:7.0f} ms ({med_toks} tokens, "
          f"{med_toks * 1000.0 / max(median_ms, 1e-9):.2f} tok/s)")
    print()
    print("Token ids (median run):", sequences[sorted(range(len(timings_ms)), key=lambda i: timings_ms[i])[len(timings_ms) // 2]])


if __name__ == "__main__":
    main()
