"""MiniLM inference benchmark (same methodology as C# Nivara sample)."""
import time, os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR
from transformers import AutoTokenizer, AutoModel


def main():
    print("=== HuggingFace MiniLM Inference (same weights as C#) ===")
    print()

    model_dir = os.path.join(MODELS_DIR, "minilm")
    print(f"Loading tokenizer and model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")
    print()

    text = "This is a long test sentence that will be tokenized to demonstrate the performance of the MiniLM model inference across multiple tokens for benchmarking purposes."

    encoded = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    print(f"Input text length: {len(text.split())} words")
    print(f"Input tokens: {encoded['input_ids'].shape[1]}")
    print()

    # Warmup
    print("Warmup (3 passes)...")
    with torch.no_grad():
        for _ in range(3):
            model(**encoded)

    # Benchmark
    print("Benchmarking (10 passes)...")
    times = []
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**encoded)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"  Average: {avg:.1f} ms")
    print(f"  Min:     {min(times):.0f} ms")
    print(f"  Max:     {max(times):.0f} ms")
    print()

    # Verify output
    emb = outputs.last_hidden_state[:, 0, :].numpy().flatten()
    emb = emb / np.linalg.norm(emb)
    print(f"Output embedding dim: {len(emb)}")
    print(f"L2 norm: {np.linalg.norm(emb):.6f}")


if __name__ == "__main__":
    main()
