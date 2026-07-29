"""Compare MiniLM Python vs C# sentence embeddings."""
import os, sys, time, struct
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoTokenizer, AutoModel


def main():
    model_dir = os.path.join(MODELS_DIR, "minilm")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    sentences = [
        "This is a cat.",
        "This is a dog.",
        "I love programming.",
        "The weather is nice today.",
        "I love coding."
    ]

    print(f"Sentences ({len(sentences)}):")
    for i, s in enumerate(sentences):
        print(f"  [{i}] {s}")
    print()

    with torch.no_grad():
        embeddings_list = []
        times = []
        for s in sentences:
            encoded = tokenizer(s, padding=True, truncation=True, max_length=128, return_tensors="pt")
            sw = time.perf_counter()
            outputs = model(**encoded)
            elapsed = time.perf_counter() - sw
            times.append(elapsed * 1000)
            # CLS token pooling (index 0) + L2 normalize
            emb = outputs.last_hidden_state[:, 0, :].numpy().flatten()
            emb = emb / np.linalg.norm(emb)
            embeddings_list.append(emb)

        embeddings = np.array(embeddings_list)

    avg_ms = np.mean(times)
    print(f"Forward pass: {sum(times):.1f} ms total, {avg_ms:.1f} ms/sentence")
    print(f"Embedding dim: {embeddings.shape[1]}")
    print()

    for i in range(len(sentences)):
        emb = embeddings[i]
        norm = np.linalg.norm(emb)
        print(f"[{i}] {sentences[i]}")
        print(f"    first 10: [{', '.join(f'{v:.6f}' for v in emb[:10])}]")
        print(f"    stats: min={emb.min():.6f}, max={emb.max():.6f}, mean={emb.mean():.6f}, L2 norm={norm:.6f}")
        print()

    # Cosine similarity matrix
    print("Cosine Similarity Matrix:")
    print("       ", end="")
    for i in range(len(sentences)):
        print(f"  [{i}]   ", end="")
    print()
    for i in range(len(sentences)):
        print(f"  [{i}]  ", end="")
        for j in range(len(sentences)):
            ei, ej = embeddings[i], embeddings[j]
            sim = float(np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej)))
            print(f"{sim:7.4f} ", end="")
        print()
    print()

    # Save embeddings for C# comparison
    save_path = os.path.join(MODELS_DIR, "compare_minilm_embeddings_py.bin")
    embeddings.astype(np.float32).tofile(save_path)
    print(f"Saved Python embeddings to {save_path}")


if __name__ == "__main__":
    main()
