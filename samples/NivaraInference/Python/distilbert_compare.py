"""Generate DistilBERT base model reference hidden states for C# comparison."""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModel, AutoTokenizer


def main():
    model_dir = os.path.join(MODELS_DIR, "distilbert")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    text = "This is a test sentence."

    with torch.no_grad():
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs = model(**encoded)

    last_hidden = outputs.last_hidden_state  # [1, 128, 768]
    hidden = last_hidden[0].numpy()

    print(f"Model: distilbert-base-uncased (base encoder, no heads)")
    print(f"Input text: \"{text}\"")
    print(f"Input ids (first 10): {encoded['input_ids'][0, :10].tolist()}")
    print(f"Output shape: {hidden.shape}")
    print(f"Stats: min={hidden.min():.6f}, max={hidden.max():.6f}, mean={hidden.mean():.6f}, std={hidden.std():.6f}")
    print(f"Output[:10]: {[f'{v:.6f}' for v in hidden.flatten()[:10]]}")
    print()

    save_path = os.path.join(model_dir, "last_hidden_state_py.bin")
    hidden.astype(np.float32).tofile(save_path)
    print(f"Saved last_hidden_state to {save_path}")


if __name__ == "__main__":
    main()
