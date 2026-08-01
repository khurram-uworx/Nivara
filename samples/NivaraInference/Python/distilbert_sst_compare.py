"""Generate DistilBERT SST-2 reference logits + softmax probs for C# comparison.

Mirrors the sentence list in samples/NivaraInference/DistilBertSst.cs. Writes
the same binary layout the C# compare mode expects:
    int32 N, then N*2 float32 logits, then N*2 float32 softmax probs.
"""
import os
import sys
import struct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModelForSequenceClassification, AutoTokenizer

SENTENCES = [
    "This movie was an absolute joy from start to finish.",
    "A complete waste of time, boring and predictable.",
    "The acting was brilliant and the plot kept me on the edge of my seat.",
    "Terrible script, awful performances, I want my money back.",
    "An emotional masterpiece that will stay with you long after the credits.",
    "Not funny at all, the jokes fall completely flat.",
    "Visually stunning with a captivating story to match.",
    "Poorly paced and overlong, nothing happens for the first hour.",
]


def main():
    model_dir = os.path.join(MODELS_DIR, "distilbert_sst")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    logits = []
    probs = []
    with torch.no_grad():
        for sentence in SENTENCES:
            encoded = tokenizer(
                sentence,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            out = model(**encoded).logits[0]  # [2]
            p = torch.softmax(out, dim=-1)
            logits.append(out.numpy())
            probs.append(p.numpy())
            label = "NEGATIVE" if out[0] > out[1] else "POSITIVE"
            print(f"[{len(logits) - 1}] {label:8}  \"{sentence}\"")

    logits = np.asarray(logits, dtype=np.float32)  # [N, 2]
    probs = np.asarray(probs, dtype=np.float32)  # [N, 2]

    print(f"Model: distilbert-base-uncased-finetuned-sst-2-english")
    print(f"Sentences: {len(SENTENCES)}")
    print(f"Logits[:10]: {logits.flatten()[:10].tolist()}")

    save_path = os.path.join(MODELS_DIR, "compare_distilbert_sst_py.bin")
    with open(save_path, "wb") as f:
        f.write(struct.pack("<i", len(SENTENCES)))
        logits.tofile(f)
        probs.tofile(f)
    print(f"Saved logits + softmax probs to {save_path}")


if __name__ == "__main__":
    main()
