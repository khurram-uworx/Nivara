"""Generate ModernBERT-large base model reference last hidden states for C# comparison."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModel, AutoTokenizer

MAX_LENGTH = 128
TEXT = "Nivara runs ModernBERT-large on CPU in 2026, with 28 layers and sliding-window attention."


def main():
    model_dir = os.path.join(MODELS_DIR, "modernbert")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    with torch.no_grad():
        encoded = tokenizer(
            TEXT,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        outputs = model(**encoded, output_hidden_states=True)

    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]
    valid_len = int(attention_mask.sum().item())
    hidden = outputs.last_hidden_state[0].numpy()  # [128, 1024]

    print(f"Model: ModernBERT-large (base encoder, no heads)")
    print(f"attn implementation: {model.config._attn_implementation}")
    print(f"Input text: \"{TEXT}\"")
    print(f"input_ids (all {valid_len}): {input_ids[:valid_len].tolist()}")
    print(f"decoded: {tokenizer.decode(input_ids[:valid_len])}")
    print(f"Output shape: {hidden.shape}")
    print(f"Stats over valid positions: min={hidden[:valid_len].min():.6f}, max={hidden[:valid_len].max():.6f}, "
          f"mean={hidden[:valid_len].mean():.6f}, std={hidden[:valid_len].std():.6f}")
    print(f"Output[:10]: {[f'{v:.6f}' for v in hidden.flatten()[:10]]}")

    # Attention is bidirectional, so a padding row far enough from the valid region has every key
    # suppressed. What such a row yields is an artifact of the mask constant, and it is not a result
    # worth comparing: this side masks with finfo.min, so the row max is finite and the row softmax is
    # uniform (the output is the mean of the value vectors), while the C# side masks with -inf, gets a
    # row max of -inf, and its safe-softmax clamp returns zeros. Both are finite, and neither can reach
    # a valid position because a valid query never reads a pad key. The C# gate therefore diffs the
    # valid positions only and reports these counts on both sides.
    row_finite = np.isfinite(hidden).all(axis=1)
    non_finite_rows = int((~row_finite).sum())
    print(f"Non-finite rows: {non_finite_rows} (all at padding positions: "
          f"{bool(not row_finite[:valid_len].any())})")
    print(f"Non-finite values within the valid region: {int((~np.isfinite(hidden[:valid_len])).sum())}")

    hidden.astype(np.float32).tofile(os.path.join(model_dir, "last_hidden_state_py.bin"))
    input_ids.numpy().astype(np.int32).tofile(os.path.join(model_dir, "input_ids_py.bin"))

    # Per-stage references so the C# side can bisect the encoder and find the first divergent
    # layer instead of guessing. ModernBertModel returns the post-embedding-norm state, then one
    # state per layer, then the final-norm state.
    stages = torch.stack([h[0] for h in outputs.hidden_states]).numpy()
    stages.astype(np.float32).tofile(os.path.join(model_dir, "hidden_states_py.bin"))
    print(f"Per-stage shape: {stages.shape} (embeddings, {model.config.num_hidden_layers} layers, final norm)")

    with open(os.path.join(model_dir, "compare_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "text": TEXT,
                "max_length": MAX_LENGTH,
                "valid_len": valid_len,
                "hidden_shape": list(hidden.shape),
                "stages_shape": list(stages.shape),
                "attn_implementation": model.config._attn_implementation,
                "non_finite_rows": non_finite_rows,
            },
            handle,
            indent=2,
        )

    print(f"Saved last_hidden_state and input_ids to {model_dir}")


if __name__ == "__main__":
    main()
