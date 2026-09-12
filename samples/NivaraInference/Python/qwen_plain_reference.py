"""Ground-truth fixture: Qwen2.5-0.5B-Instruct plain (no-tools) single-turn chat.

Run-once reference for the Nivara --qwen plain-prompt work (issue #408). Loads the
locally-downloaded Qwen2.5-0.5B-Instruct checkpoint into PyTorch and establishes the
model's REAL no-tools chat format so the C# side (QwenChatTemplate.RenderPlain) can be
A/B-diffed byte-for-byte. The no-tools branch of the checkpoint chat template always
emits the default Qwen system turn, so the rendered prompt is:

    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant

It prints and dumps:

    samples/data/qwen_plain_prompt.txt         -- the EXACT rendered prompt string
                                               (apply_chat_template with no tools)
    samples/data/qwen_plain_prompt_ids.bin     -- int32: token ids of that rendered prompt
    samples/data/qwen_plain_ids_py.bin         -- int32: every greedily generated token id
                                               (the plain answer turn)
    samples/data/qwen_plain_logits_py.bin      -- float32: logits at the final generated
                                               position (for numeric precision diff)

The prompt and max_new_tokens must match what the NivaraInference --plain benchmark uses
so both sides diff the identical sequence.
"""
import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct")
PROMPT = "What is the capital of France?"
MAX_NEW_TOKENS = 160  # generous; the model is expected to close well under this


def _identify_specials(tokenizer) -> None:
    """Print the exact ids for every special token the C# side must preserve as single tokens."""
    print("--- Special token ids ---")
    for tok in ("<|endoftext|>", "<|im_start|>", "<|im_end|>"):
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"  {tok!r:22} -> {ids}")
    print()


def _greedy(model, tokenizer, input_ids, max_new_tokens):
    """Greedily decode, returning the generated token-id list and the final-position logits."""
    gen_ids = []
    cur = torch.as_tensor([input_ids], dtype=torch.long)
    final_logits = None
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(cur)
        last = out.logits[0, -1, :]
        final_logits = last.cpu()
        nxt = int(torch.argmax(last))
        # Stop at the model's eos token (<|im_end|>).
        if nxt == tokenizer.eos_token_id:
            break
        gen_ids.append(nxt)
        # Keep the FULL running sequence: attention must span the whole prefix.
        cur = torch.cat([cur, torch.as_tensor([[nxt]], dtype=torch.long)], dim=1)
    return gen_ids, final_logits


def main():
    ap = argparse.ArgumentParser(description="Qwen2.5-0.5B plain (no-tools) prompt ground truth.")
    ap.add_argument("--prompt", default=PROMPT, help="User prompt to ask (default: capital of France).")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype="auto")
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"Vocabulary (config): {model.config.vocab_size}")
    print(f"bos_token_id={model.config.bos_token_id} eos_token_id={model.config.eos_token_id} "
          f"pad_token_id={model.config.pad_token_id}")
    _identify_specials(tokenizer)

    # ---- Render the plain single-turn prompt (no tools -> default system turn) ----
    plain_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
    )
    print("=== RENDERED PROMPT (no tools) ===")
    print(repr(plain_prompt))
    print(plain_prompt)
    print("==================================\n")

    prompt_ids = tokenizer(plain_prompt, add_special_tokens=False)["input_ids"]
    print(f"Prompt token count: {len(prompt_ids)}")
    print(f"First 8 prompt ids: {prompt_ids[:8]}  ... last 8: {prompt_ids[-8:]}")

    turn_ids, turn_logits = _greedy(model, tokenizer, prompt_ids, MAX_NEW_TOKENS)
    turn_text = tokenizer.decode(turn_ids, skip_special_tokens=False)
    print("\n=== MODEL PLAIN ANSWER TURN (raw) ===")
    print(repr(turn_text))
    print(turn_text)
    print(f"answer turn: {len(turn_ids)} tokens")

    # ---- Persist ground-truth artifacts ----
    model_dir = MODEL_DIR

    def dump(path, arr, fmt):
        with open(path, "wb") as f:
            f.write(np.asarray(arr, dtype=fmt).tobytes())
        print(f"\nWrote {path} ({len(arr)} {fmt})")

    # newline="" keeps LF so the file bytes match the in-memory string transformers
    # tokenized (no CRLF round-trip skew on Windows).
    with open(os.path.join(model_dir, "qwen_plain_prompt.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(plain_prompt)
    print(f"\nWrote {os.path.join(model_dir, 'qwen_plain_prompt.txt')} ({len(plain_prompt)} chars)")

    dump(os.path.join(model_dir, "qwen_plain_prompt_ids.bin"), prompt_ids, "int32")
    dump(os.path.join(model_dir, "qwen_plain_ids_py.bin"), turn_ids, "int32")
    dump(os.path.join(model_dir, "qwen_plain_logits_py.bin"),
         turn_logits.float().numpy(), "float32")


if __name__ == "__main__":
    main()