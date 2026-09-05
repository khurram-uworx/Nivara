"""Ground-truth fixture: Qwen2.5-0.5B-Instruct native function calling (tool loop).

Run-once reference for the Nivara --qwen tool-calling work (issue #382, Phase 1).
Loads the locally-downloaded Qwen2.5-0.5B-Instruct checkpoint into PyTorch and
establishes the model's REAL chat/tool format so the C# side can be A/B-diffed
byte-for-byte (no assumptions from the issue text or older Qwen2 docs).

It prints and dumps:

    samples/data/qwen_tool_prompt.txt         -- the EXACT rendered prompt string
                                               (apply_chat_template with tools, generation prompt)
    samples/data/qwen_tool_prompt_ids.bin     -- int32: token ids of that rendered prompt
    samples/data/qwen_tool_final_prompt.txt   -- the EXACT rendered prompt for the FINAL turn,
                                               i.e. after the tool result is fed back
    samples/data/qwen_tool_final_prompt_ids.bin -- int32: token ids of that final prompt
    samples/data/qwen_tool_ids_py.bin         -- int32: every greedily generated token id across
                                               the FULL tool loop, concatenated (assistant
                                               <tool_call> turn, then the assistant final-answer
                                               turn after the tool result is fed back)
    samples/data/qwen_tool_logits_py.bin      -- float32: logits at the FINAL generated position
                                               of the final-answer turn (for numeric precision diff)

The tool loop mirrors what the Nivara/FunctionInvokingChatClient path will do:
  1. user:  What's the weather in Paris?
  2. model: <tool_call>{"name":"getWeather","arguments":{...}}</tool_call>
  3. tool:  <tool_response>Partly cloudy, 18°C...</tool_response>   (fed back as a "tool" message)
  4. model: <|im_start|>assistant ... "The weather in Paris is Partly cloudy, 18°C." <|im_end|>

The prompt and max_new_tokens must match what the C# sample uses so both sides
diff the identical sequence.
"""
import os
import sys
import struct
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from hf_loader import MODELS_DIR

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct")
PROMPT = "What's the weather in Paris?"
MAX_NEW_TOKENS = 160  # generous; the model is expected to close well under this per turn

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "getWeather",
        "description": "Gets the current weather for a city. Returns a short description like 'Sunny, 22\u00b0C'.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Paris' or 'New York'",
                }
            },
            "required": ["city"],
        },
    },
}


def _identify_specials(tokenizer) -> None:
    """Print the exact ids for every special token the C# side must preserve as single tokens."""
    print("--- Special token ids ---")
    for tok in ("<|endoftext|>", "<|im_start|>", "<|im_end|>",
                "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"  {tok!r:22} -> {ids}")
    print()


def _get_weather(city: str) -> str:
    c = city.strip().lower()
    table = {
        "paris": "Partly cloudy, 18\u00b0C. Light breeze from the northwest.",
        "london": "Overcast with light rain, 14\u00b0C.",
        "new york": "Sunny, 25\u00b0C. Clear skies expected.",
        "tokyo": "Humid and warm, 28\u00b0C. Chance of afternoon showers.",
        "berlin": "Cool and breezy, 12\u00b0C. Mostly cloudy.",
    }
    return table.get(c, f"Clear skies, 20\u00b0C in {city}. Pleasant weather.")


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
    ap = argparse.ArgumentParser(description="Qwen2.5-0.5B tool-loop ground truth.")
    ap.add_argument("--prompt", default=PROMPT, help="User prompt to ask.")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype="auto")
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"Vocabulary (config): {model.config.vocab_size}")
    print(f"Token ids enumerable by tokenizer: {len(tokenizer.get_vocab())}")
    print(f"bos_token_id={model.config.bos_token_id} eos_token_id={model.config.eos_token_id} "
          f"pad_token_id={model.config.pad_token_id}")
    qb = model.model.layers[0].self_attn.q_proj.bias
    vb = model.model.layers[0].self_attn.v_proj.bias
    kb = model.model.layers[0].self_attn.k_proj.bias
    print(f"layer0 q_proj.bias={None if qb is None else tuple(qb.shape)} "
          f"k_proj.bias={None if kb is None else tuple(kb.shape)} "
          f"v_proj.bias={None if vb is None else tuple(vb.shape)}")
    _identify_specials(tokenizer)

    # ---- Render the FIRST assistant turn (user -> model may issue a tool call) ----
    tools_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tools=[WEATHER_TOOL],
        add_generation_prompt=True,
        tokenize=False,
    )
    print("=== RENDERED PROMPT WITH TOOLS (tools_prompt) ===")
    print(repr(tools_prompt))
    print(tools_prompt)
    print("==================================================\n")

    prompt_ids = tokenizer(tools_prompt, add_special_tokens=False)["input_ids"]
    print(f"Prompt token count: {len(prompt_ids)}")
    print(f"First 8 prompt ids: {prompt_ids[:8]}  ... last 8: {prompt_ids[-8:]}")

    tool_turn_ids, tool_logits = _greedy(model, tokenizer, prompt_ids, MAX_NEW_TOKENS)
    tool_turn_text = tokenizer.decode(tool_turn_ids, skip_special_tokens=False)
    print("\n=== MODEL TOOL-CALL TURN (raw) ===")
    print(repr(tool_turn_text))
    print(tool_turn_text)
    print(f"tool turn: {len(tool_turn_ids)} tokens")

    # ---- Feed the tool result back as a role="tool" message, per Qwen chat_template ----
    # Parse the model's tool call to know which city to query.
    import re
    m = re.search(r'"name"\s*:\s*"(\w+)"', tool_turn_text)
    tool_name = m.group(1) if m else "(none)"
    city_m = re.search(r'"city"\s*:\s*"([^"]*)"', tool_turn_text)
    city = city_m.group(1) if city_m else "Paris"
    print(f"\nRequested tool: {tool_name}, city={city!r}")

    tool_result = _get_weather(city)
    final_messages = [
        {"role": "user", "content": args.prompt},
        {"role": "assistant", "content": tool_turn_text},
        {"role": "tool", "content": tool_result, "name": tool_name},
    ]
    final_prompt = tokenizer.apply_chat_template(
        final_messages, tools=[WEATHER_TOOL], add_generation_prompt=True, tokenize=False)
    print("\n=== RENDERED PROMPT WITH TOOL RESULT (final_prompt) ===")
    print(repr(final_prompt))

    final_ids = tokenizer(final_prompt, add_special_tokens=False)["input_ids"]
    final_turn_ids, final_logits = _greedy(model, tokenizer, final_ids, MAX_NEW_TOKENS)
    final_turn_text = tokenizer.decode(final_turn_ids, skip_special_tokens=False)
    print("\n=== MODEL FINAL ANSWER TURN (raw) ===")
    print(repr(final_turn_text))
    print(final_turn_text)
    print(f"final turn: {len(final_turn_ids)} tokens")

    # ---- Persist ground-truth artifacts ----
    model_dir = MODEL_DIR
    def dump(path, arr, fmt):
        with open(path, "wb") as f:
            f.write(np.asarray(arr, dtype=fmt).tobytes())
        print(f"\nWrote {path} ({len(arr)} {fmt})")

    # The exact rendered prompt, for C# A/B diffing. newline="" keeps LF so the file bytes match
    # the in-memory string transformers tokenized (no CRLF round-trip skew on Windows).
    with open(os.path.join(model_dir, "qwen_tool_prompt.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(tools_prompt)
    print(f"\nWrote {os.path.join(model_dir, 'qwen_tool_prompt.txt')} ({len(tools_prompt)} chars)")

    dump(os.path.join(model_dir, "qwen_tool_prompt_ids.bin"), prompt_ids, "int32")

    # The exact rendered final prompt (with the fed-back tool result), also for C# A/B diffing.
    with open(os.path.join(model_dir, "qwen_tool_final_prompt.txt"), "w", encoding="utf-8", newline="") as f:
        f.write(final_prompt)
    print(f"\nWrote {os.path.join(model_dir, 'qwen_tool_final_prompt.txt')} ({len(final_prompt)} chars)")

    dump(os.path.join(model_dir, "qwen_tool_final_prompt_ids.bin"), final_ids, "int32")

    dump(os.path.join(model_dir, "qwen_tool_ids_py.bin"),
         tool_turn_ids + final_turn_ids, "int32")
    dump(os.path.join(model_dir, "qwen_tool_logits_py.bin"),
         final_logits.float().numpy(), "float32")


if __name__ == "__main__":
    main()
