# TODO — Phase 2: Native function calling via Qwen2.5-0.5B-Instruct (F32)

## GitHub issues log

- [ ] #NNN — record issue numbers here as they are created (create via `gh issue create --repo khurram-uworx/Nivara` the moment a concern is found, never hold in memory)

---

## Problem

Issue #382: Phase 2 of the `--smollm` native tool-calling work. Phase B
(`tools-weather`, PR #381, branch `khurram/causal-lm-b`) proved the whole MEAI
tool loop (`<tool_call>` → `FunctionInvokingChatClient` → `AIFunction` →
`<tool_response>` → final answer) with a 0.15B community fine-tune, but that
model never produced a final answer (it looped re-issuing tool calls until the
iteration cap → blank screen). This issue pivots to **Qwen2.5-0.5B-Instruct**
(native F32, ~2 GB, mainstream documented native function calling).

The objective is as much about **improving the library** as shipping a demo:
pick the best model variant, verify what loads through the existing library, and
**implement gaps properly (with PyTorch/Torch compatibility checks)** rather than
sidestepping them.

### Pre-flight findings already established (do NOT re-verify by assumption)

Authoritative sources pulled directly from the HF repo
(`Qwen/Qwen2.5-0.5B-Instruct`):

1. **Tool format is Hermes-style, not Qwen2-era.** `tokenizer_config.json` shows
   Qwen2.5 uses `<tool_call>…</tool_call>` (added-token ids **151657 / 151658**),
   *not* the `<|tool_call_start|>` markers used by Qwen2. The issue's premise
   ("not the Hermes `<tool_call>`") is **incorrect for this exact checkpoint**.
   Tool results go in `<tool_response>…</tool_response>` **inside a
   `<|im_start|>user…<|im_end|>` turn** (the `chat_template` renders `role=="tool"`
   messages as a `user` turn).
2. **Special-token ids:** `eos_token` = `<|im_end|>` (**151645**), `pad_token` /
   `bos_token` = `<|endoftext|>` (**151643**). `bos_token_id` in config = 151643.
   Config `eos_token_id` = 151645. **V2 (verified from `generation_config.json`):
   eos is an array `[151645, 151643]`** — generation must stop on EITHER
   `<|im_end|>` (151645) or `<|endoftext|>` (151643). Default sampling config
   (`temperature 0.7, top_p 0.8, repetition_penalty 1.1`) is for the demo default;
   ground truth uses greedy for determinism.
3. **Vocab-size split (critical risk).** `config.json`: `vocab_size: 151936`,
   `tie_word_embeddings: true`. `tokenizer.json`/`vocab.json` map only vocab ids
   **0–151642** (151,643 entries). Special tokens (151643 `<|endoftext|>`,
   151644 `<|im_start|>`, 151645 `<|im_end|>`, 151657/151658 tool tags) exist only
   in `added_tokens` / `added_tokens_decoder`, **not** in the vocab map. So the
   embed/head matrix is 151,936 rows but the tokenizer only names 151,643 —
   a ~293-row tail that exists in the weight table with no text mapping. Must be
   handled deliberately (generation indices, argmax range, BOS/EOS ids).
4. **Tensor layout matches `LlamaLoader` exactly**: `model.embed_tokens`,
   `model.layers.{i}.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
   `model.norm` — same names `StateDictLoader` already loads. `tie_word_embeddings`
   true → single embedding reused as head (matches `LlamaForCausalLM`), no `lm_head`.
5. **Config values (Qwen2.5-0.5B-Instruct):** hidden 896, layers 24, heads 14,
   kv_heads 2, intermediate 4864, rms_norm_eps 1e-6, max_position 32768,
   rope_theta 1e6, tie_word_embeddings true. `use_sliding_window: false`.
   Architecture `Qwen2ForCausalLM`. Weight count ~494M BF16 params.
6. **Tokenizer**: byte-level BPE (`vocab.json` + `merges.txt`) — same algorithm
   family as GPT-2 / SmolLM → `Gpt2BpeTokenizer` likely loads it, but vocab size
   151,643 vs SmolLM 49,152, and special-token handling has to extend beyond the
   current SmolLM `SpecialTokens` list.

### Reusable from the shelved Phase B (branch `khurram/causal-lm-b`) — ideas, not copies

- `LlamaChatClientBase<T>` generation core (KV-cached/full forward, greedy / top-p
  sampling, prompt encoding) — model-agnostic and sound, but **never merged** to
  main. This branch's main-line only has the plain `SmolLMChatClient`.
- `FunctionInvokingChatClient`-based tool loop + `MaximumIterationsPerRequest` cap.
- Tool-name case normalization (`CanonicalToolName`) — the framework matches tool
  names case-sensitively; models can emit `getWeather` vs tool `GetWeather`.
- Lenient-parser lessons for non-strict JSON (though Qwen is expected to be strict;
  keep a tolerant fallback as insurance).

### What the main branch currently has (baseline)

- `samples/NivaraChat/SmolLM/SmolLMChatClient.cs` — plain chat (no tools, no base).
- `SmollmChatTemplate.cs` — Hermes render, **no** tool-call/tool-response surface.
- `samples/Nivara.Samples/LlamaForCausalLM.cs` + `LlamaLoader` + `LlamaConfig` —
  Llama/SmolLM structural load; **hoped** to be a drop-in for Qwen (verify).
- `samples/Nivara.Samples/Gpt2BpeTokenizer.cs` — byte-level BPE.
- Tests: `tests/Nivara.Tests/AutoDiff/Gpt2BpeTokenizerTests.cs` (SmolLM tokenizer
  parity, skips when files absent).

---

## Proposed changes (phased; gaps plan kept separate from demo wiring)

4. **Tensor layout matches `LlamaLoader` exactly**: `model.embed_tokens`,
   `model.layers.{i}.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`,
   `model.norm` — same names `StateDictLoader` already loads. `tie_word_embeddings`
   true → single embedding reused as head (matches `LlamaForCausalLM`), no `lm_head`.
   **V2 (verified from safetensors): Qwen2.5-0.5B is the bias variant —**
   `q_proj.bias=(896,)`, `k_proj.bias=(128,)`, `v_proj.bias=(128,)` exist on every
   layer (o_proj has none); `config.json` omits `use_qkv_bias` but transformers
   auto-detects and loads with bias. C# `StateDictLoader.LoadLinear` already treats
   `bias` as optional, so this loads — the Linears must be created with bias=true.</think>

<｜DSML｜tool_calls>
<｜DSML｜invoke name="edit">
<｜DSML｜parameter name="newString" string="true">### Phase 1 — Ground truth & confidence (Python, NO C# code) — do this FIRST

Establishes "the model works" before any implementation, exactly the de-risking
that Phase B skipped. **Gated on human confirmation before Phase 2.**

1. **Download the model**
   - `hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir samples/data/qwen2.5-0.5b-instruct`
     (weights ~1 GB; HF_TOKEN is set; `hf` 1.2.3 verified).
   - Add `samples/data/qwen2.5-0.5b-instruct/` to `.gitignore` (follow the
     existing `samples/data/smollm-135m/` convention).
   - **Gap fix:** `samples/data/smollm-fn-135m/` is on disk but NOT gitignored on
     main (the Phase B research doc says "ignored" but the ignore rule never
     landed). Add it to `.gitignore` so the 304 MB weights can't be committed.
2. **Author ground-truth Python script**
   `samples/NivaraInference/Python/qwen_tool_reference.py` (mirror the existing
   `smollm_generate_reference.py` pattern):
   - Print the exact `apply_chat_template(..., tools=..., add_generation_prompt=True)`
     **string** and token ids (the model's real tool format), and the ids of
     `<tool_call>` / `<tool_response>` / `<|im_start|>` / `<|im_end|>` / `<|endoftext|>`.
   - Run a deterministic `GetWeather` tool loop (greedy, tuned `max_new_tokens`)
     on a fixed prompt (e.g. `"What's the weather in Paris?"`).
   - Dump ground-truth artifacts: rendered prompt string, prompt token ids, full
     greedy generation (assistant tool-call turn → tool response → final NL answer)
     so C# can A/B-diff byte-for-byte.
3. **Confidence gate (human reviews output):** the model must (a) emit
   `<tool_call>{"name":"...","arguments":{...}}</tool_call>` **and**
   `<tool_response>`, and (b) **terminate with a clean final NL answer** — not
   loop, not blank. If it can't close the loop, STOP, record findings, and
   escalate (as with the shelved Biggie run) before any implementation.

### Phase 1 results (gate PASSED — model closes the loop)

Run `python samples/NivaraInference/Python/qwen_tool_reference.py` on the
downloaded checkpoint produced (greedy, `max_new_tokens=160`):

- **Tool-call turn = 19 tokens:** `<tool_call>\n{"name": "getWeather", "arguments":
  {"city": "Paris"}}\n</tool_call>` — exact canonical JSON inside the tags.
- **Tool result rendering confirmed:** the final prompt shows the result inside
  `<tool_response>\nPartly cloudy, 18°C…</tool_response>` within a
  `<|im_start|>user…<|im_end|>` turn (format finding #1 confirmed by observation).
- **Final-answer turn = 23 tokens:** "The weather in Paris is partly cloudy with a
  high of 18°C and a light breeze from the northwest." — clean final NL answer,
  grounded in the tool result.
- End-to-end loop = 42 generated tokens, well under the cap-3 budget.

**Findings recorded (update the pre-flight facts, do not re-verify):**

- **Checkpoint is BF16, not "native F32".** `model.safetensors` is 988 MB (494M
  params × 2B), `torch_dtype: bfloat16`; the issue's "native F32 ≈ 2 GB" premise
  is wrong for this upstream repo. Not a blocker: C# runs float32 inference and
  `SafeTensorsLoader.Read<T>` already has a BF16→F32 arm (per AGENTS).
- **Tokenizer enumerates 151,665 ids** (`len(tokenizer.get_vocab())`), not the
  151,643 base vocab — added tokens are in the tokenizer map too.
- **`<tool_response>` / `</tool_response>` are NOT added/special tokens** — they
  tokenize as ordinary bytes: `[27, 14172, 9655, 29]` (`/API<`). Only
  `<tool_call>` (151657) `</tool_call>` (151658) plus `151643/151644/151645` are
  added tokens. The C# renderer writes them as plain text; only the five specials
  need added-token handling.
- **`q/k/v_proj` biases confirmed** on the loaded model (see V2 finding above):
  `q_proj.bias=(896,)`, `k_proj.bias=(128,)`, `v_proj.bias=(128,)`, o_proj none.
- Ground-truth artifacts (in the gitignored model dir, same convention as the
  SmolLM fixtures): `qwen_tool_prompt.txt` (869 chars), `qwen_tool_prompt_ids.bin`
  (206 ids), `qwen_tool_ids_py.bin` (42 ids), `qwen_tool_logits_py.bin`
  (151,936 float32).

### Phase 2 — Loader / model / tokenizer verification (library-gap checks, Torch parity)

1. **Loader drop-in check:** verify `LlamaLoader` + `LlamaForCausalLM` load Qwen's
   safetensors with zero code changes (expected — same 10 tensor names,
   `tie_word_embeddings`). If it fails, track as a library gap and fix properly.
2. **Vocab-size / special-token handling** (the audited risk item):
   - Decide final handling: `VocabSize` from config (151,936) for the embed/head
     table; generation/argmax constrained to the tokenizer's real 151,643 mapped
     ids; BOS/EOS by id (151643 / 151645); document the 293-row tail.
   - Unit-test this explicitly.
3. **Tokenizer parity:** add a `Gpt2BpeTokenizerTests`-style test against the Qwen
   vocab/merges (skip when files absent), pinning ids against the HF `AutoTokenizer`,
   and extend special-token handling beyond the SmolLM list (add `<tool_call>`,
   `</tool_call>`, `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`).
4. **Torch-compatibility / correctness check:** Python reference dumps weights + a
   fixed-prompt forward's logits + token-id sequence; compare C# `model.Forward` /
   `ForwardCached` to numeric tolerance (mirror `smollm_generate_reference` +
   `Gpt2BpeTokenizerTests`). This validates every building block end-to-end before
   wiring.

### Phase 3 — Library gaps / improvements (separate section; Torch-checked)

1. **Shared generation core** (corrected, not copied from branch):
   - `QwenChatClient<T>` + `QwenChatTemplate`: renderer + a **Qwen-correct**
     tool-call parser for `<tool_call>`/`<tool_response>` (tool results in a
     `user`-role turn).
   - Fix the three silent-failure items from Phase B:
     - Correct `<tool_call>`/`<tool_response>` rendering/parsing for Qwen's real
       format.
     - Populate `FunctionCallContent.Arguments` as a correct dict (the Phase B
       `BuildToolCallContents` JSON path was fragile).
     - BOS/EOS + vocab-size handling from Phase 2.
   - Reuse Phase B *ideas*: `CanonicalToolName` name normalization; capped
     `FunctionInvokingChatClient` loop; tolerant parser fallback.
2. **Wiring:** a distinct `--qwen` entry point with a `tools-weather` sub-mode
   (deterministic `GetWeather` `AIFunction`), `MaximumIterationsPerRequest = 3`
   (Qwen should close in 2: tool call → final answer), and a verification that the
   loop **closes with a clean final answer**. Keep existing `--smollm chat|plain`
   untouched. *(Decision confirmed with human: `--qwen` gateway, cap 3.)*
3. **Tests:** tool-format render/parse round-trip, name normalization, loop
   termination (not blank), tokenizer parity on Qwen files. Follow the
   "skip when model files absent" pattern so CI stays green.

### Phase 4 — Documentation

- `docs/research/QWEN-TOOL-CALLING.md` — ground-truth findings (real format, the
  vocab-size subtlety, what's reusable from Phase B, what was fixed, why).
- Update `docs/TODO.md` and `docs/BFLOAT16.md`/`README` as needed.

---

## Verification

- Phase 1: human review of Python ground-truth output (format + loop termination).
- Phase 2: Torch-parity numeric diff (weights/logits/token ids) within tolerance;
  `Gpt2BpeTokenizerTests`-style parity tests on Qwen files.
- Phase 3: unit tests (render/parse round-trip, normalization, loop termination);
  build via `dotnet build Nivara.slnx`.
- `dotnet test` only after explicit human confirmation (per AGENTS.md).
- Acceptance (from issue #382): `--qwen tools-weather --text "What's the weather in Paris?"`
  returns a clean final NL answer derived from `GetWeather`, not a ramble/blank;
  loop terminates within `MaximumIterationsPerRequest`; `--smollm chat|plain`
  (SmolLM-135M) untouched and working.

---

## Planned commits (one logical unit each)

1. `chore: ignore downloaded model weight dirs (qwen2.5-0.5b-instruct, smollm-fn-135m)`
2. `docs: plan Phase 2 Qwen tool-calling in TODO.md` (this file — first commit of
   the plan per iterative-work)
3. Phase 1: `feat(samples): Qwen2.5 tool-calling ground-truth reference script`
   (Python only) + gitignore + download.
4. Phase 2: loader/vocab/tokenizer verification commits (each with tests).
5. Phase 3: `QwenChatTemplate` / `QwenChatClient` / `--qwen tools-weather` wiring,
   one logical change per commit.
6. Review + remove `docs/TODO.md` (iterative-work G2) → offer push + PR (#382).

---

## Blast radius

- New files under `samples/NivaraChat/` (Qwen client/template/tools) and
  `samples/NivaraInference/Python/` (reference script).
- New tests under `tests/Nivara.Tests/`.
- `.gitignore` additions only — no existing tracked files changed except
  `docs/TODO.md` and possibly `docs/research/` + `samples/NivaraChat/Program.cs`
  (add `--qwen` dispatch) + README/help text.
- **No public library API changes** in `src/Nivara` expected unless a loader
  gap is found (then: tracked as a separate issue, Torch-checked, additive).
- Downstream: `--smollm chat|plain` must remain untouched; NivaraChat demo entry
  point (`Program.cs`) gains a new branch only.

---

## Execution guardrails (from iterative-work & AGENTS.md)

- Ask before running `dotnet test` / long verification.
- Never push; commits local only; selective staging; one logical change per commit.
- Create GitHub issues at discovery time; record in the GitHub issues log above.
- G2 review (branch as whole + against this plan) before removing `docs/TODO.md`.
