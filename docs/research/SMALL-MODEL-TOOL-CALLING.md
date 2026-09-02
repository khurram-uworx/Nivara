# Research: Small-Model Native Tool Calling — `archit11/small-function-calling` Evaluation (Shelved)

**Status:** Experiment concluded — shelved (not shipping function calling through small-LLM clients for now)
**Date:** 2026-09-02
**Related plan:** `rethink-smollm-tool-model.md` (Option B decision)

## Purpose

This note records why the **Option B** experiment — wiring the `archit11/small-function-calling`
(``Biggie-SmoLlm-0.15B``) fine-tune into the `--smollm tools-weather` demo — was started, what
was learned (including real pipeline bugs it exposed), and the honest engineering conclusion.
It exists so a future iteration does not need to re-run the ~8 expensive CPU inference sessions
that produced this finding. The code and model files are intentionally **left in place**; nothing
was reverted.

## Background (why this experiment)

The repo's stock **SmolLM-135M-Instruct** was never trained for function calling, and
**SmolLM2-135M/360M** deliberately have function calling stripped (SmolLM2 paper). Only
**SmolLM2-1.7B-Instruct** has it — impractical in native F32 (~6.8 GB, ~12× the FLOPs/token of
135M → multi-minute CPU generations). So the decision was made to try a small community
fine-tune built on Hermes function-calling data as a cheap drop-in validation before any larger
integration (see the rethink plan, Options A–E).

## What was built (kept in the tree)

- `samples/NivaraChat/SmolLM/LlamaChatClientBase.cs` — shared autoregressive-generation core
  extracted out of the old `SmolLMChatClient` (prompt encoding, KV-cached/full forward, greedy or
  top-p sampling, Hermes tool surface with `virtual` `RenderPrompt`/`TryParseToolCall`).
- `samples/NivaraChat/SmolLM/SmolLMChatClient.cs` — now lean (constructor + `ModelLabel` only).
- `samples/NivaraChat/SmolLM/BiggieChatClient.cs` — the Biggie client: a Hermes-1 style tool system
  prompt (`NousResearch/hermes-function-calling-v1` format the fine-tune was trained on) and a
  **lenient** tool-call parser to tolerate the single-quoted / Python-dict-style broken JSON the
  model emits.
- `samples/NivaraChat/SmolLM/SmollmMode.cs` — `tools-weather` sub-mode wired to pick the Biggie
  client when `--model-dir` is `samples/data/smollm-fn-135m`; tool-loop iteration cap.
- `samples/data/smollm-fn-135m/` — the fine-tune weights + base tokenizer (304 MB `model.safetensors`).

## What actually worked (validated end-to-end)

The pipeline mechanics — all of which are model-agnostic and reusable:

1. **Hermes-1 prompt** reliably elicits `<tool_call>` from the 0.15B model (the SmolLM official
   directive-style prompt did not).
2. **Name normalization** — `FunctionInvokingChatClient` matches tool names case-sensitively
   (`StringComparison.Ordinal`); the model emits `getWeather` (lowercase), so the base now maps
   parsed names to the canonical `AIFunction` casing before emitting `FunctionCallContent`.
3. **Tool invocation + feedback** — `GetWeather` is invoked and its result is fed back as
   `<tool_response>`, then the model is re-prompted.

### Bugs the experiment exposed and fixed

The lenient parser's regexes were **silently failing** — `TryParseToolCall` returned `false` even
when the model emitted a plausibly-parseable `<tool_call>`, so the raw markup was returned as plain
text (no tool loop at all):

- `NamePattern` (`name\s*[:=]...`) could not match `'name': 'getWeather'` — it did not account for
  the quotes around the key, so no name matched.
- `ArgPattern` had a quote-consumption/overlap bug (a trailing optional quote swallowed the next
  key's opening quote) so `city` was never captured, and the value class did not stop at commas
  (captured `16,`).

Both were fixed and verified against the exact captured model output with an isolated harness
(no model load needed). After the fix the loop genuinely iterates (generation → tool → generation).

## The honest finding (why it's shelved)

Even with a working parser and tool loop, **the 0.15B fine-tune never terminates the loop**: after
receiving the weather result it re-issues `<tool_call>` on every subsequent turn instead of producing
a final natural-language answer. With the loop capped (`MaximumIterationsPerRequest = 3`), the final
returned response is the leftover tool call with **no text** → a blank screen. Across ~8 runs under
varying prompts, tokens, and sampling, it either rambled, or echoed the schema, or looped re-calling
the tool — it could not reliably close with "The weather in Paris is …".

**Conclusion:** the pipeline works; the **model is the limitation**. This is precisely the risk the
rethink plan flagged for a 0.15B community fine-tune, and the reason it named **Option C
(`Qwen2.5-0.5B-Instruct`, mainstream documented native function calling)** as the production-grade
fallback. Per the user's decision, the experiment is **ended** and function calling is **not shipped**
through these small-LLM clients **for now**.

## What is reusable if this is revisited

- The `LlamaChatClientBase<T>` generation core and the `FunctionInvokingChatClient`-based tool loop
  are model-agnostic and in good shape.
- The lenient-parser bug fixes and the case-insensitive name normalization are general and would
  benefit any small-model tool client (including a future Qwen path, though Qwen has its own native
  `<tool_call>` format).
- The `Hermes1` prompt override is specific to Biggie; a Qwen integration would instead add Qwen's
  own renderer/parser plus a tokenizer port (see rethink plan Option C).

## Next step if the feature is reopened

Pivot to **Option C: `Qwen2.5-0.5B-Instruct`** (native-F32 ≈ 2 GB, Llama-loader-compatible config),
porting/validating the Qwen tokenizer and adding a Qwen tool-format renderer + parser. Reuse the
`LlamaChatClientBase` core and tool-loop plumbing. No decision to pursue it has been made.
