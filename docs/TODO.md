# TODO: NivaraChat — SmolLM causal LM as IChatClient (Stage A)

## Problem

`samples/NivaraChat` is our showcase for how Nivara participates in the .NET first-party
AI ecosystem (`Microsoft.Extensions.AI` `IChatClient` + Microsoft Agent Framework
`ChatClientAgent`). We recently shipped real SmolLM-135M-Instruct causal-LM support in
`Nivara.Samples` (`LlamaForCausalLM<T>`, `LlamaLoader`, `Gpt2BpeTokenizer`,
`SafeTensorsLoader`, greedy generation in `samples/NivaraInference`).

The gap: the only LLM-class `IChatClient` demo in NivaraChat is the tiny
*trained-on-TinyShakespeare* `BatchedChatClient`. There is no way to serve the **real
pretrained** SmolLM through the standard `IChatClient` pipeline with a proper
chat-template conversation, and no foundation for the later tool-calling stages.

This branch implements **only Stage A** of the two-demo plan (see
`C:\Users\khurram\.opencode\plan\smollm-tool-calling-nivarachat.md`): present the real
SmolLM as an `IChatClient` that renders a Hermes/ChatML conversation and
token-streams plain-text replies. No tools yet (Stages B/C are separate branches).

## Proposed changes (all additive in `samples/`)

1. **`Gpt2BpeTokenizer` (Nivara.Samples)** — add a public `TokenId(string)` accessor so
   the chat template can encode special tokens (`<|im_start|>`, `<|im_end|>`,
   `<|endoftext|>`) as single token ids (byte-level BPE would otherwise split them).

2. **`samples/NivaraChat/SmolLM/SmollmChatTemplate.cs`** — helper encapsulating Hermes
   ChatML rendering for a plain-chat conversation (no tools):
   - `Render(messages [, addGenerationPrompt])` → the prompt string using
     `<|im_start|>system\n...<|im_end|>`, `<|im_start|>user\n...<|im_end|>`,
     `<|im_start|>assistant\n`.
   - Appends `<|im_start|>assistant\n` when `addGenerationPrompt` is true.

3. **`samples/NivaraChat/SmolLM/SmolLMChatClient.cs`** — an `IChatClient` backed by
   `LlamaForCausalLM<T>` + `Gpt2BpeTokenizer` + `LlamaConfig`:
   - `GetResponseAsync`: render the conversation → greedy-generate → return the
     plain-text assistant `ChatMessage`.
   - `GetStreamingResponseAsync`: greedy-generate and yield one `ChatResponseUpdate`
     (decoded token) at a time, stopping at EOS (`<|im_end|>` id) or `maxNewTokens`.
   - Greedy argmax (mirrors `NivaraInference.Program.RunSmolLMGeneration` /
     `ArgMaxLastRow`), tokenizer-special-token aware, re-entrant (eval mode, does NOT
     own the model; `Dispose` no-op).
   - `Metadata` = "Nivara SmolLM-135M-Instruct"; `GetService` returns null (the
     `ChatClientAgent`/`FunctionInvokingChatClient` wrapping stays for later stages).
   - Generic `<T : struct, IFloatingPointIeee754<T>>`; default precision **F32**
     (`--precision f32|bf16`), mirroring `NivaraInference`.

4. **`samples/NivaraChat/SmolLM/SmollmMode.cs`** — the `--smollm` CLI/menu mode
   (analogous to `TransformerMode`):
   - Loads `config.json`, `model.safetensors`, `vocab.json`, `merges.txt` from
     `samples/data/smollm-135m` (`--model-dir` override) via `SafeTensorsLoader.Read<T>`
     + `LlamaLoader.Load<T,T>` + `Gpt2BpeTokenizer`.
   - Sub-modes for Stage A:
     - `--smollm chat [<text>]` — send a prompt and stream the (possibly multi-turn)
       reply; interactive REPL when no `--text`.
     - `--smollm plain <text>` — single-shot plain text reply (no REPL).
   - Flags: `--model-dir`, `--precision f32|bf16`, `--max-new-tokens`, `--text`.

5. **`samples/NivaraChat/Program.cs`** — wire `case "--smollm": SmollmMode.Run(...)`
   plus a main-menu entry (mirrors `--tinyshakespeare` → `TransformerMode`).

6. **`samples/NivaraChat/README.md`** — document the `--smollm` mode(s) and quick start.

## Verification steps (ask before long verification per AGENTS.md)

- `dotnet build Nivara.slnx` (sample-only changes; no core changes).
- `dotnet run --project samples/NivaraChat -- --smollm plain "The capital of France is"`:
  - ChatML prompt is rendered and passed to the model; reply is greedy-generated and
    **token-streamed**; EOS `<|im_end|>` stops generation; F32 yields no NaN.
- `--smollm chat` interactive REPL: multi-turn conversation re-entrant (each call builds
  its own tensors); consistent across repeated turns.
- Sanity: decoded output is coherent natural language (not garbage); repeated runs are
  deterministic (greedy, seed-free).

## Blast radius

- **Files changed (all additive in `samples/`):**
  - `samples/Nivara.Samples/Gpt2BpeTokenizer.cs` — one additive public method
    (`TokenId`); no behavior change to existing `Encode`/`Decode`. Downstream callers:
    `NivaraInference.Program` and `NivaraChat` (both unchanged in behavior).
  - `samples/NivaraChat/SmolLM/*` (new files), `samples/NivaraChat/Program.cs` (additive
    switch case + menu entry), `samples/NivaraChat/README.md`.
- **No changes** to core `src/Nivara` or `src/Nivara.Extensions`.
- **Tests:** no dedicated unit-test project for NivaraChat; verification is the run
  commands above (AGENTS.md: long verification requires human confirmation). Build is
  the automated check.
- **Downstream deps of changed files:** only `NivaraChat` and `NivaraInference` consume
  `Nivara.Samples`; the additive method cannot break them.

## Planned commits

1. `feat: add Gpt2BpeTokenizer.TokenId accessor for special-token encoding` — additive
   public method in `Nivara.Samples`.
2. `feat: add SmolLM Hermes ChatML rendering helper` — `SmollmChatTemplate.cs`.
3. `feat: add SmolLMChatClient IChatClient (greedy, token-streamed)` — `SmolLMChatClient.cs`.
4. `feat: add --smollm chat/plain mode to NivaraChat` — `SmollmMode.cs` + `Program.cs` wiring.
5. `docs: document SmolLM --smollm mode in NivaraChat README`.

(Commits 1–3 may be consolidated during implementation if they land as one logical unit;
the mode + wiring + README are the user-facing increment.)

## GitHub issues log

- [ ] #375 — SmolLM IChatClient follow-ups: KV cache + temperature/top-p sampling (created while concluding Stage A; intentional out-of-scope items tracked so they aren't lost after TODO.md removal)
