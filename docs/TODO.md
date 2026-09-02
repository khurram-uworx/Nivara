# TODO: SmolLM native tool-calling (Phase B) — `GetWeather`

## Status: Plan executed → experiment ended (2026-09-02)

This plan was implemented, but the `tools-weather` native tool-calling path is **not shipped**.
The committed/demo code was wired to run the actual model, and the plan's `chat`/`plain`
sub-modes work exactly as designed; however the **model** made the tool loop non-functional:

- The stock SmolLM-135M was never trained for function calling.
- To validate the pipeline before a larger integration, the `tools-weather` mode was pointed at a
  community Hermes fine-tune (`archit11/small-function-calling`, `Biggie-SmoLlm-0.15B`). The
  pipeline mechanics (render → parse → `getWeather`→`GetWeather` name normalization → invoke →
  feed back → re-prompt) all work, but the 0.15B model **never terminates the loop** — it
  re-issues `<tool_call>` on every turn and produces no final answer (iteration-capped → blank).

**Decision (user, 2026-09-02):** end the experiment; do not ship function calling through these
small-LLM clients for now. Code and model files kept as-is. Detailed findings and the option to
pivot to `Qwen2.5-0.5B-Instruct` (Option C) are recorded in
[`docs/research/SMALL-MODEL-TOOL-CALLING.md`](research/SMALL-MODEL-TOOL-CALLING.md).

## Problem

Phase A (`khurram/causal-lm-a`, on `main`) landed the SmolLM-135M-Instruct causal LM as an
`IChatClient` (`SmolLMChatClient<T>`, `SmollmChatTemplate`, `SmollmMode`) that only does plain
chat. We want to prove the full MEAI **native tool-calling loop** end-to-end with a single,
predictable tool — `GetWeather` — without pulling in Nivara models or Ollama. This is the
incremental step before adding Nivara model tools and an agent surface (Phase C).

Goal pipeline to prove:

```
model emits <tool_call>[{...}]</tool_call>
  → FunctionInvokingChatClient invokes the AIFunction
  → FunctionResultContent rendered as <tool_response> (user turn)
  → model produces a final natural-language answer
```

Closing the loop requires extending the inner `IChatClient` so the framework can do its job.
Per Microsoft Learn, `FunctionInvokingChatClient` (auto-wrapped by `ChatClientAgent`) handles
tool matching, invocation, result creation, history append, and re-prompting. Our client is
responsible for: (1) rendering `<tools>{json}</tools>` into the system prompt when
`ChatOptions.Tools` is present, (2) rendering `FunctionResultContent` as `<tool_response>`,
(3) parsing a generated `<tool_call>` block and emitting `FunctionCallContent`, and (4) returning
plain text when no tool call is present.

## Design decisions (confirmed with human)

- **Surface tool-call responses by buffering** — display only the final natural-language answer,
  hiding raw `<tool_call>` markup. Achieved by driving the `tools-weather` mode through
  `GetResponseAsync` (non-streaming) on the `FunctionInvokingChatClient` wrapper, which runs the
  whole loop internally and returns the final `ChatResponse`. This avoids noisy streaming of
  `<tool_call>` markup and is the natural fit for the framework's tool loop.
- **Default `--max-new-tokens` for `tools-weather` is 256** (tool-call + final answer need
  headroom). Plain `chat`/`plain` keep the 64 default.

## Proposed changes

### 1. `SmollmChatTemplate` — tool pipeline rendering + parsing
`E:\khurram-uworx\Nivara\samples\NivaraChat\SmolLM\SmollmChatTemplate.cs`

- New `Render(IEnumerable<ChatMessage>, bool addGenerationPrompt = true, IReadOnlyList<AITool>? tools = null)`:
  - When `tools` non-empty, emit the Hermes tool-calling system prompt:
    ```
    <|im_start|>system
    You are an expert in composing functions... You have access to the following functions:

    <tools>{json}</tools>
    <|im_end|>
    ```
  - `{json}` built from each `AIFunction`'s `.Name`, `.Description`, `.JsonSchema`:
    `[{"type":"function","function":{"name":...,"description":...,"parameters":<JsonSchema>}}]`
  - When `tools` null/empty, keep the current plain-chat behavior (no system override).
- Handle `FunctionResultContent` in message rendering: for a `user` message carrying
  `FunctionResultContent`, render a `<|im_start|>user\n<tool_response>{result}\n<|im_end|>` turn
  (skipping the plain `.Text`). Also skip `<tool_response>`/tool-marker text when a message's
  `.Text` already contains them so tool results and final answers render cleanly.
- Handle assistant `FunctionCallContent` round-trip: render any `FunctionCallContent` items back
  as a `<tool_call>` block so history fed back by the framework reconstructs correctly.
- New `TryParseToolCall(string text, out List<(string name, JsonObject args)> calls)`:
  locate `<tool_call>...</tool_call>` (case-insensitive, tolerate whitespace/stray text), parse
  the inner JSON (array or single object), extract `name` + `arguments`. Return false on
  malformed/incomplete input so callers fall back to plain text.

### 2. `SmollmTools.cs` — the weather `AIFunction` (new file)
`E:\khurram-uworx\Nivara\samples\NivaraChat\SmolLM\SmollmTools.cs`

Deterministic, network-free `GetWeather(string city)` (few known cities + generic fallback),
plus `SmollmTools.GetWeatherTools()` returning `AIFunctionFactory.Create(GetWeather)`. One
scalar string parameter, one string result — simplest possible surface for the loop.

### 3. `SmolLMChatClient` — render tools, parse tool calls, emit `FunctionCallContent`
`E:\khurram-uworx\Nivara\samples\NivaraChat\SmolLM\SmolLMChatClient.cs`

- `GetResponseAsync`: pass `options?.Tools` into `SmollmChatTemplate.Render`. After generating
  the full text, call `TryParseToolCall`; if it succeeds, return a `ChatResponse` whose assistant
  message `Contents` carry `FunctionCallContent(callId, name, arguments)` items (one per call,
  arguments deserialized to `Dictionary<string, object?>`); otherwise return plain text as now.
- `GetStreamingResponseAsync`: buffer the full generated text, then decide — no tool call →
  yield plain-text updates as now; tool call → yield a single atomic `ChatResponseUpdate`
  carrying `FunctionCallContent` in its `Contents` (never stream partial tool calls).

### 4. `SmollmMode.cs` — `tools-weather` sub-mode
`E:\khurram-uworx\Nivara\samples\NivaraChat\SmolLM\SmollmMode.cs`

- Extend accepted `Mode` set to `chat | plain | tools-weather`; add `tools-weather` to
  `PrintHelp` and the accepted-mode guard.
- New `RunToolsWeather<T>`: build `SmolLMChatClient<T>` (allow a `--max-new-tokens` default of
  256), wrap with `new FunctionInvokingChatClient(client)`, create `SmollmTools.GetWeatherTools()`.
  - Single-shot (`--text` or default "What's the weather in Paris?"): call
    `funcClient.GetResponseAsync([user], new ChatOptions { Tools = tools })` and print the final
    `ChatResponse.Text`.
  - REPL: loop reading prompts, `history.AddMessages` from each returned `ChatResponse`, print
    the final answer only.
- Wire into `Execute` dispatch. Phase-A `chat`/`plain` paths are untouched.

### 5. `README` docs
`E:\khurram-uworx\Nivara\samples\NivaraChat\README.md`

Document Stage B: the `tools-weather` command, architecture (our render/parse/emit vs.
`FunctionInvokingChatClient`'s invoke/history/re-prompt), and example output.

## Blast radius

- **`samples/NivaraChat/`** only:
  - `SmollmChatTemplate.cs` — renderer signature gains an optional `tools` param; existing
    callers (no tools) see identical output.
  - `SmolLMChatClient.cs` — `GetResponseAsync`/`GetStreamingResponseAsync` now pass tools and
    emit `FunctionCallContent` when a tool call is parsed; plain-chat output unchanged.
  - `SmollmTools.cs` — **new** static helper (no dependency risk).
  - `SmollmMode.cs` — additive `tools-weather` mode; existing modes unchanged.
  - `README.md` — documentation only.
- **No core `src/Nivara` changes.** No new NuGet packages (MEAI 10.9.0 already referenced in
  `NivaraChat.csproj` provides `FunctionInvokingChatClient`, `FunctionCallContent`,
  `FunctionResultContent`, `AITool`, `AIFunctionFactory`, `AIFunction.JsonSchema`).
- **Downstream callers of `SmollmChatTemplate.Render`**: only `SmolLMChatClient` (no external
  callers — internal class).

## Verification

Gated on the `tools-weather` milestone. Sample has no dedicated test project — verification is
build + manual run commands (per AGENTS.md, ask before long-running `dotnet` commands).

1. `dotnet build Nivara.slnx` — compiles cleanly (sample-only, no core changes).
2. Single-shot: `dotnet run --project samples/NivaraChat -- --smollm tools-weather --text "What's the weather in Paris?"`
   Expect final answer referencing Paris weather (e.g. "It's partly cloudy, 18C in Paris.").
3. REPL: repeated prompts, multi-turn history round-trips; only final answers shown, no raw
   `<tool_call>` markup.
4. Malformed tolerance: a malformed `<tool_call>` yields plain text — no exception, no hang.
5. Re-entrancy: several single-shot queries in sequence each behave independently.

## Planned commits (one logical unit each)

1. `feat: render tools and parse tool calls in SmolLM chat template`
2. `feat: add GetWeather AIFunction for SmolLM tool-calling demo`
3. `feat: emit FunctionCallContent from SmolLMChatClient when model emits tool call`
4. `feat: add --smollm tools-weather sub-mode with FunctionInvokingChatClient`
5. `docs: document Stage B SmolLM tool calling in NivaraChat README`
6. `docs: remove TODO.md — plan executed` (after G2 clears)

## GitHub issues log

- Created during execution (see below). Will be filled in as the plan executes.
