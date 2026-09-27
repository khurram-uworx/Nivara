# Qwen2.5-0.5B-Instruct (Qwen/Qwen2.5-0.5B-Instruct)

Native function calling with **Qwen2.5-0.5B-Instruct** served as a plain `Microsoft.Extensions.AI.IChatClient` over Nivara's in-process `LlamaForCausalLM` engine — no LLM server, no Python runtime. `--qwen tools-weather` runs the real `<tool_call>` → `GetWeather` → `<tool_response>` → final-answer loop, capped at 3 iterations, and closes with a clean natural-language answer.

The code lives in `samples/NivaraChat/Qwen/`; the tests in `tests/Nivara.Tests/Qwen/`.

Related Qwen work, tracked separately: #384 (qkvBias loader gap), #386 (Qwen distillation), #387 (BF16-in-memory SIMD-dot), #388 (fused BF16→F32 read), #390 (GGUF backend), #391 (revisit BF16 workarounds), #399 (backlog: `DecodeAttention` per-dot loop at large `kvLen`), #402 (P2: float single-pass sampling + REPL tokenize-prefix cache), #411 (backlog: RMSNorm gamma via `TensorPrimitives.Multiply`), #413/#414 (P2: fuse final RMSNorm + tied LM head), #416 (stretch: INT8 block-quantized weights — blocked on integer support in AutoDiff, see `docs/INTEGERS.md`).

## Model overview

The **second causal LM / generative model** and the function-calling + teacher-distillation showcase. It is a Llama-family causal LM with **biased Q/K/V projections** — the one additive `src/Nivara` gap.

- **Config**: `hidden_size=896`, `intermediate_size=4864`, 24 layers, `num_attention_heads=14`, **`num_key_value_heads=2` (GQA 14↔2)**, SiLU gated FFN, RMSNorm (`eps=1e-6`), RoPE with **`theta=1_000_000`** (10× SmolLM), `max_position_embeddings=32768`, `vocab_size=151936`, tied embeddings, **Q/K/V projections with bias**
- **Tokenizer**: GPT-2 byte-level BPE with an added **`Split` regex pretokenizer** and **added tokens** `<|im_start|>`/`<|im_end|>`/`<|tool_call|>`/`<|tool_response|>`/`<|tool_call_end|>`
- **KV cache**: `LlamaKVCache<T>` + `ForwardCached` — each generated token runs only its own position through the 24 layers instead of re-feeding the whole growing sequence
- **Function calling**: headed-`<tool_call>` JSON with `getWeather`, parsed by `QwenToolParser` and fed back as `<tool_response>`; both rendered prompts are **byte-verified against the HF `apply_chat_template` fixture** (206 + 258 ids, MATCH)
- **Distillation**: `distill` loads the teacher Qwen, classifies the 10 train sentences (FNV-1a word+bigram bag-of-words → 4096-dim features), caches the labels, then trains `SentimentMLP` (`Linear(4096→64)` + ReLU + `Linear(64→2)`) with `Adam<float>(1e-3)` + `CrossEntropyLoss` (~200 full-batch epochs inside `GradientUtils.Grad()`)

### Tool format

`tokenizer_config.json`'s `chat_template` shows Qwen2.5 uses `<tool_call>…</tool_call>` (added-token ids **151657 / 151658**) — *not* the `<|tool_call_start|>` markers of Qwen2, and *not* the SmolLM2 `HermesToolMessage` names. Tool results are rendered as a `user` turn wrapped in `<tool_response>…</tool_response>`.

### Special-token ids

| Token | Id | Role |
|---|---|---|
| `EndOfText` | 151643 | also the bos id; second eos |
| `<|im_start|>` | 151644 | ChatML turn opener |
| `<|im_end|>` | 151645 | primary eos (`eos_token_id[0]`) |
| `<tool_call>` | 151657 | added token, tool-call opener |
| `</tool_call>` | 151658 | added token, tool-call closer |

`generation_config.json` sets `eos_token_id: [151645, 151643]` — generation must stop on **either**. `bos_token_id` is 151643. `QwenChatClient` hard-codes `QwenIds.StopIds = [151645, 151643]`.

### Vocab-size subtlety

- `config.vocab_size` = **151,936** → the embed/head table is `[151936, 896]`.
- The base BPE vocab (`vocab.json`) holds **151,643** entries; generation/argmax must be able to name any of the 151,936 embed rows, so it runs over the config vocab.
- `tokenizer.json` adds 22 more → **151,665** known ids total (the "293-row tail" is the added-token gap between base vocab and embed table).
- Conclusion, applied: embed/head from config (151,936), argmax over `config.VocabSize`, BOS/EOS by id, the added tokens merged into the tokenizer.

### Tokenizer preamble divergence (Split + ByteLevel, not legacy GPT-2)

Qwen's `tokenizer.json` declares `pre_tokenizer: Sequence(Split(Isolated, regex), ByteLevel(use_regex:false))`. The legacy GPT-2 path applied the byte-level regex to the byte-mapped text *first*, which is what SmolLM's `Digits + ByteLevel(use_regex)` pipeline needs. Qwen is different: the **Split regex applies to the raw normalized text, then each chunk is byte-mapped** (`Isolated` semantics, no byte-level regex).

Fix: `Gpt2BpeTokenizer` now reads the declared `Split` regex from a `tokenizer.json` (when present) and routes encoding through `PretokenizeSplit` → per-chunk byte map. SmolLM's construction (no `tokenizer.json` / no Split) still takes the legacy GPT-2 path untouched. This was the fix (`48c789a`) that made Qwen tokenization match the HuggingFace reference.

### The load path (checkpoint → tensors)

```
config.json + model.safetensors + vocab.json + merges.txt + tokenizer.json
  → SafeTensorsLoader.Read<T> (BF16-on-disk, fused SIMD WidenBf16ToF32 → F32, #388)
  → LlamaLoader.Load + LlamaConfig.FromJson → LlamaForCausalLM<T>
  → Gpt2BpeTokenizer  (Qwen Split-regex pretokenizer + added tokens)
```

> **Why we widen to F32 at load (and don't run BF16 compute)** — Qwen is BF16-on-disk (~1 GB), but Nivara **widens to F32 once at load time** and runs a pure-F32 loop. The reason: `Vector<BFloat16>` is **unsupported** on .NET 11 (`IsSupported == false` at every width), so BF16 compute falls back to a scalar BCL path that is radically slower than F32 SIMD, while the widen is **lossless** (BF16 *is* the top 16 bits of float32) and costs only a one-time pass. This is Nivara's current recommendation; its full rationale, the A/B numbers, and the `WidenBf16ToF32` mechanics live in **[`docs/BFLOAT16.md`](BFLOAT16.md)**.
>
> Qwen-fast item **#407** added an opt-in alternative for benchmark/synthetic modes only: `--precision bf16` keeps the weights native `BFloat16` (~942 MB) and widens each lane in-register inside the dot kernel (`NarrowFloatKernels.DotBf16Core` — F32 FMA math on bf16 storage at rest, results rounded back to bf16 for the next layer). This halves weight memory and traffic but measured **uniformly slower** on the i5-1135G7 AVX2/LPDDR4x box used for the P1 verdict (per-lane widen-narrow overhead exceeds the traffic saving); the win is expected on AVX-512 / HBM-class hardware. Full tradeoff and numbers in *Making Qwen fast* below. Tools/distill stay float; fp16 stays rejected.

### The renderer: `QwenChatTemplate` (byte-exact vs Torch)

`Render(IEnumerable<ChatMessage>, bool addGenerationPrompt)` mirrors HuggingFace's `apply_chat_template` for this checkpoint exactly:

- A leading `system` message is used verbatim; otherwise the default `"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."` turn is emitted (the template's `else` branch).
- `user` / `system` turns: `<|im_start|>role\n{text}<|im_end|>\n`.
- `assistant` with tool calls: `<|im_start|>assistant` + per-call `\n<tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>` + `<|im_end|>\n`.
- `role == tool` messages become a `user` turn: `<|im_start|>user\n<tool_response>\n{result}\n</tool_response><|im_end|>\n` (no newline before `<|im_end|>` — the fixtures pin this).
- `addGenerationPrompt` appends `<|im_start|>assistant\n`.

`BuildToolsSystemMessage(tools)` bakes the tool-mode system turn (`# Tools` instructions + `<tools>` schemas + the `<tool_call>` exemplar), byte-identical to the checkpoint template's `{%- if tools %}` branch.

JSON layout is handwritten to reproduce Jinja `tojson`: spaces after `:`/`,`, and `UnsafeRelaxedJsonEscaping` (literal `'`, `°`, no HTML/Unicode escaping) — `QwenJson.ToSpaced`. `JsonObject`/`JsonArray` preserve insertion order, so the schema matches the reference property order.

The byte-exact pins are the ground-truth fixtures `qwen_tool_prompt.txt` / `qwen_tool_final_prompt.txt` in the gitignored model dir, compared byte-for-byte by `QwenChatTemplateTests`. The round-trip test parses the fixture's tool-call turn and re-renders it — the exact parse→render cycle the live loop performs — and asserts the result is the fixture.

### The parser: `QwenToolCallParser`

`Parse(text, knownToolNames)`:

1. Strict `JsonDocument` parse of each `<tool_call>(.*?)</tool_call>` block (spacing-agnostic). Requires `name` (string) + `arguments` (object); `arguments` properties become the `FunctionCallContent.Arguments` dict (values are cloned `JsonElement`s — the "correct dict" fix).
2. On `JsonException` (or wrong shape), a tolerant regex fallback extracts `name` and stores the unparseable remainder under `__raw` so the failure is observable, never silently dropped.
3. `knownToolNames` canonicalizes the emitted name by case-insensitive match (so `GetWeather` still resolves to the registered `getWeather` AIFunction); unknown names pass through as emitted.

### The client: `QwenChatClient<T>`

`IChatClient` over `LlamaForCausalLM<T>` + `Gpt2BpeTokenizer` + `LlamaKVCache<T>`:

- **Inference-default (ADR-001/002):** `model.Eval()`; the client never enters a `GradientUtils.Grad()` scope, so every operation short-circuits to the non-grad span path and no graph nodes are ever built (the ADR-002 inference-default guard).
- **Generation:** render → `tokenizer.Encode` → KV-cached prefill (one batched `ForwardPrefill` over the whole prompt, capturing K/V into the cache) → per-token `ForwardCached` → decode → parse. Greedy (`ArgMax`) is the default; `temperature > 0` switches to a temperature softmax with optional top-p nucleus filtering, drawn from a seeded shared RNG. Stops on `QwenIds.StopIds` (151645 or 151643).
- **`GetResponseAsync`:** returns exactly one assistant message — `FunctionCallContent`s when `<tool_call>` blocks parsed, else a single `TextContent` — and invokes the optional `turnCallback` with the raw decoded text for live printing.
- **`GetStreamingResponseAsync`:** yields one `ChatResponseUpdate` per generated id, decoded per-token.
- No KV-cache path available (`useKvCache: false`): full `model.Forward(ids)` per step (numeric-identical, slower).

### The loop: `--qwen tools-weather`

`QwenMode` wires the client through MEAI 10.9.0's `FunctionInvokingChatClient` as `new FunctionInvokingChatClient(inner) { MaximumIterationsPerRequest = 3 }`, passing `ChatOptions.Tools = [weather]` per request. `QwenSampleTools.GetWeather` is a deterministic `AIFunction` (`getWeather`, description `"Gets the current weather for a city. Returns a short description like 'Sunny, 22°C'."`, param `city`). The system turn is the baked `BuildToolsSystemMessage([weather])` so the first render is byte-identical to the Torch tools prompt.

Observed transcript (real run, F32, greedy):

```
You: What's the weather in Paris?
[assistant → getWeather(city: Paris)]
[tool] Partly cloudy, 18°C. Light breeze from the northwest.
Qwen: The weather in Paris is partly cloudy with a temperature of 18°C. The light breeze from the northwest is expected.
[3 turn(s) in 342567 ms]
```

Two model generations (tool-call turn + final turn), loop closes within the cap (the 343 s figure includes the ~1-min in-process F32 load of the 988 MB BF16 checkpoint). **This transcript is the pre-Qwen-fast baseline (PR #385 era, 2026-09-05):** on the current code the same run is dramatically faster — every prompt token no longer walks the whole model (P0-1 batched prefill), and each decoded token runs the fused block path (`#404`; see *Making Qwen fast* below — current benchmark protocol measures ~116.5 ms/token decode, 13.4× cache-vs-full, on the 16-logical-processor machine). `--smollm chat|plain` is untouched; `--qwen chat|plain` are plain streaming modes with the same generation core.

## How this model improved the library

### Additive `src/Nivara` change — biased Q/K/V projections (#384)

Qwen's attention projections **have bias** (`q_proj.bias` etc.), unlike SmolLM's. `LlamaCausalAttention<T>` gained a `qkvBias` ctor flag (default `false` — SmolLM semantics unchanged); `QProj/KProj/VProj` are built `bias: qkvBias`, `OProj` stays unbiased. Issue #384 tracks the gap.

```csharp
var attn = new LlamaCausalAttention<T>(hiddenSize, numHeads, numKeyValueHeads,
    maxPositionEmbeddings: 32768, ropeTheta: 1_000_000f, qkvBias: true);
var block = new LlamaDecoderBlock<T>(hiddenSize, numHeads, numKeyValueHeads,
    intermediateSize, rmsNormEps: 1e-6f, maxPositionEmbeddings: 32768,
    ropeTheta: 1_000_000f, qkvBias: true);
```

Semantics:
- `qkvBias` defaults to **`false`** — canonical Llama/SmolLM, byte identical to prior releases.
- When `true`, only **Q/K/V** projections carry a `bias` parameter; `OProj` and the FFN projections stay bias-free, matching Qwen2 architecture.
- `LlamaLoader.Load` **auto-detects** biased checkpoints from safetensors (presence of `model.layers.0.self_attn.q_proj.bias`) and passes `qkvBias: true` through — Qwen2.5 loads with zero caller changes.
- Backward is verified end-to-end: `qkvBias=true` forward output, input gradient, and Q/K/V bias gradients match PyTorch bit-for-bit within tolerance.

### What was added where (core vs sample-scoped)

| Piece | Location | Notes |
|---|---|---|
| `qkvBias` flag on `LlamaCausalAttention<T>` / `LlamaDecoderBlock<T>` | **core** `src/Nivara/AutoDiff/Nn/` | the one additive core change (#384); default `false`, SmolLM unchanged |
| `SafeTensorsLoader.Read<float>` fused BF16→F32 (SIMD `WidenBf16ToF32`) | **sample** `samples/Nivara.Samples/` | #388; BF16 widen fused into the read, no interim `ushort[]`, ~1 GB less peak memory |
| `LlamaForCausalLM<T>`, `LlamaConfig`, `LlamaLoader`, `StateDictLoader` | **sample** `samples/Nivara.Samples/` | Qwen loads via the same route SmolLM uses |
| `Gpt2BpeTokenizer` Qwen preamble | **sample** `samples/Nivara.Samples/` | `Split`-regex pretokenize + added-token merge |
| `LlamaKVCache<T>` + `ForwardCached` | **sample** | per-token KV-cached decode |
| `QwenChatClient<T>`, `QwenChatTemplate`, `QwenToolCallParser`, `QwenSampleTools` | **sample** `samples/NivaraChat/Qwen/` | the `IChatClient` + tool loop |
| `TensorsHelper.MultiplyCore` single-row GEMV fast path (`bTransposed && aRows == 1`) | **core** `src/Nivara/Tensors/TensorsHelper.cs` | P0-2 (#398); every decode matmul dots the weight row directly — no rent, no identity `CopyTo`, no clear-on-return |
| `AttentionKernels<T>.DecodeAttention` | **core** `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` | P0-3 (#400); fused single-query GQA decode attention — zero-copy cache reads, virtual GQA mapping, no `BlockCopy` / `GqaRepeatKV` |
| `LlamaCausalAttention<T>.ForwardCore` + `AttentionKernels<T>.BatchedAttention` | **core** `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs` + `Operations/AttentionKernels.cs` | #403; shared core with per-KV-head K/V capture; fused GQA-aware batched prefill attention (no repeat expansion, no mask alloc) |
| `LlamaForCausalLM<T>.ForwardPrefill(int[] ids, LlamaKVCache<T> cache)` | **sample** `samples/Nivara.Samples/` | P0-1; batched `[L, hidden]` prompt prefill with K/V capture — one weight read for the whole prompt instead of L full-model walks |
| `ForwardCachedFused` / `ForwardPrefillFused` + `LlamaFusedKernels.DecoderBlockFused` | **sample** `samples/Nivara.Samples/` | #404; whole per-token block over span/scratch buffers, zero steady-state allocs; public opt-out toggle, default `true` |
| bf16 benchmark/synthetic (`--precision bf16`, `SafeTensorsLoader.Read<BFloat16>`, `NivaraPrimitives.UseWidenSimd`) | **sample** `samples/Nivara.Samples/` + `samples/NivaraChat/` | #407; weights stay `BFloat16` and widen in-register per dot (F32 FMA math); benchmark/synthetic modes only — tools/distill stay float |

### What was not reusable from Phase B (and why)

1. **Tool format.** Phase B rendered SmolLM2-Hermes `HermesToolMessage` style and used `v3` JSON names; Qwen emits plain `<tool_call>` JSON. The parser and renderer are Qwen-specific.
2. **`FunctionInvokingChatClient` binding.** In MEAI 10.9.0 the tool binder consumes `FunctionCallContent.Arguments` as an `IDictionary<string, object?>`. Phase B's `BuildToolCallContents` JSON path was fragile and could produce a mangled dict (the silent-failure root cause). The Qwen client **builds the dict from the parsed JSON** (`QwenToolCallParser`), so the binder always sees a correct, serializable argument map.
3. **Assistant-turn contents.** `ChatMessage.Text` is *read-only* (derived from `Contents`). A tool-call assistant turn carries **only** `FunctionCallContent` — adding `TextContent` alongside double-renders the turn in the next request.
4. **Tool JSON schema source.** `AIFunction.JsonSchema` is unusable byte-for-byte: it emits `description` before `type` and escapes `'` as `\u0027`. The renderer builds the schema **manually** from `AIFunction.UnderlyingMethod` + `[Description]` attributes with type-first property order and literal characters (`QwenChatTemplate.ToolJson`), matching Jinja `tojson`.

## Library features used

| Capability | Where exercised |
|---|---|
| `LlamaCausalAttention<T>` `qkvBias` (biased Q/K/V) | 24 biased-attention layers (14 Q / 2 KV heads, GQA) |
| `LlamaKVCache<float>` + `ForwardCached` | Per-token cached decode (each new token runs only its position) |
| `LlamaForCausalLM<T>` greedy decode (inference-default) | Tool-call turn (19 tok) + answer turn, no `GradientUtils.Grad()` scope |
| Function-calling loop (`QwenToolParser` + `<tool_call>`/`<tool_response>`) | `getWeather` → tool result fed back → final answer |
| `Gpt2BpeTokenizer` `Split`-regex pretokenizer + added tokens | Qwen tokenizer path (byte-verified against the HF fixture, 206/258 ids) |
| `SafeTensorsLoader.Read<float>` fused BF16→F32 (`WidenBf16ToF32`) | Default load (BF16-on-disk → F32, memory-mapped, ~1.3 s median warm Release, ~1.88 GB peak managed) |
| `TensorsHelper.MultiplyCore` single-row GEMV fast path (`bTransposed && aRows == 1`) | #398; every decode matmul (Q/K/V/O proj, FFN, LM head) — 5 → 36 ops/s on the LM head row |
| `AttentionKernels<T>.DecodeAttention` | #400; fused single-query GQA decode attention (zero-copy cache reads, virtual GQA mapping, ~26 KB B/op flat vs kvLen) |
| `LlamaForCausalLM<T>.ForwardPrefill` + `AttentionKernels<T>.BatchedAttention` | #401/#403; batched `[L, hidden]` prefill with K/V capture; fused GQA-aware prefill attention (alloc −6.4% at [256 tok]) |
| `LlamaDecoderBlock<T>.ForwardCachedFused`/`ForwardPrefillFused` + `LlamaFusedKernels.DecoderBlockFused` | #404; fused per-token block, zero-alloc decode (decode block 131,985 → 1 B/op) |
| bf16 benchmark/synthetic (`--precision bf16`; BFloat16 weights widen in-register per dot) | #407; benchmark/synthetic modes only — tools/distill run F32 |
| Teacher distillation inside `GradientUtils.Grad()` | `SentimentMLP` 200-epoch training + linear baseline vs DistilBERT SST-2 eval table |
| FNV-1a word+bigram feature hashing (4096-dim BOW) | Student/linear input features from raw sentences |
| Resumable label cache + `--force` | `qwen_distill_labels.json` merge/recompute |

## Verification

### Torch parity (`QwenInstructParityTests`, 13 tests)

- The model/tokenizer load through `LlamaLoader` / `Gpt2BpeTokenizer` with zero *caller* changes (10 tensor names, `tie_word_embeddings`, q/k/v bias variant).
- **Tool turn: 19/19 ids byte-exact** vs the Torch greedy reference — the structural function-call contract (`<tool_call>` … `</tool_call>`, id-exact).
- **Final turn: semantic parity, with the tie-flip provably a near-tie.** At generated index 9 the F32 model assigned **0.0234 logits** separating `' high'` (19.38) vs `' temperature'` (19.40); PyTorch (computing in BF16, `torch_dtype="auto"`) resolves that hairline differently, so exact token ids diverge. The oracle is decode-then-compare: the final answer is asserted for the weather conclusion (`partly cloudy` / `northwest`), not exact ids.
- Last-row logits over the fixed Torch trajectory: worst observed absolute diff **0.399** on a low-probability tail entry (~2.3% of max-logit magnitude), argmax = 151645 (`<|im_end|>`) intact. Honest tolerance envelope: 3% relative + 0.5 absolute floor.
- Qwen pretokenization parity: the Split-regex finding from above is pinned by tokenizer tests (ids equal the HF `AutoTokenizer` reference).

### BF16 loader gap + benchmark (`SafeTensorsLoaderBf16Tests`, #388)

- **Single fused load**: `SafeTensorsLoader.Read<float>(path)` reads a BF16 checkpoint and widens directly into each tensor's `float[]` via `WidenBf16ToF32(ReadOnlySpan<ushort>, Span<float>)` — a `Vector<ushort>` SIMD chain (`Vector.Widen` → `<<16` → reinterpret) with a scalar tail for partial vectors, run **during the read** with **no interim `ushort[]`** (one pass). All 65,536 BF16 patterns property-match the scalar reference (`WidenBf16ToF32_AllBitPatterns_MatchesScalarReference`).
- **Qwen checkpoint load** (988 MB BF16, on this machine, Release): the fused `Read<float>` finishes in roughly **1.1–1.4 s** (median ~1.3 s warm [#392]) — it does **not regress** the earlier two-step path (2.07–2.16 s [#388]) and its peak **managed** memory drops ~1 GB in two steps: no interim `ushort[]` (#388) and no full-file `byte[]` — the string-path read memory-maps the file and touches each tensor's pages on demand (#392). Peak managed heap: **2.83 GB → 1.88 GB** (measured A/B). The mmap read is ~1.6× slower than a warm `ReadAllBytes` copy (per-page soft-fault overhead for random per-tensor access), so the managed-heap saving trades ~0.5 s on the one-time model load; physical working set is similar either way (the OS page cache holds the file either way). Earlier docs claimed the two-step was "~2.5× faster / half the RAM", but that compared a half-size `ushort[]` output (no widen) against a full-size `float[]` output (with widen) — a meaningless apples-to-oranges metric, since both paths end at F32. With the widen fused in at equal F32 output, the two-step offered no timing or memory win and was removed (#388).

### Qwen tool-calling (this work, 12 new tests)

- `QwenChatTemplateTests` (4): the tool prompt and the assistant-tool-call + tool-response final prompt render **byte-identical to the Torch fixtures**, including the parse→render round trip; plain-chat renders the default system turn; the tools system message contains the instructions + schema.
- `QwenToolCallParserTests` (7): canonical, compact, multi-call, case-insensitive name canonicalization, unknown-name passthrough, no-tool → empty, malformed → tolerant `__raw` fallback.
- `QwenToolsWeatherLoopTests` (1, model-gated, ~5m35s): the real `FunctionInvokingChatClient` loop emits `<tool_call>`, executes `GetWeather`, feeds back the result, and closes with a non-blank final answer containing `partly cloudy` — within the cap.
- CLI acceptance: `--qwen tools-weather --text "What's the weather in Paris?"` → transcript above, clean answer, loop terminates.
- Regression: 17 earlier parity-loader tests re-run green.

### Qwen-fast verification (perf-era suites)

- **P0-2 (GEMV fast path):** full suite **3449 passed / 0 failed**; parity tests lock single-row == row 0 of a two-row run, Dot-vs-Dot bit-exact for float/double, Half/BFloat16 widen-path parity, cold-pool alloc guard.
- **P0-3 (fused decode attention):** `LlamaCausalAttentionTests` 12/12 — fused output equals the `GqaRepeatKV` + `MultiHeadAttention` slow path within 1e-5 across GQA 14/2 · 8/4 · 12/4 · 8/8, virtual-mapping pin, cache-vs-full parity, steady-state zero-alloc guard, no graph node outside `Grad()`.
- **P0-1 (batched prefill):** `LlamaForCausalLMPrefillTests` + KV-cache/attention fixtures 21/21 + full suite **3458 passed / 1 failed** (the failure is the unrelated `Transpose_PerformanceProbe_TiledKernelBeatsBclViewMaterialization` #136 timing probe); real-checkpoint `QwenInstructParityTests` green.
- **#403 (fused batched attention):** targeted 18/18 (GQA parity 4/2 · 8/2 · 14/2 at 1e-5, kernel-level `BatchedAttention` vs the MHA reference, graph guards, KV capture, 36-token plain seat) + `QwenInstructParityTests` plain 36/36 + greedy 7/7; **#408** generated fixtures activated.
- **P1 BF16 (#407):** `LlamaCausalLMBf16ParityTests` 2/2 (greedy argmax matches through full forward, prefill, and 8-step KV-cached decode; logits within prefill/decode ~1e-4); guardrail neighborhood 16 pass / 2 skip (the skips are the pre-existing #406 model-test fixture issue, unrelated).
- **#404 (fused decoder block):** targeted 42/42 — block-level fused-vs-per-op parity **bit-identical** (qkvBias on/off), model-level toggle A/B, prefill parity, bf16 parity (both f32 and bf16 models route fused by default); harness gate `--compare qwen-decode-block-baseline.json --only Qwen --runs 3` → **16/16 PASS**. The parity suite also caught and fixed a real defect during development (in-place PostNorm clobbered the residual — `77b3df8`).
- Full-suite runs after #403/#404/P1-BF16 were deferred by human decision (targeted fixtures + same-state gates accepted; the P0-1 full suite was the last whole-suite run on the perf branch).

## Performance

### Making Qwen fast (Qwen-fast)

The Qwen inference path started from the pre-Qwen-fast baseline above (two generations ≈ 343 s for the weather turn) and went through a measured performance program (issues/PRs #398–#415, branch `khurram/qwen-perf`). Every entry below is executed work with evidence — nothing is speculative. The authoritative row history lives in **`tests/Nivara.PerformanceTests`** (its README documents the measurement protocol: child-process medians via `--runs n`, `--only <substring>` gates, gate defaults minOps 90% (fixed 25% for bandwidth-bound Qwen rows — issue #420) / alloc ≤ baseline × 1.01 / gen0 ≤ baseline + 0.05) with committed same-machine artifacts `qwen-{fast,gqa,prefill,decode-block}-{baseline,postfix}.json`. Caveat that still stands: the 2026-09-08 `qwen-prefill-baseline.json` predates the `--only` filter and is cross-machine — re-baseline fresh per item on the machine you measure on.

#### What the review found (pre-execution, 2026-09-05)

- **Prefill was O(L) full-model walks.** `SeedCache` ran `ForwardCached` once per prompt token; each walk re-read every weight (~2 GB F32), i.e. ≥120 GB of DRAM traffic for a 60-token tool prompt before the first generated token.
- **Every decode matmul copied the weights first.** `MultiplyCore` rented a transposed copy of B (`b.CopyTo(bT)`, clear-on-return) for every `aRows == 1` matmul; the LM head alone identity-copied ~545 MB per token on top of the ~545 MB dot read. Premise correction (measured, sourced from the runtime): `ArrayPool<T>.Shared` *does* pool the 136M-element bucket — the cost was memory traffic + clears, not fresh alloc/GC churn (~2.7 GB/token on the LM head op alone).
- **Decode re-materialized attention every step.** Full KV-prefix `Buffer.BlockCopy`, `GqaRepeatKV` ×7 blowup over the whole prefix, head repack, and a fresh all-zeros mask — `O(newLen·numHeads·headDim)` copies per layer per token.
- **Per-op tensor boxing.** ~150–200 `T[]` + `NivaraColumn<T>` + `ReverseGradTensor` allocations per decoded token.
- F32 decode ceiling at ~30 GB/s effective bandwidth: ~15–20 tok/s (theory).

#### Executed improvements and measured deltas

| Item | What changed | Key measured delta |
|---|---|---|
| **P0-2** single-row GEMV fast path (#398) | `TensorsHelper.MultiplyCore` dots the weight row directly for `bTransposed && aRows == 1` — no rent, no identity copy, no clear | LM head matmul 5 → 36 ops/s (**+620%**); FFN 61 → 529 ops/s (+767%); Q/K/V/O proj +940%; bit-identical numerics |
| **P0-3** fused decode attention (#400) | `AttentionKernels<T>.DecodeAttention`: zero-copy cache reads, virtual GQA mapping, no `BlockCopy`/`GqaRepeatKV`/repack/mask | decode-attn +62–177% across kvLen 64/128/256; B/op **21×/41×/81×↓** to a flat ~26 KB (kvLen-independent) |
| **P0-1** batched prefill (#401) | `LlamaForCausalLM<T>.ForwardPrefill`: one `[L, hidden]` forward captures K/V at absolute positions | prefill 5,566 → **747 ms** (~7.4×); KV-cache total 7,635 → 2,808 ms · 3.1 → 8.5 tok/s (~2.7×) |
| **#403** fused batched attention | `AttentionKernels<T>.BatchedAttention` + `LlamaCausalAttention.ForwardCore` refactor; per-KV-head packs only, no repeat, no mask | same-state A/B seed[256]: +0.1% time, **alloc −6.4%**; all seed rows −5.6…−6.4% |
| **#404** fused decoder block | `ForwardCachedFused`/`ForwardPrefillFused` + public `LlamaFusedKernels.DecoderBlockFused` (default on, opt-out); zero-alloc decode, one rented workspace per layer for prefill | decode block 131,985 → **1 B/op**; decode fwd 3,620,919 → **619,631 B/op** (logits floor); seed rows −96…−99%; gen0 → 0; E2E **116.5 ms/token (6.2 tok/s), 13.4× cache-vs-full** |
| **P1 BF16** on-the-fly (#407) | `--precision bf16`: weights stay `BFloat16` (~942 MB), widened in-register per dot (`NarrowFloatKernels.DotBf16Core`, F32 FMA); benchmark/synthetic only | measured **uniformly slower** on the i5-1135G7 (prefill 1.9–2.2×, decode 1.6–1.8×); halves memory/traffic — win expected on AVX-512/HBM |
| **#408** plain-prompt coverage | `qwen benchmark --plain` / `qwen --plain` + Torch fixtures (`qwen_plain_*`) + parity | 36/36 tokenizer ids, greedy 7/7 ids "The capital of France is Paris."; real-checkpoint plain 36-tok: prefill 1,758 ms, decode 243 ms/tok, **6.1×** |

**Post-perf E2E anchors** (synthetic 64-token prompt + 24-token decode, median of 3, this 16-logical-processor machine): KV-cache decode **116.5 ms/token (6.2 tok/s)**, prefill **941 ms**, cache total **3,737 ms / 24 tok**, full re-forward 2,074 ms/token → **13.4×** cache-vs-full — an ≈4.5–4.9× decode speedup over the pre-#404 ~524–571 ms/token on the same protocol. Real-checkpoint plain seat: prefill 1,758 ms, decode ~243 ms/token, 6.1×. (The P0-1-era 86.2 ms/token decode was a different, faster machine — cross-machine numbers are not comparable.) The dedicated-machine full E2E refresh (distill cycle + benchmark + tools rows in Release) is tracked in **#386**.

#### Decode throughput

| Scenario | Load parse | KV-cached per-token | Full re-forward per-token | Speedup |
|---|---|---|---|---|
| f32 fused, synthetic 64+24-tok prompt/decode (median-of-3, #404) | — (synthetic) | **116.5 ms/token · 6.2 tok/s** | 2,074 ms/tok | **13.4×** |
| f32 fused, real checkpoint, plain 36-tok prompt (warm single-shot, #403/#408) | ~1.3 s median warm (1.1–1.4 s) | **243 ms/tok** | — | 6.1× |
| f32 fused, real checkpoint — **historical pre-Qwen-fast** (2026-09-06) | ~1.3 s median warm (1.1–1.4 s) | 9,153 ms/tok (Release) | 10,757 ms/tok (Debug) | — |

#### Distill eval (`qwen distill --teacher-examples 3`, 2026-09-06, Release build)

| Model | Accuracy |
|---|---|
| Teacher (Qwen2.5-0.5B-Instruct, `classify_sentiment` tool) | 4/8 (50%) |
| Student (`SentimentMLP`, 3 teacher-labeled train rows) | 4/8 (50%) |
| Linear baseline (BOW 4096→2) | 4/8 (50%) |
| DistilBERT SST-2 (dedicated fine-tuned classifier) | 8/8 (100%) |

#### Remaining / not done

| Item | Issue | Notes |
|---|---|---|
| P2 — float single-pass sampling + REPL tokenize-prefix cache | **#402** | `temperature > 0` only; default greedy path unaffected |
| P2 — fuse final RMSNorm + tied LM head into fused decode | **#413** | decode fwd already at the ≈608 KB logits floor; fuses the last chain |
| P2 — fuse final Norm + LM head into prefill | **#414** | |
| Backlog — `DecodeAttention` per-dot loop at large `kvLen` | **#399** | small at Qwen chat context; grows toward 32k positions |
| Backlog — RMSNorm gamma as `TensorPrimitives.Multiply` | **#411** | kernel cleanup; no E2E win expected |
| Backlog — keep BF16 in memory as `ushort` + SIMD-dot (skip load-time widen) | **#387** / **#391** | revisit when `Vector<BFloat16>` SIMD lands |
| Stretch — GGUF backend | **#390** | load-format only; reuses renderer/parser/loop wiring |
| Stretch — INT8 block-quantized weights (llama.cpp-style Q8, ~4× traffic cut, ~65+ tok/s ceiling) | **#416** | **blocked**: AutoDiff requires `IFloatingPointIeee754<T>` — integer-tensor support first (`docs/INTEGERS.md`) |

**Not worth investing now** (recorded so the reasoning survives): precomputing a transposed weight-view cache — weights already live in the transposed-B layout the P0-2 fast path dots directly; parallelizing decode — `aRows == 1` keeps the `ShouldParallelize` gate off by design, parallelism belongs in batched prefill; speculative decoding / top-k tricks — greedy argmax is already a single scan and the win is in memory traffic, not the scan.

### GGUF backend (#390)

A future `--qwen gguf` sub-mode would load the Qwen checkpoint as GGUF through a .NET GGUF library (candidates: LlamaSharp, TensorSharp — see the issue for the evaluation plan) and reuse the *same* `QwenChatTemplate` renderer, `QwenToolCallParser`, and tool-loop wiring — the format contract is model-loading-agnostic. Tracked in #390.

## GPU path

Qwen does not yet have a GPU runner. The GPU investigation for the Llama-family causal LM is in [docs/SMOLLM.md](SMOLLM.md) (the probe-first investigation that established the Level Zero access-chain wall and the DX12/ComputeSharp fallback recommendation). The bf16 benchmark/synthetic support (#407) lives in the sample.
