# NivaraChat — Architecture

## Agents pipeline architecture

```
Input text
    │
    v
[NivaraSentiment]          IChatClient → ChatClientAgent
    │   SentimentTextModel wraps TextClassifierModel<float>
    │   Output: "Positive (confidence: 0.92)" or "Unable to determine sentiment (confidence: 0.31)"
    v
[NivaraEntity]             IChatClient → ChatClientAgent
    │   EntityTextModel wraps TokenClassifierModel<float>
    │   Output: {"person":["John"],"org":["Acme Corp"],"date":["January 15"],"location":[]}
    v
[NivaraValidator]          IChatClient → ChatClientAgent
    │   ValidatorTextModel wraps TextClassifierModel<float>
    │   Output: {"validation":"VALID","confidence":0.87}
    v
[OllamaLLM]                (optional) IChatClient → ChatClientAgent
    │   Receives accumulated results, reasons about confidence signals
    v
Final output: structured result with confidence scores
```

Key design decisions:
- **No conditional edges** — low-confidence signals are expressed in the model output text itself (e.g. "Unable to determine sentiment (confidence: 0.31)"), letting downstream agents — including the LLM — reason about uncertainty naturally
- **Stateless models** — each agent extracts the original user message from the conversation history, ignoring prior turns
- **Same `IChatClient` abstraction** — Nivara models and Ollama LLM use the identical `AsAIAgent()` pipeline, no special executor types needed

## Workflow architecture (fan-out/fan-in)

The `--workflow` mode uses a different graph topology with explicit fan-out/fan-in:

```
Input text
    │
    v
[TextRouter]                   Pass-through, fans out to both analyzers
    │
    ├──> [SentimentExecutor]   Nivara-trained model, deterministic, <1ms
    │        returns: "positive" / "negative" / "neutral"
    │
    └──> [EntityExtractor]     Nivara-trained NER model, deterministic, <1ms
             returns: { person, org, date, location }
    │
    v  (fan-in barrier — waits for both)
[ValidatorExecutor]            Rule-based consistency check, deterministic, <1ms
    │
    v
[LLMAgent]                     (optional) ChatClientAgent + Ollama, stochastic
    │
    v
Final output: structured result
```

## Agent Framework integration patterns

Lessons learned from building this sample with Microsoft.Agents.AI.Workflows. Agent Framework is external; this section captures only Nivara-specific integration notes.

- `Executor<TInput, TOutput>` with `public override` — return value auto-sends downstream
- `.WithOutputFrom()` on `WorkflowBuilder` — registers executors as output sources
- Read `run.NewEvents` for `ExecutorCompletedEvent` (executor output) and `AgentResponseEvent` (LLM output)
- `OllamaApiClient` constructor doesn't throw — actual connection happens on `GetResponseAsync`
- **Workflow objects are single-use per run.** Do not reuse a `Workflow` instance across multiple `InProcessExecution.RunAsync` calls. Create a fresh workflow from the builder for each run (use a factory function / lambda). See [State Isolation](https://learn.microsoft.com/agent-framework/workflows/state#state-isolation).
- **Streaming output** arrives as `AgentResponseUpdateEvent` with one token per event. Accumulate per-executor-ID, then flush on `ExecutorCompletedEvent` or after all events to avoid printing each token on its own line.

Further reading:
- [Microsoft Agent Framework docs](https://learn.microsoft.com/agent-framework/workflows/executors)
- API reference and integration patterns: `docs/research/AGENT-FRAMEWORK.md`

## Architecture

```
NivaraChat/
├── Program.cs                         # Thin CLI dispatcher: option parsing + mode routing
├── Modes/                             # One static class per CLI mode
│   ├── ModeContext.cs                 # Immutable option bag shared by every mode runner
│   ├── ModeHelpers.cs                 # Shared model loaders + agent run/print helpers
│   ├── TrainingMode.cs                # --train / --intent-train
│   ├── WorkflowMode.cs                # --workflow (fan-out/fan-in pipeline, Ollama optional)
│   ├── AgentsMode.cs                  # --agents / --interactive (agents pipeline)
│   ├── HandoffMode.cs                 # --handoff (confidence-based LLM handoff)
│   ├── ToolsMode.cs                   # --tools (LLM orchestrator + Nivara AIFunction tools)
│   ├── CriticMode.cs                  # --critic (writer-critic loop)
│   ├── IntentMode.cs                  # --intent (5-class routing to specialist executors)
│   ├── OnlineLearningMode.cs          # --online-learning (LLM feedback + retrain)
│   ├── EmbeddingMode.cs               # --embed (primitive contrast: parallel arrays + TensorPrimitives)
│   ├── RagMode.cs                     # --rag (the #867 convention: frame + RowCosineSimilarity)
│   ├── RagMultiViewMode.cs            # --rag-multiview (#914: four views, late fusion, disagreement)
│   ├── RagLearnMode.cs                # --rag-learn (#915: AutoDiff fusion weights + query gate)
│   └── RagAgentMode.cs                # --rag-agent (RAG with TextSearchProvider)
├── Executors/                         # Agent Framework Executor<string, string> subclasses
│   ├── TextRouter.cs                  # Pass-through executor for fan-out routing
│   ├── SentimentExecutor.cs           # Sentiment classification executor (--workflow)
│   ├── EntityExtractor.cs             # NER entity extraction executor (--workflow)
│   ├── ValidatorExecutor.cs           # Rule-based validator executor (--workflow)
│   ├── LlmExecutor.cs                 # Ollama LLM executor (--workflow)
│   ├── ConfidenceRouter.cs            # Confidence-based routing executor (--handoff)
│   ├── CriticExecutor.cs              # Scores LLM response quality (--critic)
│   ├── IntentClassifier.cs            # Intent classification executor (--intent)
│   ├── FactualExecutor.cs             # RAG-based factual executor (--intent)
│   ├── QuestionExecutor.cs            # General Q&A executor (--intent)
│   ├── CommandExecutor.cs             # Tool-calling executor (--intent)
│   ├── EscalationExecutor.cs          # Complaint escalation executor (--intent)
│   └── ChitchatExecutor.cs            # Casual conversation executor (--intent)
├── Models/                            # Text-in/text-out wrappers around ML models
│   ├── ITextModel.cs                  # Text-in/text-out abstraction for ML models
│   ├── SentimentTextModel.cs          # ITextModel wrapping TextClassifierModel<float>
│   ├── EntityTextModel.cs             # ITextModel wrapping TokenClassifierModel<float>
│   ├── ValidatorTextModel.cs          # ITextModel wrapping TextClassifierModel<float>
│   ├── NivaraChatClient.cs            # IChatClient wrapping ITextModel for agent participation
│   └── PassthroughTextModel.cs        # ITextModel wrapping IChatClient (Ollama passthrough)
├── Helpers/
│   ├── ModelInferenceHelper.cs        # Shared inference pipeline (DRY)
│   ├── DocumentChunk.cs               # DocumentChunk + DocumentChunker for RAG indexing
│   ├── RagSearch.cs                   # RagIndex + RagSearch (--rag frame store and ranking)
│   ├── RagIndexCache.cs               # #912 single-view Parquet index cache
│   ├── MultiViewChunk.cs              # #914 fence-aware section-scoped multi-view chunker
│   ├── MultiViewScoring.cs            # #914 per-view scoring, late fusion, divergence, rank bands
│   ├── MultiViewExplorer.cs           # #914 agreement buckets and explanations
│   ├── MultiViewIndexCache.cs         # #914 multi-view Parquet index cache
│   ├── MultiViewIndex.cs              # #914 build pipeline (chunk, dedupe, embed, cache, filter)
│   ├── RetrievalTrainingData.cs       # #915 candidate pool, training set builder, labels, query split
│   ├── RetrievalTrainer.cs            # #915 masked fusion, global + gated models, Adam loop, early stop
│   ├── RetrievalEvaluation.cs         # #915 ranking metrics, fixed-baseline path, paired comparison
│   ├── IndexCacheFile.cs              # Shared lock / temp / atomic replace / footer read
│   ├── FeedbackCollector.cs           # LLM fallback + feedback buffer (--online-learning)
│   └── WriterCriticLoop.cs            # Bounded writer-critic retry loop (--critic)
├── Tools/
│   └── NivaraToolFunctions.cs         # Nivara models as AIFunction tools (--tools)
├── Training/
│   ├── SentimentTrainer.cs            # Train sentiment model
│   ├── EntityTrainer.cs               # Train entity NER model
│   ├── ValidatorTrainer.cs            # Train workflow validator model
│   ├── AgentsValidatorTrainer.cs      # Train agents validator model
│   └── IntentTrainer.cs               # Train intent classifier + incremental retrain
├── Data/
│   ├── SyntheticDataGenerator.cs      # Generate all four datasets
│   └── IntentDataGenerator.cs         # Generate 5-class intent data
├── Transformer/
│   ├── TransformerMode.cs             # --tinyshakespeare CLI mode + interactive entry
│   ├── BatchedTransformer.cs          # BatchedTransformer<T> + BatchedTransformerBlock<T>
│   ├── BatchedChatClient.cs           # IChatClient over a trained BatchedTransformer<float>
│   ├── PositionEncoding.cs            # Fixed sinusoidal position encoding
│   └── TinyShakespeare.cs             # Corpus downloader + line-document loader
├── SmolLM/
│   ├── SmollmMode.cs                  # --smollm CLI mode + interactive entry (chat/plain)
│   ├── SmolLMChatClient.cs            # IChatClient over LlamaForCausalLM<T> (greedy/sampled, KV-cached, token-streamed)
│   └── SmollmChatTemplate.cs          # Hermes ChatML conversation rendering
├── Qwen/
│   ├── QwenMode.cs                    # --qwen CLI mode (tools-weather/chat/plain), tool-loop wiring
│   ├── QwenChatClient.cs              # IChatClient over LlamaForCausalLM<T> (<tool_call> parsing, KV-cached, token-streamed)
│   ├── QwenChatTemplate.cs            # Byte-exact Qwen2.5 ChatML renderer (+ Jinja-tojson JSON writer)
│   ├── QwenToolCallParser.cs          # <tool_call> → FunctionCallContent (strict + tolerant, name canonicalization)
│   └── QwenSampleTools.cs             # GetWeather AIFunction for the tools-weather demo
├── NivaraChat.csproj                  # Core + Agent Framework packages
├── (relevance-fixture.tsv moved to samples/data/nivarachat/eval/ — generated, gitignored)
└── README.md                          # This file
```

### Models

**Sentiment (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 3)
```
3 classes: positive, negative, neutral.

**Entity extraction (`TokenClassifierModel<float>`):**
```
Embedding(vocab, 32) → Linear(32, 64) → ReLU → Linear(64, 5)
```
5 classes per token: O, B-person, B-org, B-date, B-location. No MeanPool — per-token predictions.

**Workflow validator (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 2)
```
2 classes: valid, invalid. Trained on `"original || response"` format.

**Agents validator (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 2)
```
2 classes: valid, invalid. Trained on multi-line accumulated pipeline output format.

**Intent classification (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 5)
```
5 classes: factual, question, command, complaint, chitchat.

## TinyShakespeare — batched-transformer `IChatClient` mode (`--tinyshakespeare`)

The former `samples/NivaraChatClient/` companion project now ships as a built-in
mode of this sample: a **word-level batched causal transformer** trained on
TinyShakespeare with Nivara's AutoDiff, then served through the standard
`Microsoft.Extensions.AI.IChatClient` interface and wired up via DI. Where the
rest of this README demonstrates mixing trained ML models with an LLM
(fan-out/fan-in, agents, tools), this mode proves Nivara can *train* a real
transformer and *serve* it in an ecosystem-compatible way:

```
TinyShakespeare.txt → word-level TextTokenizer → batched causal transformer training
    → ModelSerializer.Save/Load → BatchedChatClient : IChatClient
    → services.AddChatClient(factory) → console/ASP.NET/MAUI
```

It is reachable from the interactive menu (option 4, which prompts with
defaults) or directly: `dotnet run --project samples/NivaraChat -- --tinyshakespeare`.

### How it differs from MicroGpt

MicroGpt does a per-position forward with a KV cache; this mode uses a proper
batched causal transformer over `[B, L]` tensors:

| Aspect | MicroGpt | `--tinyshakespeare` |
|---|---|---|
| Forward pass | Per-position, KV cache | Batched sequence `[B, L]` |
| Attention | Per-head dot-product loop | Batched multi-head attention (core `BatchedMultiHeadAttention` op) |
| Embedding | `Forward(int)` single token | `Forward(ReverseGradTensor<T>)` batch `[B, L] → [B, L, D]` |
| Normalization | RMSNorm op | `LayerNorm<T>` module |
| Position encoding | Learned `Embedding` | Sinusoidal (fixed, sample-side) |
| Loss | Hand-rolled NLL | `CrossEntropyLoss<T>` |
| Optimizer | Adam | Adam |
| Training | Manual grad-scope loop | Manual grad-scope loop (same pattern) |
| Serialization | None | `ModelSerializer` full round-trip + tokenizer JSON |
| Inference | Generate via sampling | `IChatClient` standard API |
| Data | Character-level names | Word-level TinyShakespeare |

### Architecture

```
Input tokens: [B, L]
    → Embedding → [B, L, D]
    → + sinusoidal position encoding
    → N × TransformerBlock:
        LayerNorm → Q/K/V projections → BatchedMultiHeadAttention (causal mask) → residual
        LayerNorm → MLP (GELU, expand 4×, compress) → residual
    → LayerNorm → tied LM head (MatMul(x, wteᵀ)) → [B*L, V] logits
```

- `BatchedTransformer<T>` (`Transformer/BatchedTransformer.cs`) composes core modules
  (`Embedding`, `Linear`, `LayerNorm`, `Activation.Gelu`) around the
  `ReverseGradOperations.BatchedMultiHeadAttention` op. The causal `[B, L, L]`
  mask is built once per forward and reused across blocks.
- The tied LM head is `MatMul(x, wteᵀ)`, matching the `[B*L, V]` → `CrossEntropyLoss`
  (shape[0] = batch, shape[1] = classes) convention.
- `BatchedChatClient : IChatClient` (`Transformer/BatchedChatClient.cs`) runs the
  model in eval mode and generates autoregressively with temperature sampling.
  Each call builds its own tensors, so concurrent use is safe; it does **not** own
  the model (disposal belongs to the caller / DI container).

### CLI (options after `--tinyshakespeare`)

Run `--tinyshakespeare --help` for the full list. Options mirror the flags below;
the interactive menu asks for the key ones (model path, vocab size, prompt) with
defaults.

```
--n-embd <int>          Embedding dimension (default: 96)
--n-layer <int>         Transformer layers (default: 2)
--block-size <int>      Context window (default: 64)
--n-head <int>          Attention heads (default: 4)
--dropout <float>       Dropout probability (default: 0.1)
--epochs <int>          Training epochs (default: 20)
--batch-size <int>      Batch size (default: 32)
--lr <float>            Learning rate (default: 3e-3)
--beta1/--beta2 <float> Adam betas (default: 0.9/0.95)
--vocab-size <int>      Max word-vocab size (default: 8000)
--temperature <float>   Sampling temperature (default: 0.8)
--max-new-tokens <int>  Max tokens per reply (default: 96)
--samples <int>         Generated samples (default: 5)
--seed <int>            RNG seed (default: 42)
--data <path>           Corpus path (downloaded to samples/data/tinyshakespeare.txt on first use)
--prompt <text>         Chat with the model using this user prompt
--save <path>           Save trained model to JSON (+ <path>.tokenizer.json)
--load <path>           Load model from JSON (pass the same architecture flags used at save time)
--no-di-demo            Skip the DI + IChatClient demo
--help, -h              Show this help
```

Default behavior: train (when no `--load`), save (when `--save`), print `--samples`
generated replies for `--prompt`, then run a DI demo that resolves `IChatClient`
from a `ServiceCollection` via `AddChatClient(factory)`.

> **CLI delta from the NEXT.md spec:** the spec's `--train`/`--interactive` REPL
> flags were dropped in favor of the NivaraGpt-style default-train + `--load`
> skip-training model; `--seq-len` was renamed `--block-size`. Durable NEXT.md
> content lives here instead.

### Gaps resolved along the way

The NEXT.md gap list is almost entirely obsolete — most items were already fixed
in core before this work (Grounding Audit): `LayerNorm<T>` module, batched
`Embedding<T>.Forward(ReverseGradTensor<T>)` (now Gather-based, no one-hot
MatMul), `ReverseGradOperations.Concat`/`Gather`,
`Embedding<T> : Module<T>`, `CrossEntropyLoss<T>`, `TrainingLoop<T>` +
`DataLoader<T>` + `TensorDataset<T>`, `ModelSerializer`. Two gaps no longer apply:
attention softmax is handled *inside* the MHA op kernel over the last dimension
(`AttentionKernels.SoftmaxRows`), so no public `Softmax(dim)` was needed, and the
"Nivara's MatMul is rank-2 only" workaround (flatten batch×head into one matrix)
is moot because the batched attention op does the batched scores/context math
directly. The remaining gaps were closed by this work:

- **Batched attention op** — `ReverseGradOperations.BatchedMultiHeadAttention<T>`
  (rank-3 `[B, L, D]`, optional `[B, qLen, kvLen]` additive mask, `Parallel.For`
  over the batch past a workload threshold); single-sequence MHA untouched.
  Mirrors PyTorch (see `tests/Nivara.Tests/NivaraTorch/BatchedAttentionTests.cs`).
- **IChatClient thread safety** — eval-mode, re-entrant generation; model
  ownership lives outside the client.

### Not doing (stretch / future)

- Batched KV cache — generation uses a simple autoregressive loop without caching.
- Top-p/top-k sampling, beam search — temperature + random sampling only.
- Fine-tuning/LoRA, quantization — full float32 CPU training only, matching
  Nivara's scope.
- BPE/subword tokenization, multi-modal, ASP.NET hosting — out of scope, but the
  sample is DI-compatible so hosting is a one-liner.

### Notes

- **Performance:** the word vocab dominates cost — core `TensorsHelper.MultiplyCore`
  emits one `TensorPrimitives.Dot` per output element, so the tied LM head over a
  ~8k vocab is ~1M short dot-calls/batch (~730 tok/s at `D=32, B=8`). Smoke runs
  use `--vocab-size 1200`.
- Load requires matching architecture flags; the strict shape validation in
  `LoadStateDict` fails loudly on a mismatch, and the matching
  `<model>.tokenizer.json` is auto-restored.

