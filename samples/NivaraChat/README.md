# NivaraChat — Hybrid Agent Workflow

A sample project demonstrating Nivara-trained domain-specific models as first-class participants in a `Microsoft.Agents.AI.Workflows` graph, mixed with an Ollama-backed `ChatClientAgent` node. This is the showcase example for Nivara's value proposition: **deterministic, lightweight, fast models working alongside an LLM in a production workflow.**

**Target audience:** .NET developers building AI workflows, integrating ML models into agent pipelines, exploring hybrid deterministic + stochastic architectures.

## What it does

NivaraChat trains four small domain-specific models (sentiment classifier, entity extractor, workflow validator, agents validator) and wires them into a workflow graph. Two execution paths are available:

- **`--workflow`** — classic executor-based graph with fan-out/fan-in topology
- **`--agents` / `--interactive`** — each model wrapped as an `IChatClient` via `NivaraChatClient`, participating as `ChatClientAgent`s through `AsAIAgent()`

With `--ollama`, an Ollama-backed LLM agent is appended after the validator for fluent response generation.

## Quick start

```bash
# Train all four models (overwrites existing)
dotnet run --project samples/NivaraChat -- --train

# Run workflow (Nivara nodes only — no LLM needed)
dotnet run --project samples/NivaraChat -- --workflow

# Single-shot test
dotnet run --project samples/NivaraChat -- --workflow --text "I love this product!"

# Multi-word entity examples
dotnet run --project samples/NivaraChat -- --workflow --text "John Smith from Acme Corp reported great work on January 15"
dotnet run --project samples/NivaraChat -- --workflow --text "Acme Corp in New York announced on March 3"

# Run workflow with Ollama LLM
dotnet run --project samples/NivaraChat -- --workflow --ollama --model llama3.2

# Agents mode (Nivara-only, single-shot)
dotnet run --project samples/NivaraChat -- --agents --text "Jane Doe at TechStart Inc reported issues in San Francisco"

# Agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --agents --ollama --text "Acme Corp in New York announced on March 3"

# Interactive agents mode (Nivara-only)
dotnet run --project samples/NivaraChat -- --interactive

# Interactive agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --interactive --ollama

# Confidence handoff — Nivara decides if LLM is needed (requires --ollama)
dotnet run --project samples/NivaraChat -- --handoff --ollama --text "I love this product!"
dotnet run --project samples/NivaraChat -- --handoff --ollama --text "This product is interesting but I'm not sure"

# Tool calling — LLM orchestrates Nivara models as AIFunction tools (requires --ollama)
dotnet run --project samples/NivaraChat -- --tools --ollama --text "John Smith from Acme Corp reported great work"

# Writer-critic loop — LLM writes, Nivara scores, retry if poor (requires --ollama)
dotnet run --project samples/NivaraChat -- --critic --ollama --text "Explain quantum computing to a 5-year-old"

# Embedding search (index documents, retrieve context via IEmbeddingGenerator)
dotnet run --project samples/NivaraChat -- --embed

# RAG pipeline: chunk docs, retrieve context, LLM generate answer (requires --ollama)
dotnet run --project samples/NivaraChat -- --rag --ollama --text "How does embedding search work?"

# RAG agent: same with TextSearchProvider auto-context injection (requires --ollama)
dotnet run --project samples/NivaraChat -- --rag-agent --ollama --text "What is NivaraChat?"
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | — | Mode: train all four models (overwrites existing) |
| `--workflow` | — | Mode: executor-based workflow pipeline with fan-out/fan-in |
| `--interactive` | — | Mode: agents pipeline with live interactive input |
| `--agents` | — | Mode: same as `--interactive`, supports `--text` for single-shot |
| `--handoff` | — | Mode: confidence-based handoff — Nivara decides if LLM is needed |
| `--tools` | — | Mode: LLM orchestrator calls Nivara models as AIFunction tools |
| `--critic` | — | Mode: writer-critic loop — LLM writes, Nivara scores, retry if poor |
| `--embed` | — | Mode: embedding search — index documents, retrieve context via `IEmbeddingGenerator` |
| `--rag` | — | Mode: RAG pipeline — chunk markdown docs, retrieve via vector search, LLM generates answer |
| `--rag-agent` | — | Mode: RAG agent — same as `--rag` with `TextSearchProvider` auto-context injection |
| `--intent-train` | — | Mode: train intent classifier (5 classes) |
| `--intent` | — | Mode: intent routing — classify input and route to specialist executor |
| `--text <message>` | — | Single-shot: run pipeline on one message and exit |
| `--ollama [url]` | — | Flag: enable Ollama LLM agent (optional URL, default: `http://localhost:11434`) |
| `--model <name>` | `llama3.2` | Ollama model name |
| `--threshold <float>` | `0.8` | Confidence threshold for `--handoff` mode |
| `--docs-dir <path>` | `docs/` | Documents directory for `--rag` and `--rag-agent` modes |
| `--top-k <int>` | `3` | Number of chunks to retrieve for RAG modes |

## Modes of use

### Training (`--train`)
Trains all four models on synthetic data: sentiment classifier, entity extractor, workflow validator, and agents validator. Each model follows the same pattern: generate data → tokenize → build frame → train with `TrainingLoop<T>` → save with `ModelSerializer`. No external datasets required.

### Workflow (`--workflow`)
Classic executor-based pipeline with fan-out/fan-in topology. `TextRouter` broadcasts input to `SentimentExecutor` and `EntityExtractor` in parallel; results merge at `ValidatorExecutor` via barrier. Without `--ollama`, runs Nivara nodes only. With `--ollama`, appends an LLM agent after the validator.

### Agents (`--agents`)
Sequential pipeline where each trained model is wrapped as an `IChatClient` via `NivaraChatClient` and participates as a `ChatClientAgent`. Supports `--text` for single-shot execution. With `--ollama`, an Ollama LLM agent is appended after the validator.

### Interactive (`--interactive`)
Same as `--agents` but with live input. Type `quit` to exit. With `--ollama`, the LLM agent is appended after the validator.

### Confidence handoff (`--handoff`)
Demonstrates the hybrid deterministic/stochastic pattern. Nivara models run first; if both sentiment and entity extraction are confident (>= `--threshold`, default 0.8), the result is returned without calling the LLM. If either is uncertain, the partial Nivara results are forwarded to the LLM for enrichment. Requires `--ollama`.

```
Input text
    │
    v
[TextRouter] --fan-out--> [SentimentExecutor, EntityExtractor]
                               │
                          fan-in barrier
                               │
                               v
                        [ConfidenceRouter]
                         /           \
              confident (>=0.8)    uncertain (<0.8)
                    │                    │
                    v                    v
            Nivara result          [Ollama LLM]
```

Tested examples:

| Input | Threshold | Path taken | Why |
|-------|-----------|------------|-----|
| `"I love this product!"` | 0.8 (default) | LLM | Sentiment 0.58, entity 0.27 — both below threshold |
| `"I love this product!"` | 0.7 | LLM | Entity confidence 0.27 still below 0.7 |
| `"John Smith from Acme Corp reported great work on January 15"` | 0.8 | LLM | Entity confidence 0.798 just below 0.8 |
| `"John Smith from Acme Corp reported great work on January 15"` | 0.7 | Nivara only | Both above 0.7 — no LLM needed |

The entity model's average per-token confidence tends to cap around 0.8 for multi-entity inputs. Use `--threshold 0.7` for a more practical cutoff.

### Tool calling (`--tools`)
Flips the architecture: the LLM *decides* when to call Nivara models. Nivara models are wrapped as `AIFunction` tools via `AIFunctionFactory` with `[Description]` attributes. The LLM receives tool definitions and chooses when to invoke sentiment analysis, entity extraction, or response validation. Requires `--ollama`.

Tested examples:

| Input | Tools called | Notes |
|-------|-------------|-------|
| `"John Smith from Acme Corp reported great work"` | ExtractEntities, AnalyzeSentiment | LLM chose both tools, summarized results |
| `"Acme Corp in New York announced on March 3"` | ExtractEntities, AnalyzeSentiment | Multi-entity extraction works well |

The LLM decides which tools to call based on the `[Description]` attributes. Tool results are fed back automatically by the `ChatClientAgent` framework.

### Writer-critic loop (`--critic`)
The LLM generates a response, a Nivara validator model scores it for quality/consistency, and the LLM re-generates if the score is below threshold. Bounded to 3 iterations with structured feedback. Demonstrates Nivara models evaluating LLM output, not just generating their own. Requires `--ollama`.

Tested examples:

| Input | Result | Score |
|-------|--------|-------|
| `"Explain quantum computing to a 5-year-old"` | PASS on attempt 1 | 0.98 |

The validator model was trained on `"original || response"` format for consistency checking. High scores indicate the response is consistent with the query. Max 3 iterations — if all fail, the last attempt is returned with a notice.

### Embedding search (`--embed`)
Indexes 8 knowledge documents using `IEmbeddingGenerator` backed by a local MiniLM transformer, then runs an interactive REPL. Type a query and the system retrieves the top-4 most relevant documents ranked by cosine similarity. This demonstrates the retrieval step for RAG (Retrieval-Augmented Generation) — in a full pipeline, retrieved context would be injected into the LLM prompt via `TextSearchProvider`. Uses `NivaraEmbeddingGenerator<string>` from `Nivara.Extensions`, the same interface as OpenAI/Ollama embedding providers.

### RAG pipeline (`--rag`)
Full Retrieval-Augmented Generation pipeline. Loads real Nivara documentation (markdown files from `docs/` + `README.md`), chunks them into paragraphs, indexes via `InMemoryVectorStore` with auto-embedding from the local MiniLM model, then runs an interactive REPL. User questions are matched against stored chunks via cosine similarity, top-K chunks are injected into a manually constructed prompt, and the LLM generates a grounded answer. Shows retrieval time and LLM time separately. Requires `--ollama`.

```
Documents (docs/*.md, README.md)
    │
    v
ChunkText (paragraph splitting, ~500 chars)
    │
    v
InMemoryVectorStore + MiniLMEmbeddingGenerator
    │  auto-embeds each chunk via Nivara IEmbeddingGenerator
    v
User query
    │
    v
collection.SearchAsync(query, top: K)  →  ranked chunks
    │
    v
Manual prompt: "Answer based on context:\n{chunks}\n\nQuestion: {query}"
    │
    v
Ollama LLM  →  grounded response
```

Tested examples:

| Input | Top-K | Retrieval | LLM | Answer quality |
|-------|-------|-----------|-----|----------------|
| `"How does embedding search work?"` | 3 | 371ms | 27.5s | Describes MiniLM → InMemoryVectorStore → cosine similarity pipeline |
| `"What is NivaraChat?"` | 3 | 538ms | 17.2s | Correctly identifies RAG pipeline with MiniLM + TextSearchProvider |

Uses: `MiniLMEmbeddingGenerator.Create()`, `InMemoryVectorStore`, `DocumentChunker.ChunkText()`, `collection.SearchAsync()`.

### RAG agent (`--rag-agent`)
Same retrieval pipeline as `--rag`, but uses `TextSearchProvider` from the Agent Framework for automatic context injection instead of manual prompt construction. `TextSearchProvider` intercepts each LLM call, performs a search, and injects the retrieved context before the LLM sees the query. This is the standard ecosystem pattern for RAG and composes with other `AIContextProvider` implementations. Requires `--ollama`.

```
Documents (same as --rag)
    │
    v
InMemoryVectorStore + MiniLMEmbeddingGenerator
    │
    v
TextSearchProvider (SearchTime = BeforeAIInvoke)
    │  auto-searches before every LLM call
    │  injects top-K chunks as additional context
    v
ChatClientAgent + Ollama LLM
    │
    v
Grounded response with source citations
```

Tested examples:

| Input | Answer quality |
|-------|----------------|
| `"How does embedding search work?"` | Describes MiniLM → InMemoryVectorStore → auto-embedding pipeline with code example |
| `"What is NivaraChat?"` | Identifies as Nivara project component for RAG pipeline |

### Intent routing (`--intent`)
5-class intent classifier routes user input to specialist executors using conditional edges. Requires `--ollama` for specialist executors (except escalation). Training produces `models/intent_model.json` and `models/intent_tokenizer.json`.

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel, 5 classes
    │
    ├── "factual"      ──> [FactualExecutor]       RAG retrieval + LLM generation
    ├── "question"     ──> [QuestionExecutor]      General Q&A via Ollama
    ├── "command"      ──> [CommandExecutor]       LLM with AIFunction tools
    ├── "complaint"    ──> [EscalationExecutor]    Human-in-the-loop (no LLM)
    └── "chitchat"     ──> [ChitchatExecutor]      Casual conversation via Ollama
```

Tested examples:

| Input | Intent | Response quality |
|-------|--------|------------------|
| `"I'm unhappy with the service"` | complaint | Escalation message with timestamp |
| `"What is the capital of France?"` | question | LLM answer: Paris |
| `"Hello!"` | chitchat | Friendly greeting |

Uses: `IntentClassifier`, `FactualExecutor`, `QuestionExecutor`, `CommandExecutor`, `EscalationExecutor`, `ChitchatExecutor`, `AddEdge<string>` conditional routing.

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
- API reference and integration patterns: `docs/RESEARCH-AGENT-FRAMEWORK.md`

## Architecture

```
NivaraChat/
├── Program.cs                         # CLI entry, all mode orchestration
├── TextRouter.cs                      # Pass-through executor for fan-out routing
├── SentimentExecutor.cs               # Sentiment classification executor (--workflow)
├── EntityExtractor.cs                 # NER entity extraction executor (--workflow)
├── ValidatorExecutor.cs               # Rule-based validator executor (--workflow)
├── LlmExecutor.cs                     # Ollama LLM executor (--workflow)
├── ConfidenceRouter.cs                # Confidence-based routing executor (--handoff)
├── NivaraResultFormatter.cs           # Formats confident Nivara results (--handoff)
├── CriticExecutor.cs                  # Scores LLM response quality (--critic)
├── WriterCriticLoop.cs                # Bounded writer-critic retry loop (--critic)
├── NivaraToolFunctions.cs             # Nivara models as AIFunction tools (--tools)
├── ITextModel.cs                      # Text-in/text-out abstraction for ML models
├── SentimentTextModel.cs              # ITextModel wrapping TextClassifierModel<float>
├── EntityTextModel.cs                 # ITextModel wrapping TokenClassifierModel<float>
├── ValidatorTextModel.cs              # ITextModel wrapping TextClassifierModel<float>
├── NivaraChatClient.cs                # IChatClient wrapping ITextModel for agent participation
├── PassthroughTextModel.cs            # ITextModel wrapping IChatClient (Ollama passthrough)
├── ModelInferenceHelper.cs            # Shared inference pipeline (DRY)
├── Training/
│   ├── SentimentTrainer.cs           # Train sentiment model
│   ├── EntityTrainer.cs              # Train entity NER model
│   ├── ValidatorTrainer.cs           # Train workflow validator model
│   ├── AgentsValidatorTrainer.cs     # Train agents validator model
│   └── IntentTrainer.cs              # Train intent classifier model
├── Data/
│   ├── SyntheticDataGenerator.cs     # Generate all four datasets
│   └── IntentDataGenerator.cs        # Generate 5-class intent data
├── IntentClassifier.cs               # Intent classification executor (--intent)
├── FactualExecutor.cs                # RAG-based factual executor (--intent)
├── QuestionExecutor.cs               # General Q&A executor (--intent)
├── CommandExecutor.cs                # Tool-calling executor (--intent)
├── EscalationExecutor.cs             # Complaint escalation executor (--intent)
├── ChitchatExecutor.cs               # Casual conversation executor (--intent)
├── NivaraChat.csproj                  # Core + Agent Framework packages
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

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `TextClassifierModel<T>` | `Nivara.Samples` (shared with NivaraClassifier) | Document-level classification (sentiment, validator) |
| `TokenClassifierModel<T>` | `Nivara.Samples` | Token-level classification (NER) |
| `TextTokenizer` | `Nivara.Samples` (shared with NivaraClassifier) | Word-level tokenization, vocab, encode/decode |
| `Module<T>` | All models | Model base class |
| `Embedding<T>` | All models | Learned word embeddings |
| `Linear<T>` | All models | Fully connected layers |
| `CrossEntropyLoss<T>` | Training | Classification loss |
| `Adam<T>` | Training | Optimizer |
| `TrainingLoop<T>` | Training | Training orchestration |
| `DataLoader<T>` | Training | Batched data loading |
| `TensorDataset<T>` | Training | Frame-backed dataset |
| `ModelSerializer.Save/Load` | Training + inference | JSON model persistence |
| `Executor<TInput, TOutput>` | Executors (`--workflow`) | Workflow node with type-safe routing |
| `WorkflowBuilder` | Program.cs | Workflow graph construction with fan-out/fan-in |
| `AddFanOutEdge` | Program.cs | Broadcast input to multiple executors in parallel |
| `AddFanInBarrierEdge` | Program.cs | Wait for all parallel executors before proceeding |
| `InProcessExecution.RunAsync` | Program.cs | Static workflow execution |
| `AIFunctionFactory.Create` | NivaraToolFunctions.cs | Wrap static methods as LLM-callable tools |
| `IChatClient` | NivaraChatClient.cs | Microsoft.Extensions.AI chat abstraction |
| `AsAIAgent()` | Program.cs | Convert `IChatClient` to `ChatClientAgent` |
| `ChatClientAgent` | Program.cs | Agent Framework participant from `IChatClient` |
| `NivaraEmbeddingGenerator<T>` | Nivara.Extensions | `IEmbeddingGenerator<TInput, Embedding<float>>` implementation for local models |

## Requirements

- .NET 10.0 SDK
- Nivara core library (`src/Nivara/Nivara.csproj`)
- Ollama (optional — only when `--ollama` flag is used; install from [ollama.com](https://ollama.com))

### Packages (example project only — core stays clean)

| Package | Version | Purpose |
|---------|---------|---------|
| `Microsoft.Agents.AI` | 1.15.0 | `ChatClientAgent` for LLM integration |
| `Microsoft.Agents.AI.Workflows` | 1.15.0 | `Executor`, `WorkflowBuilder`, `InProcessExecution` |
| `Microsoft.Agents.AI.Workflows.Generators` | 1.15.0 | Source generator for `[MessageHandler]` |
| `Microsoft.Extensions.AI` | 10.8.1 | `IChatClient` abstraction |
| `OllamaSharp` | 5.4.30 | `OllamaApiClient` implementing `IChatClient` |

## Library gaps this example resolved

### Library additions driven by this example

| New API | Location | Purpose |
|---------|----------|---------|
| `TextClassifierModel<T>` | `samples/Nivara.Samples/TextClassifierModel.cs` | Embedding → MeanPool → MLP document classifier. |
| `TokenClassifierModel<T>` | `samples/Nivara.Samples/TokenClassifierModel.cs` | Embedding → MLP per-token classifier for NER and sequence labeling. |
| `TextTokenizer` | `samples/Nivara.Samples/TextTokenizer.cs` | Word-level tokenizer with vocab, encode/decode, special tokens, save/load. |
| `MiniLMEmbeddingGenerator` | `samples/Nivara.Samples/BertModel.cs` | Factory wiring MiniLM weights + BertTokenizer into `NivaraEmbeddingGenerator<string>`. |

## Limitations

- **Word-level tokenization** — no subword (BPE) support. Out-of-vocabulary words map to UNK. Sufficient for synthetic data.
- **Synthetic training data** — entity extraction and validation use template-based synthetic data. Real applications would use annotated corpora.
- **No LLM streaming** — the workflow runs non-streaming. The LLM response is collected in full before validation.
- **Sequential agents** — the agents pipeline runs sequentially (Sentiment → Entity → Validator). Fan-out parallelism is only available in `--workflow` mode.
