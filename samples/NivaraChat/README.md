# NivaraChat — Hybrid Agent Workflow

A sample project demonstrating Nivara-trained domain-specific models as first-class participants in a `Microsoft.Agents.AI.Workflows` graph, mixed with an Ollama-backed `ChatClientAgent` node. This is the showcase example for Nivara's value proposition: **deterministic, lightweight, fast models working alongside an LLM in a production workflow.**

**Target audience:** .NET developers building AI workflows, integrating ML models into agent pipelines, exploring hybrid deterministic + stochastic architectures.

## What it does

NivaraChat trains three small domain-specific models (sentiment classifier, entity extractor, output validator) and wires them into a workflow graph. Two execution paths are available:

- **`--workflow`** — classic executor-based graph with fan-out/fan-in topology
- **`--agents` / `--interactive`** — each model wrapped as an `IChatClient` via `NivaraChatClient`, participating as `ChatClientAgent`s through `AsAIAgent()`

With `--ollama`, an Ollama-backed LLM agent is appended after the validator for fluent response generation.

## Quick start

```bash
# Train all three models (overwrites existing)
dotnet run --project samples/NivaraChat -- --train

# Run workflow (Nivara nodes only — no LLM needed)
dotnet run --project samples/NivaraChat -- --workflow

# Single-shot test
dotnet run --project samples/NivaraChat -- --workflow --text "I love this product!"

# Multi-word entity examples
dotnet run --project samples/NivaraChat -- --workflow --text "John Smith from Acme Corp reported great work on January 15"
dotnet run --project samples/NivaraChat -- --workflow --text "Acme Corp in New York announced on March 3"

# Run workflow with Ollama LLM
dotnet run --project samples/NivaraChat -- --workflow --ollama http://localhost:11434 --model llama3.2

# Interactive agents mode (Nivara-only)
dotnet run --project samples/NivaraChat -- --interactive

# Interactive agents mode with Ollama LLM
dotnet run --project samples/NivaraChat -- --interactive --ollama http://localhost:11434

# Single-shot agents mode
dotnet run --project samples/NivaraChat -- --agents --text "Jane Doe at TechStart Inc reported issues in San Francisco"
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | — | Mode: train all three models (overwrites existing) |
| `--workflow` | (default) | Mode: executor-based workflow pipeline with fan-out/fan-in |
| `--interactive` | — | Mode: agents pipeline with live interactive input |
| `--agents` | — | Mode: same as `--interactive`, supports `--text` for single-shot |
| `--ollama <url>` | — | Ollama endpoint — only connects when this flag is present |
| `--model <name>` | `llama3.2` | Ollama model name |
| `--text "<message>"` | — | Single-shot: run pipeline on one message and exit |

## Modes of use

### Training (`--train`)
Trains all three models on synthetic data. Each model follows the same pattern: generate data → tokenize → build frame → train with `TrainingLoop<T>` → save with `ModelSerializer`. No external datasets required.

### Workflow (`--workflow`)
Classic executor-based pipeline with fan-out/fan-in topology. `TextRouter` broadcasts input to `SentimentExecutor` and `EntityExtractor` in parallel; results merge at `ValidatorExecutor` via barrier. Without `--ollama`, runs Nivara nodes only. With `--ollama`, appends an LLM agent after the validator.

### Interactive (`--interactive`)
Agents-based pipeline with live input. Each trained model is wrapped as an `IChatClient` via `NivaraChatClient` and participates as a `ChatClientAgent`. Sequential pipeline: NivaraSentiment → NivaraEntity → NivaraValidator. With `--ollama`, an Ollama LLM agent is appended. Type `quit` to exit.

### Agents single-shot (`--agents`)
Same pipeline as `--interactive`, but accepts `--text` for single-shot execution. Useful for scripting and testing.

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
- **Stateless models** — each agent extracts the last user message from the conversation history, ignoring prior turns
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

## Architecture

```
NivaraChat/
├── Program.cs                         # CLI entry, all mode orchestration
├── TextRouter.cs                      # Pass-through executor for fan-out routing
├── SentimentExecutor.cs               # Sentiment classification executor (--workflow)
├── EntityExtractor.cs                 # NER entity extraction executor (--workflow)
├── ValidatorExecutor.cs               # Rule-based validator executor (--workflow)
├── ITextModel.cs                      # Text-in/text-out abstraction for ML models
├── SentimentTextModel.cs              # ITextModel wrapping TextClassifierModel<float>
├── EntityTextModel.cs                 # ITextModel wrapping TokenClassifierModel<float>
├── ValidatorTextModel.cs              # ITextModel wrapping TextClassifierModel<float>
├── NivaraChatClient.cs                # IChatClient wrapping ITextModel for agent participation
├── PassthroughTextModel.cs            # ITextModel wrapping IChatClient (Ollama passthrough)
├── Training/
│   ├── SentimentTrainer.cs           # Train sentiment model
│   ├── EntityTrainer.cs              # Train entity NER model
│   └── ValidatorTrainer.cs           # Train validator model
├── Data/
│   └── SyntheticDataGenerator.cs     # Generate all three datasets
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

**Validator (`TextClassifierModel<float>`):**
```
Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 2)
```
2 classes: valid, invalid. Checks if extracted data is consistent and meaningful.

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `TextClassifierModel<T>` | Core (`AutoDiff/Nn/`) | Document-level classification (sentiment, validator) |
| `TokenClassifierModel<T>` | Core (`AutoDiff/Nn/`) | Token-level classification (NER) |
| `TextTokenizer` | Core (`AutoDiff/Nn/`) | Word-level tokenization, vocab, encode/decode |
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
| `IChatClient` | NivaraChatClient.cs | Microsoft.Extensions.AI chat abstraction |
| `AsAIAgent()` | Program.cs | Convert `IChatClient` to `ChatClientAgent` |
| `ChatClientAgent` | Program.cs | Agent Framework participant from `IChatClient` |

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
| `OllamaSharp` | 5.4.27 | `OllamaApiClient` implementing `IChatClient` |

## Library gaps this example resolved

### Core additions driven by this example

| New API | Location | Purpose |
|---------|----------|---------|
| `TextClassifierModel<T>` | `src/Nivara/AutoDiff/Nn/TextClassifierModel.cs` | Promoted from NivaraClassifier. Embedding → MeanPool → MLP document classifier. |
| `TokenClassifierModel<T>` | `src/Nivara/AutoDiff/Nn/TokenClassifierModel.cs` | New. Embedding → MLP per-token classifier for NER and sequence labeling. |
| `TextTokenizer` | `src/Nivara/AutoDiff/Nn/TextTokenizer.cs` | Promoted from NivaraClassifier. Word-level tokenizer with vocab, encode/decode, special tokens, save/load. |

## Limitations

- **Word-level tokenization** — no subword (BPE) support. Out-of-vocabulary words map to UNK. Sufficient for synthetic data.
- **Synthetic training data** — entity extraction and validation use template-based synthetic data. Real applications would use annotated corpora.
- **No LLM streaming** — the workflow runs non-streaming. The LLM response is collected in full before validation.
- **Single-model training** — `--train` trains all three models at once. Individual model training/retuning is not yet supported via CLI.
- **Sequential agents** — the agents pipeline runs sequentially (Sentiment → Entity → Validator). Fan-out parallelism is only available in `--workflow` mode.
