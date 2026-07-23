# NivaraChat — Hybrid Agent Workflow

A sample project demonstrating Nivara-trained domain-specific models as first-class participants in a `Microsoft.Agents.AI.Workflows` graph, mixed with an Ollama-backed `ChatClientAgent` node. This is the showcase example for Nivara's value proposition: **deterministic, lightweight, fast models working alongside an LLM in a production workflow.**

**Target audience:** .NET developers building AI workflows, integrating ML models into agent pipelines, exploring hybrid deterministic + stochastic architectures.

## What it does

NivaraChat trains three small domain-specific models (sentiment classifier, entity extractor, output validator) and wires them into a `WorkflowBuilder` graph. With `--ollama`, an Ollama-backed LLM agent is appended after the validator for fluent response generation.

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
    │   checks entity extraction quality
    v
[LLMAgent]                     (optional) ChatClientAgent + Ollama, stochastic
    │   receives validator output
    │   returns: a helpful response
    v
Final output: structured result with confidence score
```

Key characteristics:
- **Fan-out/fan-in topology** — `TextRouter` broadcasts input to SentimentExecutor and EntityExtractor in parallel; results merge at ValidatorExecutor via barrier
- **First Nivara sample with Agent Framework integration** — `Executor<TInput, TOutput>` with `override` for type-safe routing
- **Three separate trained models** in one workflow — exercises `TrainingLoop<T>`, `DataLoader<T>`, `TensorDataset<T>`, `ModelSerializer`
- **Hybrid deterministic + stochastic** — Nivara nodes are fast/consistent; LLM node is fluent/contextual
- **NER (token classification)** — `TokenClassifierModel<T>` for sequence labeling (new core API)
- **Document classification** — `TextClassifierModel<T>` for sentiment and validation (promoted from NivaraClassifier to core)
- **`TextTokenizer`** — word-level tokenizer (promoted from NivaraClassifier to core)
- **Ollama integration** — uses `OllamaSharp` for local LLM inference via `IChatClient`

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
dotnet run --project samples/NivaraChat -- --workflow --text "Jane Doe at TechStart Inc noted issues in San Francisco"
dotnet run --project samples/NivaraChat -- --workflow --text "Acme Corp in New York announced on March 3"
dotnet run --project samples/NivaraChat -- --workflow --text "Bob Wilson will visit Tokyo next week"

# Run workflow with Ollama LLM (requires --ollama flag)
dotnet run --project samples/NivaraChat -- --workflow --ollama http://localhost:11434 --model llama3.2

# Interactive demo
dotnet run --project samples/NivaraChat -- --interactive
```

## CLI options

| Option | Default | Description |
|--------|---------|-------------|
| `--train` | — | Mode: train all three models (overwrites existing) |
| `--workflow` | (default) | Mode: run the workflow pipeline (Nivara-only unless `--ollama` passed) |
| `--interactive` | — | Mode: type text, see full pipeline output |
| `--ollama <url>` | — | Ollama endpoint — only connects when this flag is present |
| `--model <name>` | `llama3.2` | Ollama model name |
| `--text "<message>"` | — | Single-shot: run workflow on one message and exit |

## Modes of use

### Training (`--train`)
Trains all three models on synthetic data. Each model follows the same pattern as NivaraClassifier: generate data → tokenize → build frame → train with `TrainingLoop<T>` → save with `ModelSerializer`. No external datasets required.

### Workflow (`--workflow`)
Runs the full pipeline on user input. Without `--ollama`, the workflow runs Nivara nodes only (TextRouter → [Sentiment, EntityExtractor] → Validator). With `--ollama`, sends structured context to Ollama and validates the LLM response. Use `--text "message"` for single-shot testing. Example phrases: "John Smith from Acme Corp reported great work on January 15", "Acme Corp in New York announced on March 3".

### Interactive (`--interactive`)
Type sentences and see the full pipeline output: sentiment, entities, and optionally LLM response. Without `--ollama`, runs Nivara nodes only. With `--ollama`, also sends to Ollama for a fluent response. Type `quit` to exit.

## Architecture

```
NivaraChat/
├── Program.cs                         # CLI entry, training orchestration, workflow setup
├── TextRouter.cs                      # Pass-through executor for fan-out routing
├── SentimentExecutor.cs               # Sentiment classification executor
├── EntityExtractor.cs                 # NER entity extraction executor
├── ValidatorExecutor.cs               # LLM output validator executor
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
2 classes: consistent, inconsistent. Checks if LLM output matches extracted entities.

### Workflow execution

The workflow uses `Microsoft.Agents.AI.Workflows` for type-safe message routing with fan-out/fan-in topology:

### Executor structure

Each executor wraps a Nivara-trained model using `Executor<TInput, TOutput>` with `override`:

Output is surfaced via `.WithOutputFrom()` on the `WorkflowBuilder` and read from `ExecutorCompletedEvent` in `run.NewEvents`.

## Nivara APIs demonstrated

| API | Where | Purpose |
|-----|-------|---------|
| `TextClassifierModel<T>` | Core (`AutoDiff/Nn/`) | Document-level classification (sentiment, validator) |
| `TokenClassifierModel<T>` | Core (`AutoDiff/Nn/`) | Token-level classification (NER) |
| `TextTokenizer` | Core (`AutoDiff/Nn/`) | Word-level tokenization, vocab, encode/decode |
| `Module<T>` | All executors | Model base class |
| `Embedding<T>` | All models | Learned word embeddings |
| `Linear<T>` | All models | Fully connected layers |
| `CrossEntropyLoss<T>` | Training | Classification loss |
| `Adam<T>` | Training | Optimizer |
| `TrainingLoop<T>` | Training | Training orchestration |
| `DataLoader<T>` | Training | Batched data loading |
| `TensorDataset<T>` | Training | Frame-backed dataset |
| `ModelSerializer.Save/Load` | Training + inference | JSON model persistence |
| `Executor<TInput, TOutput>` | Executors | Workflow node with type-safe routing |
| `WorkflowBuilder` | Program.cs | Workflow graph construction with fan-out/fan-in |
| `AddFanOutEdge` | Program.cs | Broadcast input to multiple executors in parallel |
| `AddFanInBarrierEdge` | Program.cs | Wait for all parallel executors before proceeding |
| `InProcessExecution.RunAsync` | Program.cs | Static workflow execution |
| `ChatClientAgent` | Program.cs | LLM agent node (Ollama) |

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
- **Simple validator** — the validator checks entity consistency via rule-based heuristics, not a trained model. A production validator would use more sophisticated checks.
- **Single-model training** — `--train` trains all three models at once. Individual model training/retuning is not yet supported via CLI.

## File map

| File | Purpose |
|------|---------|
| `Program.cs` | CLI entry, training orchestration, workflow setup, interactive REPL |
| `TextRouter.cs` | Pass-through executor that fans out input to SentimentExecutor and EntityExtractor |
| `SentimentExecutor.cs` | `Executor` subclass wrapping `TextClassifierModel<float>` for sentiment |
| `EntityExtractor.cs` | `Executor` subclass wrapping `TokenClassifierModel<float>` for NER |
| `ValidatorExecutor.cs` | Rule-based `Executor` for LLM output consistency checking |
| `Training/SentimentTrainer.cs` | Training pipeline: synthetic data → tokenize → train → save |
| `Training/EntityTrainer.cs` | Training pipeline for token-level NER model |
| `Training/ValidatorTrainer.cs` | Training pipeline for validator classifier |
| `Data/SyntheticDataGenerator.cs` | Generates sentiment, entity, and validator datasets |
| `NivaraChat.csproj` | Project file referencing Nivara core + Agent Framework |
| `README.md` | This file |
