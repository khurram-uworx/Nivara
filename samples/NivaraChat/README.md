# NivaraChat — Hybrid Agent Workflow

A sample project demonstrating Nivara-trained domain-specific models as first-class participants in a `Microsoft.Agents.AI.Workflows` graph, mixed with an Ollama-backed `ChatClientAgent` node. This is the showcase example for Nivara's value proposition: **deterministic, lightweight, fast models working alongside an LLM in a production workflow.**

**Target audience:** .NET developers building AI workflows, integrating ML models into agent pipelines, exploring hybrid deterministic + stochastic architectures.

## What it does

NivaraChat trains three small domain-specific models (sentiment classifier, entity extractor, output validator) and wires them into a `WorkflowBuilder` graph. With `--ollama`, an Ollama-backed LLM agent is appended after the validator for fluent response generation.

```
Input text
    │
    v
[SentimentExecutor]          Nivara-trained model, deterministic, <1ms
    │   returns: "positive" / "negative" / "neutral"
    v
[EntityExtractor]            Nivara-trained NER model, deterministic, <1ms
    │   returns: { person, org, date, location }
    v
[ValidatorExecutor]          Rule-based consistency check, deterministic, <1ms
    │   checks entity extraction quality
    v
[LLMAgent]                   (optional) ChatClientAgent + Ollama, stochastic
    │   receives validator output
    │   returns: a helpful response
    v
Final output: structured result with confidence score
```

Key characteristics:
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
Runs the full pipeline on user input. Without `--ollama`, the workflow runs Nivara nodes only (Sentiment → EntityExtractor → Validator). With `--ollama`, sends structured context to Ollama and validates the LLM response. Use `--text "message"` for single-shot testing.

### Interactive (`--interactive`)
Type sentences and see the full pipeline output: sentiment, entities, and optionally LLM response. Without `--ollama`, runs Nivara nodes only. With `--ollama`, also sends to Ollama for a fluent response. Type `quit` to exit.

## Architecture

```
NivaraChat/
├── Program.cs                         # CLI entry, training orchestration, workflow setup
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

The workflow uses `Microsoft.Agents.AI.Workflows` for type-safe message routing:

```csharp
var sentimentExecutor = new SentimentExecutor(model, tokenizer);
var entityExecutor = new EntityExtractor(model, tokenizer);
var validatorExecutor = new ValidatorExecutor();

var workflow = new WorkflowBuilder(sentimentExecutor)
    .AddEdge(sentimentExecutor, entityExecutor)
    .AddEdge(entityExecutor, validatorExecutor)
    .WithOutputFrom(sentimentExecutor, entityExecutor, validatorExecutor)
    .Build();

var run = await InProcessExecution.RunAsync(workflow, "Acme Corp owes $5000 by March 15");

foreach (var evt in run.NewEvents)
{
    if (evt is ExecutorCompletedEvent executorEvt)
        Console.WriteLine($"{executorEvt.ExecutorId}: {executorEvt.Data}");
}
```

### Executor structure

Each executor wraps a Nivara-trained model using `Executor<TInput, TOutput>` with `override`:

```csharp
internal sealed class SentimentExecutor : Executor<string, string>
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;

    public SentimentExecutor(TextClassifierModel<float> model, TextTokenizer tokenizer)
        : base("Sentiment")
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
    }

    public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, CancellationToken cancellationToken = default)
    {
        var tokens = _tokenizer.Encode(text, fixedLength: MaxSeqLen);
        var input = ToTensor(tokens);
        var logits = _model.Forward(input);
        int predicted = ArgMax(logits);
        string[] classes = ["negative", "neutral", "positive"];
        return ValueTask.FromResult(classes[predicted]);
    }
}
```

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
| `WorkflowBuilder` | Program.cs | Workflow graph construction |
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

### What this exposed in the library

| Gap | Problem | Resolution |
|-----|---------|------------|
| **No document classifier module** | Simplest classifier requires composing Embedding → MeanPool → Linear manually. | `TextClassifierModel<T>` promoted from NivaraClassifier to core. Reusable for any document classification task. |
| **No token classifier module** | NER/sequence labeling needs per-token predictions without pooling. No core module for this. | `TokenClassifierModel<T>` created in core. Same as TextClassifierModel but without MeanPool — outputs `[B, L, numClasses]`. |
| **No word-level tokenizer** | Only char-level tokenizer existed (MicroGpt). | `TextTokenizer` promoted from NivaraClassifier to core. Word-level with vocab building, encode/decode, special tokens, save/load. |
| **Agent Framework not integrated** | No Nivara sample used `Microsoft.Agents.AI.Workflows`. | `NivaraChat/` references Agent Framework packages. Demonstrates `Executor<TInput, TOutput>` + `WorkflowBuilder` + `InProcessExecution` pattern. |
| **No hybrid workflow example** | Deterministic ML and LLM nodes not shown working together. | Full pipeline: Nivara sentiment → Nivara NER → Ollama LLM → validator. Shows the value of mixing deterministic and stochastic models. |

### ADR-001 opportunistic cleanup

While building this example, removed redundant inner null checks in `Module.cs` (`RegisterModules` and `RegisterParameters`) — under ADR-001, modules/parameters passed to these methods should never be null.

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
| `SentimentExecutor.cs` | `Executor` subclass wrapping `TextClassifierModel<float>` for sentiment |
| `EntityExtractor.cs` | `Executor` subclass wrapping `TokenClassifierModel<float>` for NER |
| `ValidatorExecutor.cs` | Rule-based `Executor` for LLM output consistency checking |
| `Training/SentimentTrainer.cs` | Training pipeline: synthetic data → tokenize → train → save |
| `Training/EntityTrainer.cs` | Training pipeline for token-level NER model |
| `Training/ValidatorTrainer.cs` | Training pipeline for validator classifier |
| `Data/SyntheticDataGenerator.cs` | Generates sentiment, entity, and validator datasets |
| `NivaraChat.csproj` | Project file referencing Nivara core + Agent Framework |
| `README.md` | This file |
