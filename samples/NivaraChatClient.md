# NivaraChatClient — Agent Orchestration with a Nivara-trained Domain Model

## Goal

Train a Nivara-backed domain-specific model (classifier, scorer, or NER extractor) and make it a first-class participant in a `Microsoft.Agent.Framework` workflow. The model is not an LLM competitor; it does deterministic tasks that LLMs are unreliable at, and works alongside an LLM in a workflow graph.

## Motivation

Simple LLM agents can hallucinate or produce inconsistent results for tasks requiring precision: sentiment scoring, entity extraction, document classification, or numeric validation. Nivara-trained models are:

- Deterministic — same input always yields same output
- Lightweight — no API calls, no network
- Fast — CPU-only, sub-millisecond inference
- Trainable on private/corporate data

The Microsoft Agent Framework's `WorkflowBuilder` (from `Microsoft.Agents.AI.Workflows`) allows mixing:
- Custom `Executor` subclasses for deterministic computation (Nivara models)
- `ChatClientAgent` nodes for LLM-backed reasoning

This example shows how to build a hybrid workflow using both.

## Architecture

### What we build

```
Input text
    │
    v
[NivaraSentimentExecutor]       custom Executor, Nivara-trained model
    │   returns: one of "positive", "negative", "neutral"
    v
[NivaraEntityExtractor]         custom Executor, Nivara-trained model
    │   returns: structured entities (person, org, date, location)
    v
[LLMAgent]                      ChatClientAgent with OpenAI/Ollama/Azure
    │   receives sentiment + entities + original text
    │   returns: a helpful response
    v
[NivaraValidator]               custom Executor, Nivara-trained model
    │   checks LLM output for factual consistency
    v
Final output
```

The workflow graph is defined once in `WorkflowBuilder` and all routing is type-safe at compile time.

### What this exercises in Nivara

| Feature | How it's used |
|---------|---------------|
| **Training Loop** | Train sentiment / entity / validator models |
| **Module system** | `Module<T>` base for each model |
| **LayerNorm / Linear / Embedding** | Building blocks for classification heads |
| **ModelSerializer** | Save trained models to file, load at pipeline start |
| **StateDict / LoadStateDict** | Transfer model state between training and production |
| **Softmax + CrossEntropyLoss** | For probability-based classification |
| **Training** | Each model trained independently with synthetic or CSV data |

### What this discovers and fixes in Nivara's library (Gaps)

#### Gap 1: Embedding\<T\> does not extend Module\<T\>

**Problem:** `Embedding<T>` implements `IDisposable` but not `Module<T>`. It cannot be used with `TrainingLoop<T>`, `StateDict()`, `LoadStateDict()`, or `ModelSerializer`.

**Fix:** Change `Embedding<T>` from `IDisposable` to `Module<T>`. Remove its own `Parameters` property (inherited from `Module<T>`). Remove its own weights list (use Module's built-in `RegisterParameters`). The `Forward(int)` method remains; add a batch `Forward(ReverseGradTensor<T> tokenIds)` method (see Gap 2).

#### Gap 2: Embedding\<T\> has no batch/lookup Forward for token IDs

**Problem:** `Embedding<T>.Forward(int)` only supports single token lookup. A batched transformer needs `Embedding<T>.Forward(ReverseGradTensor<T> tokenIds)`.

**Fix:** Add `Forward(ReverseGradTensor<T> tokenIds)` where `tokenIds` is an integer tensor `[B, L]`. The implementation uses Gather (preferred) or one-hot + MatMul (fallback) to select embeddings.

#### Gap 3: CrossEntropyLoss already exists — reuse

**Check:** `CrossEntropyLoss<T>` exists in `src/Nivara/AutoDiff/Nn/Functional/CrossEntropyLoss.cs`. Reuse directly. No fix needed.

#### Gap 4: No simple classifier module in core library

**Problem:** The simplest classifier is a `Sequential(Linear, Softmax)`. Convenience class would help.

**Fix:** Add a `LinearClassifier<T>` convenience class in `AutoDiff/Nn/` once the need is confirmed after building the example.

#### Gap 5: Agent Framework packages not used anywhere

**Problem:** Nivara core has no dependency on Agent Framework packages. The example project references them; core stays clean.

**Fix:** Documentation/example concern only — no core library change.

### Example NivaraSentimentExecutor

This is the key showcase: a Nivara-trained model operating as a custom Workflow Executor.

```csharp
using Microsoft.Agents.AI.Workflows;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Serialization;
using System.Numerics;

// Trained linear classifier model wrapped in an executor
internal sealed partial class NivaraSentimentExecutor : Executor
{
    private readonly LinearClassifier<float> _model;

    public NivaraSentimentExecutor(LinearClassifier<float> model)
        : base("Sentiment")
    {
        _model = model;
    }

    [MessageHandler]
    public ValueTask<string> HandleAsync(string text, IWorkflowContext context)
    {
        // Convert text to feature vector and run the model
        var inputVec = TextToFeature(text);
        var input = inputVec.ToReverseGradTensor();
        var output = _model.Forward(input); // [1, numClasses]
        var probs = ReverseGradOperations.Softmax(output);

        // Convert probabilities to class string
        int maxIdx = ArgMax(probs);
        string[] classes = { "negative", "neutral", "positive" };
        return ValueTask.FromResult(classes[maxIdx]);
    }

    private ReverseGradTensor<float> TextToFeature(string text) { ... }
    private static int ArgMax(ReverseGradTensor<float> tensor) { ... }
}
```

### Example workflow definition

Mixing Nivara executors and an LLM ChatClientAgent:

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

// Load trained Nivara models
var sentimentModel = ModelSerializer.LoadModel("sentiment_model.json");
var entityModel = ModelSerializer.LoadModel("entity_model.json");
var validatorModel = ModelSerializer.LoadModel("validator_model.json");

// Define the LLM agent
var llmClient = new OpenAIClient("API_KEY").GetChatClient("gpt-4o-mini").AsIChatClient();
var llmAgent = new ChatClientAgent(llmClient, "You are a helpful assistant.");

// Define the Nivara executors
var sentimentExecutor = new NivaraSentimentExecutor(sentimentModel);
var entityExecutor = new NivaraEntityExtractor(entityModel);
var validatorExecutor = new NivaraValidator(validatorModel);

// Build the workflow graph
var workflow = new WorkflowBuilder(sentimentExecutor)
    .AddEdge(sentimentExecutor, entityExecutor)
    .AddEdge(entityExecutor, llmAgent)
    .AddEdge(llmAgent, validatorExecutor)
    .Build();

// Run the workflow
var run = await InProcessExecution.RunAsync(workflow, "User input text");
```

### Example project structure

```
samples/NivaraWorkflow/
├── Program.cs                         # Entry point, DI, workflow setup
├── NivaraSentimentExecutor.cs         # Sentiment classification executor
├── NivaraEntityExtractor.cs           # Entity extraction executor
├── NivaraValidator.cs                 # LLM output validator executor
├── Training/
│   ├── TrainSentimentModel.cs        # Training pipeline for sentiment model
│   ├── TrainEntityModel.cs           # Training pipeline for entity model
│   └── TrainValidatorModel.cs        # Training pipeline for validator model
├── Data/
│   ├── synthetic_sentiment.csv       # Training data (or synthetic generation)
│   ├── synthetic_entities.csv
│   └── synthetic_validator.csv
├── Models/                           # Trained model files (gitignored)
│   ├── sentiment_model.json
│   ├── entity_model.json
│   └── validator_model.json
├── TextTokenizer.cs                  # Word/subword tokenizer
└── NivaraWorkflow.csproj
```

### NivaraWorkflow.csproj

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.Agents.AI" Version="1.13.0" />
    <PackageReference Include="Microsoft.Agents.AI.Workflows" Version="1.13.0" />
    <PackageReference Include="Microsoft.Extensions.AI" Version="10.7.0" />
    <PackageReference Include="Microsoft.Extensions.AI.Abstractions" Version="10.7.0" />
    <!-- Optional: pick one LLM provider -->
    <!-- <PackageReference Include="Microsoft.Extensions.AI.OpenAI" Version="10.7.0" /> -->
    <!-- <PackageReference Include="Microsoft.Extensions.AI.Ollama" Version="10.7.0" /> -->
    <ProjectReference Include="..\..\src\Nivara\Nivara.csproj" />
  </ItemGroup>

</Project>
```

### Differences from previous design

The prior version of this document (before July 2026) described a standalone IChatClient implementation backed by a full batched transformer trained on TinyShakespeare. That direction was replaced:

| Prior Aspect | Current Decision | Rationale |
|---|---|---|
| Batched transformer from scratch | Reuse MicroGpt's transformer, build on top | MicroGpt already demonstrates a working transformer; redundant to build another |
| TinyShakespeare language model | Domain-specific classifier/scorer/validator | Better showcases Nivara's deterministic value — not an LLM competitor |
| IChatClient as primary showcase | Custom workflow Executor as primary | Agent Framework's WorkflowBuilder is the more natural .NET pattern |
| Single monolithic example | Training pipeline + inference workflow | Separates concerns; both are valuable to demonstrate |

### See also

- Research document: `docs/AGENT-FRAMEWORK-RESEARCH.md`
- Cross-framework parity examples: `samples/README.md`
- MicroGpt: `samples/MicroGpt/`
