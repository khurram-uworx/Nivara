# NivaraChat — Agent Mode Implementation Plan

## Goal

Add a new `--agents` mode to NivaraChat that wraps the three existing
Nivara models as `IChatClient`s via a thin `ITextModel` abstraction,
then creates `ChatClientAgent`s from them. This demonstrates Nivara
models as first-class Agent Framework participants alongside LLMs,
using the standard `IChatClient` → `AsAIAgent()` pipeline.

The existing `--train`, `--workflow`, and `--interactive` modes remain
unchanged. `--agents` is additive.

## What changes vs. existing modes

| Aspect | `--workflow` (current) | `--agents` (new) |
|--------|----------------------|------------------|
| Executor type | `Executor<string, string>` | `ChatClientAgent` via `IChatClient` |
| Model wrapping | Direct call in executor | `ITextModel` → `NivaraChatClient : IChatClient` |
| Agent creation | Manual executor classes | `IChatClient.AsAIAgent()` |
| Confidence | Not exposed | Emitted in response text with metadata |
| LLM integration | `LlmExecutor` (custom executor) | `ChatClientAgent` wrapping Ollama `IChatClient` |
| Ecosystem fit | Nivara-specific | Standard .NET AI stack |

## Architecture

```
--agents mode:

Input text
    │
    v
[SentimentAgent]        NivaraChatClient(SentimentModel)
    │                   text in → "positive (confidence: 0.94)"
    │                   or      → "Unable to determine sentiment (confidence: 0.31)"
    │
    v
[EntityAgent]           NivaraChatClient(EntityModel)
    │                   text in → "Entities: John Smith (person), Acme Corp (org)"
    │                   or      → "Unable to extract entities (confidence: 0.22)"
    │
    v
[ValidatorAgent]        NivaraChatClient(ValidatorModel)
    │                   text in → "Consistent (confidence: 0.91)"
    │                   or      → "Validation uncertain (confidence: 0.45)"
    │
    v  (fan-in collects all agent outputs)
[Optional: LlmAgent]   ChatClientAgent(OllamaApiClient)
                        receives all Nivara results as context
                        produces final natural language response
```

Each agent is a `ChatClientAgent` created from a `NivaraChatClient`
which wraps an `ITextModel`. The LLM is also a `ChatClientAgent`
created from Ollama's `IChatClient`. All four agents participate
in the same workflow graph through the same API.

### Confidence signaling (no conditional edges)

Each `ITextModel` returns structured text that includes confidence.
When confidence is below threshold, the text explicitly says so:

```
High confidence:  "positive (confidence: 0.94)"
Low confidence:   "Unable to determine sentiment (confidence: 0.31)"
```

The LLM agent receives all outputs and is intelligent enough to:
- Trust high-confidence Nivara results
- Re-analyze or qualify low-confidence results
- Synthesize a final response incorporating all signals

This keeps the workflow graph simple (linear pipeline or fan-out/fan-in)
while still demonstrating the hybrid ML + LLM pattern.

## New files

```
samples/NivaraChat/
├── ITextModel.cs                    # Interface: text in, text out + confidence
├── SentimentTextModel.cs            # ITextModel wrapping TextClassifierModel
├── EntityTextModel.cs               # ITextModel wrapping TokenClassifierModel
├── ValidatorTextModel.cs            # ITextModel wrapping TextClassifierModel (2-class)
├── NivaraChatClient.cs              # IChatClient wrapping ITextModel
├── NivaraChat.csproj                # (modified) add Microsoft.Extensions.AI.Abstractions
└── ...existing files unchanged...
```

## Detailed design

### ITextModel

```csharp
// samples/NivaraChat/ITextModel.cs
namespace NivaraChat;

/// <summary>
/// A text-in, text-out model with a confidence score.
/// Implemented by Nivara-trained classifiers and extractors.
/// </summary>
public interface ITextModel
{
    string Name { get; }
    float Confidence { get; }
    string Process(string input);
}
```

Three properties:
- `Name` — identifies the model (used in agent naming and output labeling)
- `Confidence` — score from the last `Process()` call (0.0–1.0)
- `Process(string)` — the core operation: takes text, returns text

### SentimentTextModel

```csharp
// samples/NivaraChat/SentimentTextModel.cs
namespace NivaraChat;

internal sealed class SentimentTextModel : ITextModel
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    private static readonly string[] Classes = ["negative", "neutral", "positive"];
    private const float ConfidenceThreshold = 0.6f;

    public string Name => "Sentiment";
    public float Confidence { get; private set; }

    public SentimentTextModel(TextClassifierModel<float> model, TextTokenizer tokenizer, int maxSeqLen = 20)
    {
        _model = model;
        _model.Eval();
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
    }

    public string Process(string input)
    {
        var tokens = _tokenizer.Encode(input, fixedLength: _maxSeqLen);
        var data = new float[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
            data[i] = tokens[i];

        var tensor = ReverseGradTensor<float>.FromMatrix(data, 1, _maxSeqLen, requiresGrad: false);
        var logits = _model.Forward(tensor);

        // Softmax for confidence
        float maxLogit = float.MinValue;
        for (int c = 0; c < 3; c++)
            if (logits.Data[c] > maxLogit) maxLogit = logits.Data[c];

        float sumExp = 0;
        var probs = new float[3];
        for (int c = 0; c < 3; c++)
        {
            probs[c] = MathF.Exp(logits.Data[c] - maxLogit);
            sumExp += probs[c];
        }
        for (int c = 0; c < 3; c++)
            probs[c] /= sumExp;

        int bestClass = 0;
        for (int c = 1; c < 3; c++)
            if (probs[c] > probs[bestClass]) bestClass = c;

        Confidence = probs[bestClass];

        if (Confidence < ConfidenceThreshold)
            return $"Unable to determine sentiment (confidence: {Confidence:F2})";

        return $"{Classes[bestClass]} (confidence: {Confidence:F2})";
    }
}
```

### EntityTextModel

Same pattern but wraps `TokenClassifierModel<float>`:

```csharp
internal sealed class EntityTextModel : ITextModel
{
    // ... token classification logic from EntityExtractor ...
    // Aggregates per-token predictions into entity groups
    // Returns: "Entities: John Smith (person), Acme Corp (org)"
    // Or:      "Unable to extract entities (confidence: 0.22)"
    // Confidence = average max-probability across token positions
}
```

### ValidatorTextModel

```csharp
internal sealed class ValidatorTextModel : ITextModel
{
    // Wraps 2-class TextClassifierModel (consistent/inconsistent)
    // Input: "original || response" format
    // Returns: "Consistent (confidence: 0.91)" or "Validation uncertain (confidence: 0.45)"
}
```

### NivaraChatClient : IChatClient

```csharp
// samples/NivaraChat/NivaraChatClient.cs
using Microsoft.Extensions.AI;
using System.Runtime.CompilerServices;

namespace NivaraChat;

/// <summary>
/// IChatClient implementation backed by a Nivara-trained ITextModel.
/// Text in → text out, with confidence metadata.
/// Compatible with the full .NET AI ecosystem (DI, middleware, Agent Framework).
/// </summary>
internal sealed class NivaraChatClient : IChatClient
{
    private readonly ITextModel _model;

    public NivaraChatClient(ITextModel model)
    {
        _model = model;
    }

    public ChatClientMetadata Metadata { get; } = new(
        "NivaraChatClient",
        new Uri("urn:nivara:local"),
        "nivara-text-model-v1");

    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // Extract the last user message — Nivara models are stateless,
        // they don't need conversation history
        var lastUserMsg = messages.LastOrDefault(m => m.Role == ChatRole.User);
        var text = lastUserMsg?.Text ?? "";

        var result = _model.Process(text);

        var response = new ChatResponse(new ChatMessage(ChatRole.Assistant, result));

        // Attach confidence as metadata for downstream consumption
        response.Extensions["model"] = _model.Name;
        response.Extensions["confidence"] = _model.Confidence;

        return Task.FromResult(response);
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Nivara models are synchronous — yield the full result as one chunk
        var response = await GetResponseAsync(messages, options, cancellationToken);
        yield return new ChatResponseUpdate(ChatRole.Assistant, response.Text);
    }

    public object? GetService(Type serviceType, object? serviceKey) =>
        serviceKey is not null ? null
        : serviceType == typeof(ChatClientMetadata) ? Metadata
        : serviceType?.IsInstanceOfType(this) is true ? this
        : null;

    public TService? GetService<TService>(object? key = null) where TService : class =>
        this as TService;

    public void Dispose() { }
}
```

Key design decisions:
- **Ignores conversation history** — extracts only the last user message.
  This is correct: a sentiment classifier doesn't need chat history.
- **Returns structured text** — confidence is embedded in the response text
  so the LLM can read and reason about it.
- **Extensions metadata** — confidence score available programmatically
  for future use (logging, conditional routing, etc.).

### NivaraChatClient.csproj changes

```xml
<!-- Add to existing ItemGroup -->
<PackageReference Include="Microsoft.Extensions.AI.Abstractions" Version="10.7.0" />
```

Only the abstractions package — no `Microsoft.Extensions.AI` (no middleware
needed yet), no `Microsoft.Agents.AI` (already referenced via
`Microsoft.Agents.AI.Workflows`).

## Program.cs changes

### New `--agents` mode in switch

```csharp
case "--agents":
    await RunAgents(ollamaUrl, modelName, workflowText, useOllama);
    break;
```

### RunAgents function

```csharp
async Task RunAgents(string ollamaUrl, string modelName, string? singleShotText, bool useOllama)
{
    Console.WriteLine("=== NivaraChat Agent Mode ===\n");

    // 1. Load trained models (same as existing --workflow)
    if (!File.Exists(Path.Combine(ModelsDir, "sentiment_model.json")))
    {
        Console.WriteLine("Models not found. Run with --train first.");
        return;
    }

    var (sentimentModel, sentimentTok) = LoadSentimentModel();
    var (entityModel, entityTok) = LoadEntityModel();
    var (validatorModel, validatorTok) = LoadValidatorModel();

    // 2. Wrap in ITextModel implementations
    ITextModel sentimentText = new SentimentTextModel(sentimentModel, sentimentTok);
    ITextModel entityText = new EntityTextModel(entityModel, entityTok);
    ITextModel validatorText = new ValidatorTextModel(validatorModel, validatorTok);

    // 3. Create IChatClient wrappers
    var sentimentClient = new NivaraChatClient(sentimentText);
    var entityClient = new NivaraChatClient(entityText);
    var validatorClient = new NivaraChatClient(validatorText);

    // 4. Create ChatClientAgents via AsAIAgent()
    var sentimentAgent = sentimentClient.AsAIAgent(
        name: "SentimentAnalyzer",
        instructions: "You are a sentiment analysis model. Analyze the sentiment of the input text.");

    var entityAgent = entityClient.AsAIAgent(
        name: "EntityExtractor",
        instructions: "You are a named entity recognition model. Extract person, organization, date, and location entities from the input text.");

    var validatorAgent = validatorClient.AsAIAgent(
        name: "OutputValidator",
        instructions: "You are a consistency validator. Check if the provided information is internally consistent.");

    // 5. Optional: LLM agent
    AIAgent? llmAgent = null;
    if (useOllama)
    {
        var ollamaClient = new OllamaApiClient(new Uri(ollamaUrl), modelName);
        llmAgent = ollamaClient.AsAIAgent(
            name: "Assistant",
            instructions: "You are a helpful assistant. Analyze the results from the ML models " +
                         "and provide a clear, helpful response to the user's input.");
    }

    // 6. Build workflow
    //    Option A: Sequential pipeline
    //      Sentiment → Entity → Validator → (optional) LLM
    //
    //    Option B: Fan-out/fan-in (like existing --workflow)
    //      Router → [Sentiment, Entity] → Validator → (optional) LLM
    //
    //    Start with Option A (simpler), demonstrate both later.

    ExecutorBinding sentiment = sentimentAgent;
    ExecutorBinding entity = entityAgent;
    ExecutorBinding validator = validatorAgent;

    Workflow workflow;
    if (llmAgent != null)
    {
        ExecutorBinding llm = llmAgent;
        workflow = new WorkflowBuilder(sentiment)
            .AddEdge(sentiment, entity)
            .AddEdge(entity, validator)
            .AddEdge(validator, llm)
            .WithOutputFrom(sentiment, entity, validator, llm)
            .Build();
        Console.WriteLine("Graph: Sentiment → Entity → Validator → LLM\n");
    }
    else
    {
        workflow = new WorkflowBuilder(sentiment)
            .AddEdge(sentiment, entity)
            .AddEdge(entity, validator)
            .WithOutputFrom(sentiment, entity, validator)
            .Build();
        Console.WriteLine("Graph: Sentiment → Entity → Validator\n");
    }

    // 7. Run
    if (singleShotText != null)
    {
        var run = await InProcessExecution.RunAsync(workflow, singleShotText);
        PrintAgentResults(run);
    }
    else
    {
        Console.WriteLine("Type a message (or 'quit' to exit):\n");
        while (true)
        {
            Console.Write("> ");
            var input = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(input) || input == "quit") break;

            var run = await InProcessExecution.RunAsync(workflow, input);
            PrintAgentResults(run);
            Console.WriteLine();
        }
    }

    // Cleanup
    sentimentModel.Dispose();
    entityModel.Dispose();
    validatorModel.Dispose();
}

void PrintAgentResults(WorkflowRun run)
{
    Console.WriteLine("\n--- Agent Results ---");
    foreach (var evt in run.NewEvents)
    {
        switch (evt)
        {
            case ExecutorCompletedEvent executorEvt:
                if (executorEvt.Data?.ToString() is string data && !string.IsNullOrEmpty(data))
                    Console.WriteLine($"  [{executorEvt.ExecutorId}] {data}");
                break;
            case AgentResponseEvent agentEvt:
                Console.WriteLine($"  [Agent] {agentEvt.Data}");
                break;
        }
    }
}
```

### LoadValidatorModel helper

```csharp
(TextClassifierModel<float> model, TextTokenizer tokenizer) LoadValidatorModel()
{
    var tokenizer = TextTokenizer.Load(Path.Combine(ModelsDir, "validator_tokenizer.json"));
    var model = new TextClassifierModel<float>(tokenizer.VocabSize, 32, 64, 2, 40);
    ModelSerializer.Load(model, Path.Combine(ModelsDir, "validator_model.json"));
    model.Eval();
    return (model, tokenizer);
}
```

### Updated PrintUsage

```csharp
void PrintUsage()
{
    Console.WriteLine("Usage: NivaraChat <mode> [options]\n");
    Console.WriteLine("Modes:");
    Console.WriteLine("  --train              Train sentiment, entity, and validator models");
    Console.WriteLine("  --workflow           Run the Executor-based workflow (Ollama optional)");
    Console.WriteLine("  --agents             Run as ChatClientAgents via IChatClient (new)");
    Console.WriteLine("  --interactive        Interactive mode: local inference + Ollama chat");
    // ...options unchanged...
}
```

## What this demonstrates

| Concept | How it's shown |
|---------|---------------|
| `ITextModel` abstraction | Three Nivara models behind a uniform text-in/text-out interface |
| `NivaraChatClient : IChatClient` | Standard .NET AI abstraction wrapping deterministic ML |
| `IChatClient.AsAIAgent()` | Each Nivara model becomes a `ChatClientAgent` |
| `ChatClientAgent` | Agent Framework agent backed by local model, not LLM |
| Fan-out / fan-in workflow | Sentiment + Entity run in parallel via `WorkflowBuilder` |
| Confidence signaling | Models emit structured text with confidence scores |
| LLM + ML hybrid | Optional LLM agent receives Nivara outputs and synthesizes |
| Ecosystem compatibility | Same `NivaraChatClient` works with DI, middleware, etc. |
| Zero library changes | Uses only existing Nivara AutoDiff APIs |

## Implementation order

1. Create `ITextModel.cs` — the interface
2. Create `SentimentTextModel.cs` — wrap existing sentiment model
3. Create `EntityTextModel.cs` — wrap existing entity model
4. Create `ValidatorTextModel.cs` — wrap existing validator model
5. Create `NivaraChatClient.cs` — `IChatClient` wrapper
6. Update `NivaraChat.csproj` — add `Microsoft.Extensions.AI.Abstractions`
7. Update `Program.cs` — add `--agents` mode, `RunAgents()`, `LoadValidatorModel()`
8. Test with `--train && --agents --text "John Smith from Acme Corp reported great work"`
9. Test with `--agents --ollama` to verify LLM agent integration
10. Test interactive mode: `--agents`

## Future extensions (not in this plan)

- **Confidence-based conditional routing** — use `AddEdge(from, to, condition: ...)` to skip the LLM when all Nivara agents are confident
- **Nivara as `IEmbeddingGenerator`** — plug into vector stores (see NivaraChat-NEXT.md #F)
- **Nivara as AIFunction tools** — LLM calls Nivara models on demand (see NivaraChat-NEXT.md #C)
- **RAG with Nivara embeddings** — local vector search (see NivaraChat-NEXT.md #D)
- **Writer-critic loop** — Nivara validates LLM output iteratively (see NivaraChat-NEXT.md #E)
- **`--agents` with fan-out** — parallel Nivara agents + barrier before LLM
- **Multiple workflow topologies** — sequential vs. fan-out vs. handoff, selectable via CLI flag
