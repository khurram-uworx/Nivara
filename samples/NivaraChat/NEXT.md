# NivaraChat-NEXT — Roadmap for Hybrid AI/ML Showcase

NivaraChat currently demonstrates fan-out/fan-in with three trained models
plus an optional Ollama LLM. The ideas below extend it into a comprehensive
showcase of mixing deterministic ML with stochastic LLMs, exercising both
Nivara's AutoDiff domain and the .NET AI ecosystem (Microsoft.Extensions.AI,
Agent Framework Workflows).

Each section is a self-contained feature that can be implemented independently
or composed together. They are ordered roughly by dependency (earlier ideas
don't require later ones).

| # | Idea | Key Ecosystem APIs | Nivara Gaps |
|---|------|--------------------|-------------|
| A | IChatClient backed by batched transformer | `IChatClient`, DI | LayerNorm, batched Embedding, Concat |
| B | Confidence-based handoff | Conditional edges | Score extraction from models |
| C | Nivara as AIFunction tools | `AIFunctionFactory`, tool calling | None (existing models) |
| D | RAG with Nivara embeddings | `IEmbeddingGenerator`, vector search | Embedding generator wrapper |
| E | Writer-critic feedback loop | Conditional edges, feedback | None (existing validator) |
| F | Nivara as IEmbeddingGenerator | `IEmbeddingGenerator<T>` | Embedding wrapper module |
| G | Intent classification router | Handoff orchestration | Intent classifier model |
| H | Online learning from LLM feedback | `DataLoader` incremental | Incremental training API |

---

# A. IChatClient Implementation Backed by a Batched Transformer

## Goal

Implement the standard .NET `Microsoft.Extensions.AI.IChatClient` interface
backed by a Nivara-trained transformer model. The example trains a proper
batched causal transformer on TinyShakespeare (or similar small text
corpus), then wraps the trained model in an `IChatClient` that can be
wiredup via dependency injection.

MicroGpt proves Nivara can train a transformer; NivaraChatClient proves it
can serve one in ecosystem-compatible way.

```
TinyShakespeare.txt  (or similar text corpus)
    │
    v
Word-level tokenizer (TextTokenizer)
    │
    v
Batched causal transformer training  ←── Gaps to fix: LayerNorm,
    │                                       batched embedding,
    │                                       proper attention, Concat
    v
ModelSerializer.Save / Load
    │
    v
NivaraChatClient : IChatClient           ←── Implements Microsoft.Extensions.AI
    ├── GetResponseAsync(...)                 standard interface
    ├── GetStreamingResponseAsync(...)
    ├── GetService(...)
    └── IDisposable
    │
    v
DI-compatible: services.AddChatClient<NivaraChatClient>()
    │
    v
Console app, ASP.NET, or MAUI
```

## Architecture

### What changes from MicroGpt

MicroGpt does per-position forward (one token at a time, KV cache). This
example replaces that with a **proper batched causal transformer**:

| Aspect | MicroGpt | NivaraChatClient |
|---|---|---|
| Forward pass | Per-position, KV cache | Batched sequence `[B, L]` |
| Attention | Per-head dot product loop | Proper batched multi-head attention |
| Embedding | `Forward(int)` single token | `Forward(ReverseGradTensor<T>)` batch |
| Normalization | RMSNorm (op only) | LayerNorm (new module) |
| Training | Manual grad scope loop | `TrainingLoop<T>` + `DataLoader<T>` |
| Inference | Generate via sampling | `IChatClient` standard API |
| Data | Character-level names | Word-level text corpus |
| Loss | Hand-rolled NLL | `CrossEntropyLoss<T>` |
| Save/Load | None | `ModelSerializer` |

### Batched Causal Transformer Architecture

```
Input tokens: [B, L]  (batch, sequence length)
    │
    v
Embedding: [B, L] → [B, L, D]
    │
    v
Position encoding (sinusoidal, added to embeddings)
    │
    v
N × TransformerBlock:
    │
    ├── LayerNorm → Multi-Head Self-Attention (causal mask) → Residual
    │       │
    │       └── Q, K, V projections: [B, L, D] → [B, L, D] each
    │           Reshape to [B, H, L, D/H]
    │           Scores: [B, H, L, D/H] @ [B, H, D/H, L] → [B, H, L, L]
    │           Causal mask + Softmax
    │           Context: [B, H, L, L] @ [B, H, L, D/H] → [B, H, L, D/H]
    │           Reshape → [B, L, D] → Output projection
    │
    └── LayerNorm → MLP (expand 4×, ReLU, compress) → Residual
    │
    v
LayerNorm → LM Head: [B, L, D] → [B, L, V]  (logits)
    │
    v
CrossEntropyLoss (training) or Sampled tokens (inference)
```

**Key simplification for Nivara's current op set:** Since Nivara's MatMul
only supports rank-2, the batched attention is implemented by flattening
the batch×head dimensions into a single matrix dimension:

```
Q, K, V: [B, L, D] → reshape to [B*L, D] for Linear → reshape back
Attention scores: flatten [B, H, L, D/H] → [B*H*L, D/H] then [B*H, L, D/H]
  scores = MatMul([B*H, L, D/H], [B*H, D/H, L]^T) → [B*H, L, L]
  context = MatMul(softmax(scores), [B*H, L, D/H]) → [B*H, L, D/H]
  reshape back to [B, L, D]
```

This is a workaround. The real fix would be batch MatMul support, but
the workaround is feasible and instructive.

### NivaraChatClient : IChatClient

```csharp
public sealed class NivaraChatClient : IChatClient
{
    private readonly BatchedTransformer<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;
    private readonly float _temperature;

    public NivaraChatClient(
        BatchedTransformer<float> model,
        TextTokenizer tokenizer,
        int maxSeqLen = 256,
        float temperature = 0.8f)
    {
        _model = model;
        _tokenizer = tokenizer;
        _maxSeqLen = maxSeqLen;
        _temperature = temperature;
    }

    public ChatClientMetadata Metadata { get; } = new(
        "NivaraChatClient",
        new Uri("urn:nivara:local"),
        "nivara-transformer-v1");

    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        // 1. Format conversation into prompt string
        var prompt = FormatConversation(messages);

        // 2. Tokenize
        var inputIds = _tokenizer.Encode(prompt, fixedLength: _maxSeqLen);

        // 3. Run inference — Generate tokens autoregressively
        var outputIds = Generate(inputIds, cancellationToken);

        // 4. Detokenize
        var text = _tokenizer.Decode(outputIds);

        // 5. Return as ChatResponse
        return new ChatResponse(new ChatMessage(ChatRole.Assistant, text));
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var prompt = FormatConversation(messages);
        var inputIds = _tokenizer.Encode(prompt, fixedLength: _maxSeqLen);

        // Token-by-token generation, yielding each token as it's produced
        foreach (var token in GenerateStreaming(inputIds, cancellationToken))
        {
            var text = _tokenizer.Decode([token]);
            yield return new ChatResponseUpdate(ChatRole.Assistant, text);
        }
    }

    public object? GetService(Type serviceType, object? serviceKey) =>
        serviceKey is not null ? null
        : serviceType == typeof(ChatClientMetadata) ? Metadata
        : serviceType?.IsInstanceOfType(this) is true ? this
        : null;

    public TService? GetService<TService>(object? key = null) where TService : class =>
        this as TService;

    public void Dispose() => _model.Dispose();

    // Internal generation logic
    private int[] Generate(int[] inputIds, CancellationToken ct) { ... }
    private IEnumerable<int> GenerateStreaming(int[] inputIds, CancellationToken ct) { ... }
    private static string FormatConversation(IEnumerable<ChatMessage> messages) { ... }
}
```

## Gaps This Will Discover and Fix

### Gap 1: LayerNorm module does not exist

**Problem:** `RMSNorm` exists as an op but normalizes over the entire
flattened tensor, not per-feature. Transformers need LayerNorm that
normalizes over the last dimension (embedding dimension).

**Fix:** Add `LayerNorm<T>` module in `src/Nivara/AutoDiff/Nn/`:
```csharp
public sealed class LayerNorm<T> : Module<T> where T : struct, INumber<T>
{
    public int NormalizedShape { get; }
    public ReverseGradTensor<T> Gamma { get; }
    public ReverseGradTensor<T>? Beta { get; }

    public LayerNorm(int normalizedShape, double eps = 1e-5, bool bias = true);

    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input);
}
```

And the corresponding low-level op `ReverseGradOperations.LayerNorm<T>`
with the full backward pass.

### Gap 2: Batched embedding lookup

**Problem:** `Embedding<T>.Forward(int)` only supports single-token lookup.

**Fix:** Add `Forward(ReverseGradTensor<T> tokenIds)` that takes
`[batchSize, seqLen]` integer tokens, flattens to `[B*L]`, builds a
one-hot matrix `[B*L, V]`, MatMuls with weight `[V, D]` to get
`[B*L, D]`, and reshapes to `[B, L, D]`.

Also make `Embedding<T>` extend `Module<T>` so it works with
`TrainingLoop<T>`, `ModelSerializer`, etc.

### Gap 3: No Concat operation

**Problem:** Transformer implementation may need `Concat` for residual
connections or head merging. MicroGpt uses `PadRight`/`PadLeft` with
selection matrices as a workaround, which is O(n²).

**Fix:** Add `ReverseGradOperations.Concat<T>(tensors, dim)` that
supports concatenating along a specified dimension with proper gradient
split in the backward pass.

### Gap 4: ~~No Gather operation~~ **PARTIALLY RESOLVED**

Batched embedding lookup (`Embedding<T>.Forward(ReverseGradTensor<T>)`) is implemented
via batched one-hot + single MatMul. For token selection during autoregressive generation
(selecting next-token logits from `[B, L, V]`), a formal `Gather` op is still needed.
The one-hot fallback is acceptable for small-to-medium vocabularies.

### Gap 5: No LayerNorm op (see Gap 1)

Coupled with Gap 1 — needs both the op and the module.

### Gap 6: Softmax with explicit dim

**Problem:** `Softmax<T>` functional module ignores the `dim` parameter.
For attention, softmax must apply over the last dimension (L), not the
entire tensor.

**Fix:** Make `ReverseGradOperations.Softmax<T>` accept a `dim`
parameter. For rank-3 tensors `[B, L, D]`, `dim=2` softmaxes each
feature vector of length D independently.

### Gap 7: ~~Embedding<T> not a Module<T>~~ **RESOLVED**

`Embedding<T>` now extends `Module<T>` (see NivaraChatClient.md Gap 1).

### Gap 8: Position encoding

Sinusoidal position encoding is not a module. We can implement it as a
non-trainable helper:
```csharp
public static class PositionEncoding
{
    public static ReverseGradTensor<T> Sinusoidal<T>(int seqLen, int embedDim)
        where T : struct, INumber<T>;
}
```

No backward pass needed — it's a constant added to the embedding.

### Gap 9: IChatClient thread safety

`IChatClient` requires thread-safe concurrent use. Nivara's model
forward pass is currently single-threaded. We need to ensure:
- Inference (non-training) forward passes are re-entrant
- No shared mutable state during generation
- Each chat session gets its own KV cache

This is primarily a design concern for the chat client wrapper, not a
library change.

## Files

```
samples/NivaraChatClient/
├── Program.cs                        # Entry point, CLI, DI setup
├── BatchedTransformer.cs             # Batched causal transformer model (Module<T>)
├── MultiHeadAttention.cs             # Multi-head attention (composed from ops)
├── LayerNorm.cs                      # LayerNorm module (NEW — to be promoted to core)
├── PositionEncoding.cs               # Sinusoidal position encoding
├── NivaraChatClient.cs               # IChatClient implementation
├── TextTokenizer.cs                  # Word-level tokenizer (reusable)
├── TinyShakespeareDownloader.cs      # Downloads TinyShakespeare (or generates text)
└── NivaraChatClient.csproj           # Project file referencing Nivara + Microsoft.Extensions.AI.Abstractions
```

### NivaraChatClient.csproj

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.Extensions.AI.Abstractions" Version="10.7.0" />
    <PackageReference Include="Microsoft.Extensions.DependencyInjection" Version="10.*" />
    <PackageReference Include="Microsoft.Extensions.Hosting" Version="10.*" />
    <ProjectReference Include="..\..\src\Nivara\Nivara.csproj" />
  </ItemGroup>

</Project>
```

## CLI Interface

```
--train              Train the transformer on TinyShakespeare
--load <path>        Load a trained model
--save <path>        Save trained model
--prompt <text>      Run inference on a prompt (non-interactive)
--interactive        Interactive chat REPL
--epochs <int>       Training epochs (default: 5)
--batch-size <int>   Batch size (default: 16)
--seq-len <int>      Sequence length (default: 128)
--n-embd <int>       Embedding dimension (default: 64)
--n-layer <int>      Number of transformer layers (default: 4)
--n-head <int>       Number of attention heads (default: 4)
--lr <float>         Learning rate (default: 0.001)
--temperature <float> Sampling temperature (default: 0.8)
--seed <int>         RNG seed (default: 42)
--help, -h           Show this help
```

### Modes

**Default:** Train (if no model loaded), then start interactive chat.

**`--train`:** Download/load TinyShakespeare, train for `--epochs` epochs,
save model.

**`--load <path>`:** Load saved model and tokenizer. Skip training.

**`--interactive`:** REPL with conversation history. Type `quit` to exit.

**`--prompt "..."`:** Single-turn inference, print response, exit.

### DI wiring (program showcase)

```csharp
var builder = Host.CreateApplicationBuilder(args);

builder.Services.AddChatClient(services =>
{
    var model = LoadModel(args);
    var tokenizer = LoadTokenizer(args);
    return new NivaraChatClient(model, tokenizer);
})
.UseDistributedCache()
.UseOpenTelemetry();

var host = builder.Build();
var chatClient = host.Services.GetRequiredService<IChatClient>();
```

## What This Exercises vs. MicroGpt

| Feature | MicroGpt | NivaraChatClient |
|---|---|---|
| **Transformer forward** | Per-position, KV cache | Batched causal `[B, L]` |
| **Multi-head attention** | Per-head loops | Proper batched MHA |
| **LayerNorm** | No (RMSNorm only) | Yes (new module) |
| **Embedding batch lookup** | No | Yes |
| **Concat/Gather ops** | No (selection matrix hacks) | Yes (new ops) |
| **Training infrastructure** | Manual loop | `TrainingLoop<T>` + `DataLoader<T>` |
| **Loss function** | Hand-rolled NLL | `CrossEntropyLoss<T>` |
| **Optimizer** | Adam | AdamW |
| **Serialization** | None | `ModelSerializer` full round-trip |
| **IChatClient** | No | Yes — ecosystem standard |
| **DI integration** | No | Yes — `AddChatClient<>()` |
| **Streaming responses** | No | Yes — `IAsyncEnumerable` |
| **Conversation history** | No | Yes — multi-turn chat |
| **Position encoding** | Learned `Embedding` | Sinusoidal (fixed) |
| **Data** | Character names | Word-level corpus |
| **Evaluation** | Subjective | Perplexity + sample quality |

## Core Library Changes

After validation, these new modules would be promoted to `src/Nivara/`:

| New API | File | Status |
|---|---|---|
| `LayerNorm<T>` module | `src/Nivara/AutoDiff/Nn/LayerNorm.cs` | **New** |
| `ReverseGradOperations.LayerNorm` op | `src/Nivara/AutoDiff/Operations/` | **New** |
| `ReverseGradOperations.Concat` op | `src/Nivara/AutoDiff/Operations/` | **New** |
| `ReverseGradOperations.Gather` op | `src/Nivara/AutoDiff/Operations/` | **New** |
| `Embedding<T>` as `Module<T>` | `src/Nivara/AutoDiff/Nn/Embedding.cs` | **Refactor** |
| `Embedding<T>.Forward(ReverseGradTensor<T>)` | `src/Nivara/AutoDiff/Nn/Embedding.cs` | **New** |
| `Softmax(dim)` support | `src/Nivara/AutoDiff/Operations/` | **Fix** |
| `AttentionMask` helper | `src/Nivara/Helpers/` | **New** |

## Not Doing (Stretch / Future)

- **KV cache in batched form** — inference uses simple autoregressive
  loop without caching. Implementing batched KV cache is future work.
- **Top-p / top-k sampling** — uses temperature + random sampling only.
- **Beam search** — greedy decoding only.
- **Fine-tuning / LoRA** — full training only.
- **Quantization** — full float32 only.
- **GPU support** — CPU training only, matching Nivara's current scope.
- **Multi-modal (image/audio)** — text-only chat.
- **ASP.NET Core hosting** — console app only, but DI-compatible so
  hosting in ASP.NET is a one-liner for the user.

---

# B. Confidence-Based Handoff Pattern

## Goal

Use Agent Framework conditional edges to route based on Nivara model
confidence. When models are confident, return deterministic results
directly (<1ms). When uncertain, hand off to the LLM for richer reasoning.

This is the core hybrid thesis: **deterministic ML owns the fast-path,
stochastic LLM handles the uncertain tail.**

```
Input text
    │
    v
[SentimentExecutor]         Nivara classifier, returns label + confidence
    │
    v
[EntityExtractor]           Nivara NER, returns entities + confidence
    │
    v
[ConfidenceRouter]          Checks scores from both executors
    │
    ├── confidence >= 0.8  ──> [ReturnNivaraResult]     Fast path (<1ms)
    │
    └── confidence < 0.8  ──> [LLMAgent]                Slow path (stochastic)
                              │  receives Nivara partial results as context
                              │  generates enriched response
                              v
                         [ResponseValidator]             Nivara checks LLM output
                              │
                              v
                         Final structured result
```

Key differences from current NivaraChat:
- Current: fan-in always goes to validator, then optionally to LLM.
- Here: **conditional edge** decides whether LLM is needed at all.
- Demonstrates `AddEdge(from, to, condition: ...)` which NivaraChat
  doesn't currently use.

## Architecture

### Confidence extraction

Each Nivara executor returns a JSON payload with both the result and a
confidence score derived from the model's softmax output:

```csharp
public record ClassificationResult(string Label, float Confidence);

// In SentimentExecutor:
var logits = _model.Forward(input);
var probs = Softmax(logits);            // or just use argmax + max-prob
float confidence = probs.Data.Max();    // highest class probability
string label = probs.Data.argmax() switch
{
    0 => "negative", 1 => "neutral", 2 => "positive"
};
return JsonSerializer.Serialize(new ClassificationResult(label, confidence));
```

### Conditional edge

```csharp
// Build the confidence-gated path
var router = new ConfidenceRouter();         // checks both results
var nivaraFast = new ReturnNivaraResult();   // formats deterministic output
var llmAgent = new LlmExecutor(chatClient); // calls Ollama
var validator = new ResponseValidator();     // checks LLM output

builder.AddEdge(sentiment, router);
builder.AddEdge(entities, router);

// Conditional: high confidence → fast path
builder.AddEdge(router, nivaraFast,
    condition: msg => msg is RouterOutput o && o.HighConfidence);

// Conditional: low confidence → LLM path
builder.AddEdge(router, llmAgent,
    condition: msg => msg is RouterOutput o && !o.HighConfidence);

builder.AddEdge(llmAgent, validator);
builder.AddEdge(nivaraFast, output);   // fast path joins output
builder.AddEdge(validator, output);    // slow path joins output
```

### When the LLM fires

The LLM receives structured context from Nivara, not just raw text:

```
"Nivara analysis: sentiment=neutral (confidence=0.62),
 entities=[{name:'Acme Corp', type:'org'}] (confidence=0.71).
 Please provide a richer analysis given these partial results."
```

This reduces the LLM's workload — it doesn't need to re-extract entities
or re-classify sentiment, only to reason about the uncertain cases.

## What This Exercises

| API / Feature | Purpose |
|---|---|
| `AddEdge(from, to, condition: ...)` | Conditional routing (new pattern for NivaraChat) |
| `Softmax` confidence extraction | Score from existing Nivara models |
| JSON-structured executor messages | Typed message passing through workflow |
| `ClassificationResult` record | Strongly-typed data flowing between executors |
| Ollama `ChatClientAgent` | LLM integration (same as current, but gated) |

## Core Library Changes

None required. Uses existing Nivara modules as-is. This is purely a
workflow architecture demo.

---

# C. Nivara as AIFunction Tools for the LLM

## Goal

Flip the architecture: instead of Nivara feeding results into the LLM
pipeline, the LLM **decides** when to call Nivara models as tools.
The LLM becomes the orchestrator; Nivara models become callable functions.

This demonstrates `AIFunctionFactory`, `FunctionInvokingChatClient`, and
the Agent Framework's tool-calling support — while showing deterministic
ML models as first-class AI tools.

```
User: "Analyze the sentiment and extract entities from
       'John Smith from Acme Corp reported great work'"

LLM sees tools:
  ├── analyze_sentiment(text) → { label, confidence }
  ├── extract_entities(text) → { entities: [...] }
  └── validate_response(original, response) → { consistent, confidence }

LLM decides:
  1. Call analyze_sentiment(...)     → "positive" (0.94)
  2. Call extract_entities(...)      → [{John Smith, person}, {Acme Corp, org}]
  3. Call validate_response(...)     → "consistent" (0.91)
  4. Generate final response using tool results

Response: "The text expresses positive sentiment (94% confidence)
           about John Smith from Acme Corp. Two entities were identified..."
```

Key insight: the LLM doesn't waste tokens re-doing classification or NER.
It calls the deterministic model and gets instant, consistent results.
The LLM focuses on **reasoning and language generation**.

## Architecture

### Wrapping Nivara models as AIFunction

```csharp
using Microsoft.Extensions.AI;

// Wraps TextClassifierModel as a callable tool
[Description("Analyze sentiment of text. Returns positive/negative/neutral with confidence.")]
static string AnalyzeSentiment(
    [Description("The text to analyze")] string text)
{
    var inputIds = _sentimentTokenizer.Encode(text, fixedLength: 20);
    var input = ReverseGradTensor<float>.FromMatrix(
        inputIds.Select(i => (float)i).ToArray(), 1, 20, requiresGrad: false);
    var output = _sentimentModel.Forward(input);
    // ... extract label + confidence, return JSON
    return JsonSerializer.Serialize(new { label = "positive", confidence = 0.94f });
}

[Description("Extract named entities (person, org, date, location) from text.")]
static string ExtractEntities(
    [Description("The text to extract entities from")] string text)
{
    var inputIds = _entityTokenizer.Encode(text, fixedLength: 20);
    // ... run TokenClassifierModel, return JSON
    return JsonSerializer.Serialize(new {
        entities = new[] {
            new { name = "John Smith", type = "person" },
            new { name = "Acme Corp", type = "org" }
        }
    });
}
```

### Agent with Nivara tools

```csharp
var tools = new[] {
    AIFunctionFactory.Create(AnalyzeSentiment),
    AIFunctionFactory.Create(ExtractEntities),
    AIFunctionFactory.Create(ValidateResponse),
};

var agent = ollamaClient.AsAIAgent(
    instructions: "You are an analyst. Use the provided tools to analyze " +
                  "text precisely. Always call tools before generating your response.",
    tools: tools
);

// The LLM will automatically call Nivara tools when it determines they're useful
var response = await agent.RunAsync(
    "Analyze: John Smith from Acme Corp reported great work on January 15");
```

### Workflow variant

Alternatively, wrap tool-calling in an Agent Framework executor:

```csharp
public partial class ToolOrchestratorExecutor : Executor<string>
{
    [MessageHandler]
    public async ValueTask<string> HandleAsync(string input, IWorkflowContext ctx)
    {
        var agent = _chatClient.AsAIAgent(tools: _nivaraTools);
        var result = await agent.RunAsync(
            $"Analyze this text using available tools: {input}");
        return result.Message.Text;
    }
}
```

## What This Exercises

| API / Feature | Purpose |
|---|---|
| `AIFunctionFactory.Create(...)` | Wrapping .NET methods as AI-callable tools |
| `IChatClient.AsAIAgent(...)` | Agent creation with tool definitions |
| `Agent.RunAsync(...)` | LLM invocation with automatic tool calling |
| `FunctionInvokingChatClient` | Auto tool-call execution loop |
| `ChatToolChoice.ForRequired(...)` | Force tool use (optional, for testing) |
| Nivara `TextClassifierModel` | As a deterministic tool, not a workflow node |
| Nivara `TokenClassifierModel` | As a deterministic tool, not a workflow node |

## Core Library Changes

None. Uses existing Nivara models as-is, wrapped via `AIFunctionFactory`.

---

# D. RAG Pipeline with Nivara Embeddings

## Goal

Use Nivara's learned `Embedding<T>` + `CosineSimilarity` + `TopKDescending`
as a local vector search engine. Wire it into the Agent Framework's
`TextSearchProvider` for Retrieval-Augmented Generation without external
vector databases.

This demonstrates Nivara as the **retrieval backbone** — a local,
zero-dependency embedding + search layer.

```
Documents (e.g., README, API docs, FAQ)
    │
    v
TextTokenizer → chunks → Embedding<T> → NivaraFrame (store vectors)
    │
    v
User query
    │
    v
Embedding<T> → query vector
    │
    v
CosineSimilarity(query, all_chunks) → TopKDescending(3)
    │
    v
[TextSearchProvider] injects top-K chunks into LLM context
    │
    v
LLM generates grounded response
```

## Architecture

### Document indexing

```csharp
public class NivaraVectorStore
{
    private readonly Embedding<float> _embedder;
    private readonly TextTokenizer _tokenizer;
    private readonly List<DocumentChunk> _chunks = new();

    public void IndexDocuments(string[] documents, int chunkSize = 100)
    {
        foreach (var doc in documents)
        {
            var textChunks = ChunkText(doc, chunkSize);
            foreach (var chunk in textChunks)
            {
                var tokenIds = _tokenizer.Encode(chunk, fixedLength: chunkSize);
                var embedding = ComputeEmbedding(tokenIds);  // MeanPool of Embedding<T> output
                _chunks.Add(new DocumentChunk(chunk, embedding));
            }
        }
    }

    private float[] ComputeEmbedding(int[] tokenIds)
    {
        var input = ToTensor(tokenIds);
        var output = _embedder.Forward(input);
        return MeanPool(output);  // [1, D] → [D]
    }
}
```

### Vector search

```csharp
public List<DocumentChunk> Search(string query, int topK = 3)
{
    var queryTokens = _tokenizer.Encode(query, fixedLength: 128);
    var queryEmbedding = ComputeEmbedding(queryTokens);

    // Use Nivara's CosineSimilarity + TopKDescending
    // Build a NivaraFrame of all stored embeddings for batch comparison
    var results = new (DocumentChunk chunk, float score)[_chunks.Count];
    for (int i = 0; i < _chunks.Count; i++)
    {
        results[i] = (_chunks[i],
            TensorPrimitives.CosineSimilarity(queryEmbedding, _chunks[i].Embedding));
    }

    return results
        .OrderByDescending(r => r.score)
        .Take(topK)
        .Select(r => r.chunk)
        .ToList();
}
```

### Agent Framework TextSearchProvider integration

```csharp
// Register NivaraVectorStore as the search backend for TextSearchProvider
var vectorStore = new NivaraVectorStore(embedder, tokenizer);
vectorStore.IndexDocuments(ReadDocuments("docs/"));

// TextSearchProvider calls our search function
var textSearchProvider = new TextSearchProvider(
    query => Task.FromResult(vectorStore.Search(query).Select(c => c.Text).ToList()));

// Attach to agent
agentThread.AIContextProviders.Add(textSearchProvider);
```

### Standalone RAG executor (workflow variant)

```csharp
public partial class RagRetrievalExecutor : Executor<string>
{
    [MessageHandler]
    public async ValueTask<string> HandleAsync(string query, IWorkflowContext ctx)
    {
        var chunks = _vectorStore.Search(query, topK: 3);
        var context = string.Join("\n\n", chunks.Select(c => c.Text));
        return $"Retrieved context:\n{context}\n\nQuery: {query}";
    }
}
```

## What This Exercises

| API / Feature | Purpose |
|---|---|
| `Embedding<T>.Forward(int[])` | Token → dense vector |
| `TensorPrimitives.CosineSimilarity` | Vector similarity search |
| `NivaraFrame` | Storage for document embeddings |
| `TextSearchProvider` | Agent Framework RAG integration |
| `IEmbeddingGenerator` (optional) | Standard embedding interface (see #F) |

## Core Library Changes

| New API | File | Purpose |
|---|---|---|
| `NivaraEmbeddingGenerator<T>` | `src/Nivara/AutoDiff/Nn/` | `IEmbeddingGenerator` wrapper (see #F) |

---

# E. Writer-Critic Feedback Loop

## Goal

Use the Agent Framework's conditional edges to create an iterative
refinement loop: an LLM generates a response, a Nivara model validates
it, and if the quality is below threshold, the LLM is re-prompted
with feedback to try again.

This demonstrates feedback loops, bounded iteration, and the
complementary roles of stochastic generation + deterministic evaluation.

```
User input
    │
    v
[LLM Writer]                 Generates initial response
    │
    v
[Nivara Critic]              Scores response quality (trained classifier)
    │
    ├── score >= 0.8  ──> [Return Result]           Done (1 iteration)
    │
    └── score < 0.8   ──> [LLM Writer]              Re-prompt with feedback
              ↑                  │                     "Previous response scored
              │                  v                      0.4. Please improve:
              └──────────────────┘                      clarity, accuracy, structure"
                                          (max 3 iterations)
```

## Architecture

### Critic model

Reuse the existing `TextClassifierModel<float>` (validator) trained on:
- Input: `"original || response"` (user query + generated response)
- Output: 2 classes (consistent/helpful vs. inconsistent/unhelpful)
- Can be enhanced with a multi-class quality score (poor/fair/good/excellent)

### Bounded loop executor

```csharp
public partial class IterativeWriterExecutor : Executor<WriterInput>
{
    private const int MaxIterations = 3;
    private const float QualityThreshold = 0.8f;

    [MessageHandler]
    public async ValueTask<string> HandleAsync(WriterInput input, IWorkflowContext ctx)
    {
        string? response = null;
        string feedback = "";

        for (int i = 0; i < MaxIterations; i++)
        {
            // Generate
            response = await _chatClient.GetResponseAsync(
                BuildPrompt(input.Query, feedback));

            // Critique
            var score = Evaluate(input.Query, response);

            if (score >= QualityThreshold)
                return response;

            feedback = $"Previous attempt scored {score:F2}. " +
                       $"Improve on: clarity, accuracy, relevance to the query.";
        }

        return response!; // Return best attempt after max iterations
    }

    private float Evaluate(string query, string response)
    {
        var input = _criticTokenizer.Encode($"{query} || {response}", fixedLength: 40);
        var tensor = ToTensor(input);
        var output = _criticModel.Forward(tensor);
        return GetClassProbability(output, classIndex: 1); // probability of "consistent"
    }
}
```

### Workflow with feedback loop

```csharp
var writer = new IterativeWriterExecutor(chatClient, criticModel);
var output = new OutputExecutor();

// Single pass (no loop) — or use conditional edges for loop
builder.AddEdge(writer, output);

// For true loop topology, the writer re-invokes itself internally
// (bounded loop inside the executor is simpler than workflow-level loops)
```

## What This Exercises

| API / Feature | Purpose |
|---|---|
| Nivara `TextClassifierModel` | Quality scoring of LLM output |
| `IChatClient.GetResponseAsync` | LLM generation |
| Conditional iteration | Bounded retry loop |
| Structured feedback | Passing scoring context back to LLM |

## Core Library Changes

None. Uses existing Nivara models and Agent Framework patterns.

---

# F. Nivara as Local IEmbeddingGenerator

## Goal

Implement `IEmbeddingGenerator<string, Embedding<float>>` backed by a
trained Nivara `Embedding<T>` layer + `MeanPool`. This plugs into the
entire .NET AI ecosystem: vector stores, `TextSearchProvider`, RAG
pipelines, and any code that accepts `IEmbeddingGenerator`.

Simpler than the full IChatClient (#A) but still a powerful ecosystem
integration point.

```csharp
public sealed class NivaraEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly Module<float> _model;     // Embedding + optional projection
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxLen;
    private readonly int _dimensions;

    public EmbeddingGeneratorMetadata Metadata { get; } = new(
        "NivaraEmbeddingGenerator",
        new Uri("urn:nivara:local"),
        "nivara-embedding-v1");

    public async Task<IList<Embedding<float>>> GenerateAsync(
        IEnumerable<string> values,
        EmbeddingGenerationOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var results = new List<Embedding<float>>();
        foreach (var text in values)
        {
            var tokens = _tokenizer.Encode(text, fixedLength: _maxLen);
            var tensor = ToTensor(tokens);
            var embedding = _model.Forward(tensor);
            var vector = embedding.Data.ToArray();  // [D]
            results.Add(new Embedding<float>(vector));
        }
        return results;
    }

    public object? GetService(Type serviceType, object? serviceKey) => this;
    public TService? GetService<TService>(object? key = null) where TService : class => this as TService;
    public void Dispose() { }
}
```

### DI wiring

```csharp
builder.Services.AddEmbeddingGenerator(sp =>
{
    var model = LoadEmbeddingModel();
    var tokenizer = LoadTokenizer();
    return new NivaraEmbeddingGenerator(model, tokenizer);
})
.UseDistributedCache()
.UseOpenTelemetry();
```

### Usage with vector store

```csharp
var generator = sp.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

// Generate embeddings
var embeddings = await generator.GenerateAsync(new[] { "hello world", "test query" });

// Use with any IVectorStore that accepts IEmbeddingGenerator
```

## What This Exercises

| API / Feature | Purpose |
|---|---|
| `IEmbeddingGenerator<string, Embedding<float>>` | Standard embedding interface |
| `EmbeddingGeneratorMetadata` | Provider metadata |
| `Embedding<float>` | Standard embedding result type |
| Nivara `Embedding<T>` + `MeanPool` | Learned embedding model |
| DI `AddEmbeddingGenerator<>()` | Ecosystem integration |

## Core Library Changes

| New API | File | Purpose |
|---|---|---|
| `NivaraEmbeddingGenerator` | `samples/NivaraChat/` or `src/Nivara/AutoDiff/Nn/` | `IEmbeddingGenerator` wrapper |

---

# G. Intent Classification Router

## Goal

Train a small Nivara intent classifier and use it as the root router
in an Agent Framework **handoff orchestration**. The classifier
determines the user's intent, then hands off to the appropriate
specialist agent.

This demonstrates the handoff pattern and shows Nivara as the
"triage" layer that determines which specialist (ML or LLM)
should handle the request.

```
User input
    │
    v
[IntentClassifier]           Nivara TextClassifierModel, 5 classes
    │
    ├── "factual"      ──> [RAG Agent]           retrieval + LLM generation
    ├── "question"     ──> [LLM Agent]           general Q&A via Ollama
    ├── "command"      ──> [Tool Agent]          LLM with AIFunction tools
    ├── "complaint"    ──> [EscalationAgent]     human-in-the-loop
    └── "chitchat"     ──> [NivaraChatClient]    local transformer (#A)
```

### Intent model

```
TextClassifierModel<float>:
  Embedding(vocab, 32) → MeanPool → Linear(32, 64) → ReLU → Linear(64, 5)

5 classes: factual, question, command, complaint, chitchat
```

Training data: synthetic templates with varied phrasings per intent.

### Handoff workflow

```csharp
using Microsoft.Agents.AI.Workflows;

// Define agents
var intentClassifier = new IntentClassifierExecutor(classifierModel, tokenizer);
var ragAgent = chatClient.AsAIAgent(instructions: "Answer using provided context...");
var llmAgent = chatClient.AsAIAgent(instructions: "Answer the user's question...");
var toolAgent = chatClient.AsAIAgent(instructions: "Execute the requested action...", tools: nivaraTools);
var escalationAgent = new EscalationExecutor();

// Build handoff workflow
var workflow = AgentWorkflowBuilder
    .BuildSequential(intentClassifier)  // start with classifier
    // Handoffs are configured based on intent labels
    .Build();

// Or use conditional edges for more control:
builder.AddEdge(intentClassifier, ragAgent,
    condition: msg => msg is IntentResult r && r.Intent == "factual");
builder.AddEdge(intentClassifier, llmAgent,
    condition: msg => msg is IntentResult r && r.Intent == "question");
builder.AddEdge(intentClassifier, toolAgent,
    condition: msg => msg is IntentResult r && r.Intent == "command");
// ... etc
```

## What This Exercises

| API / Feature | Purpose |
|---|---|
| `TextClassifierModel<T>` | Intent classification (existing model) |
| `AgentWorkflowBuilder` / Handoff | Dynamic routing (new pattern) |
| Conditional edges | Intent-based routing |
| Multiple `AIAgent` instances | Specialist agents per intent |
| `ChatClientAgent` + `IChatClient` | LLM-backed agents |

## Core Library Changes

| New API | File | Purpose |
|---|---|---|
| `IntentClassifierModel<T>` | `samples/NivaraChat/Training/` | 5-class intent model (same arch as sentiment) |

---

# H. Online Learning from LLM Feedback

## Goal

When the LLM generates high-quality responses, use them as training
data to progressively improve the Nivara models. The LLM becomes a
"teacher" that generates labeled examples, and Nivara retrains on them.

This demonstrates the **AI improves ML** direction — the stochastic
model generates data that the deterministic model learns from,
eventually reducing reliance on the LLM.

```
Phase 1: Initial training
  Synthetic data → Train Nivara models → Save

Phase 2: Online augmentation
  User queries → LLM generates responses → Nivara validates
  ├── validated (consistent) → add to training set
  └── rejected (inconsistent) → discard

Phase 3: Retrain
  Augmented training set → Fine-tune Nivara models → Save
  (repeat phases 2-3 periodically)

Over time: Nivara handles more cases, fewer LLM calls needed
```

### Feedback collection executor

```csharp
public partial class FeedbackCollectorExecutor : Executor<string>
{
    private readonly List<TrainingExample> _buffer = new();
    private const int RetrainThreshold = 100;

    [MessageHandler]
    public async ValueTask<string> HandleAsync(string input, IWorkflowContext ctx)
    {
        // 1. Get LLM response
        var llmResponse = await _chatClient.GetResponseAsync(input);

        // 2. Validate with Nivara critic
        var isValid = _criticModel.Predict($"{input} || {llmResponse.Text}");

        // 3. If valid, add to training buffer
        if (isValid == "consistent")
        {
            _buffer.Add(new TrainingExample(input, llmResponse.Text));

            // 4. When buffer is full, trigger retraining
            if (_buffer.Count >= RetrainThreshold)
            {
                await RetrainModels(_buffer.ToArray());
                _buffer.Clear();
            }
        }

        return llmResponse.Text;
    }
}
```

### Incremental training

```csharp
private async Task RetrainModels(TrainingExample[] examples)
{
    // Build augmented dataset from original + LLM-generated examples
    var allTexts = examples.Select(e => e.Input).ToArray();
    var allLabels = examples.Select(e => e.Label).ToArray();

    // Tokenize, build frame, create DataLoader
    // Use same TrainingLoop<T> but with warm-start from existing weights
    var model = LoadExistingModel();  // warm start
    var loader = BuildDataLoader(allTexts, allLabels);
    var loop = new TrainingLoop<float>(model, loader, lossFn, optimizer, epochs: 5);
    loop.Run();

    // Save improved model
    ModelSerializer.Save(model, "models/sentiment-v2.json");
}
```

## What This Exercises

| API / Feature | Purpose |
|---|---|
| `TrainingLoop<T>` | Retraining with augmented data |
| `DataLoader<T>` | New data loading |
| `ModelSerializer.Save/Load` | Warm-start from existing weights |
| `IChatClient.GetResponseAsync` | LLM as data generator |
| Nivara `TextClassifierModel` | Progressive improvement |

## Core Library Changes

| New API | File | Purpose |
|---|---|---|
| Incremental `TrainingLoop` option | `src/Nivara/AutoDiff/Training/` | Warm-start support (load existing weights before training) |

---

# Composition: The Full Hybrid Showcase

These features compose into a complete demo:

```
User input
    │
    v
[IntentClassifier] (#G)      Nivara triage
    │
    ├── "factual" → [RAG Retriever] (#D) → [LLM Agent] → output
    │                    ↑
    │           Nivara embeddings + vector search
    │
    ├── "question" → [ConfidenceRouter] (#B)
    │                    ├── high conf → Nivara result (fast)
    │                    └── low conf  → [LLM Agent] → output
    │
    ├── "command" → [Tool Agent] (#C)    LLM calls Nivara tools
    │
    ├── "complaint" → [Escalation]       human-in-the-loop
    │
    └── "chitchat" → [NivaraChatClient] (#A)  local transformer
                         ↑
                   IChatClient backed by Nivara

All paths optionally feed into [Writer-Critic Loop] (#E) for quality
All validated LLM outputs feed into [Online Learning] (#H) for improvement
```

### What the full showcase demonstrates

| Capability | Feature |
|---|---|
| Deterministic ML classification | Intent router (#G) |
| Deterministic ML NER | Entity extractor (existing) |
| Local vector search | RAG with Nivara embeddings (#D) |
| LLM tool calling with ML tools | AIFunction tools (#C) |
| Confidence-gated routing | Handoff pattern (#B) |
| Local LLM inference | IChatClient transformer (#A) |
| Iterative quality improvement | Writer-critic loop (#E) |
| ML model improvement from LLM | Online learning (#H) |
| Agent Framework patterns | Sequential, concurrent, conditional, handoff |
| .NET AI ecosystem | IChatClient, IEmbeddingGenerator, AIFunctionFactory |
| Nivara AutoDiff | Module<T>, TrainingLoop, DataLoader, ModelSerializer |

---

# Recommendation: Implementation Priority

Not all features are equal. Some are quick wins that exercise new patterns
with zero library changes; others require new core APIs. Here's how I'd
sequence them for maximum impact per unit of effort.

## Tier 1 — Quick wins, no library changes (1-2 days each)

**Start here.** These use existing Nivara models and Agent Framework
patterns as-is. They prove the hybrid thesis without touching core.

### C. AIFunction Tools (highest ROI)

This is the easiest "wow" demo. You take the three models that already
work in NivaraChat, wrap them with `AIFunctionFactory.Create(...)`, and
suddenly the LLM can call them. No new training, no new modules, no
workflow changes — just a different composition angle. It also exercises
the most .NET AI ecosystem surface area (`AIFunctionFactory`,
`AsAIAgent`, `FunctionInvokingChatClient`) for the least effort.

The key insight this demonstrates: deterministic ML models are
**composable tools** in the .NET AI stack, not just workflow nodes.
That's a different and arguably more powerful framing than the
existing fan-out/fan-in approach.

### E. Writer-Critic Loop

Also zero library changes. Reuses the existing validator model as the
critic. The bounded retry loop inside the executor is simpler than
workflow-level feedback topology and still demonstrates the concept.
Good for showing that Nivara models can **evaluate** LLM output,
not just generate their own.

### B. Confidence-Based Handoff

Uses existing models, adds conditional edges (a new Agent Framework
pattern for NivaraChat). The confidence extraction is just reading
the softmax output — no new Nivara APIs needed. This is the cleanest
demonstration of the hybrid fast-path thesis: deterministic when
confident, stochastic when not.

## Tier 2 — Small additions (2-3 days each)

These need a thin wrapper or a new training setup but no core library
changes.

### F. IEmbeddingGenerator

A thin wrapper class (~100 lines) around an existing `Embedding<T>` +
`MeanPool` trained model. The value: plugs Nivara into the vector
store ecosystem and sets up the RAG story. The `IEmbeddingGenerator`
interface is simpler than `IChatClient` — fewer methods, no streaming,
no conversation history. Good intermediate step before #A.

### G. Intent Classification Router

Training a 5-class intent model is the same pattern as the existing
sentiment model — same `TextClassifierModel<float>` architecture,
just different synthetic data. The interesting part is the handoff
workflow topology, which exercises a new Agent Framework pattern.

### D. RAG with Nivara Embeddings

Depends on #F (IEmbeddingGenerator) or can use raw `Embedding<T>` +
`CosineSimilarity` directly. The NivaraVectorStore is a simple
in-memory list with linear scan — no ANN index needed for a demo
with hundreds of chunks. The `TextSearchProvider` integration
shows how Nivara fits into the official Agent Framework RAG path.

## Tier 3 — Significant effort (1-2 weeks)

These require new core library APIs and real training infrastructure.

### A. IChatClient Backed by Batched Transformer

The big one. Requires LayerNorm, batched Embedding, Concat ops,
and a real transformer architecture. The `NivaraChat-NEXT.md` spec
already has the full architecture and gap analysis. This is the
flagship demo — "Nivara trains and serves its own LLM-compatible
chat client" — but it's also the most work. Depends on the gap
fixes in sections Gap 1-9.

My suggestion: do this **after** Tier 1 and 2 are working. By then
you'll have validated the Agent Framework integration, the hybrid
patterns, and the ecosystem wiring. The transformer becomes the
capstone that ties it all together: the intent classifier routes
to NivaraChatClient, which is itself an IChatClient that can be
used anywhere in the .NET AI stack.

### H. Online Learning from LLM Feedback

The most ambitious in terms of implications (ML improving from LLM
output) but the least urgent from a demo perspective. Requires
incremental `TrainingLoop` support (warm-start), a feedback buffer,
and a periodic retrain flow. Save this for when the other features
are working and you want to show the "virtuous cycle" story.

## The Narrative Arc

If you implement these in the recommended order, the NivaraChat sample
tells a progressive story:

1. **v1 (existing):** Three trained models + optional LLM in a
   fan-out/fan-in workflow. Proves Nivara can train and serve.

2. **v2 (Tier 1):** Same models, but now the LLM *calls them as tools*.
   Confidence routing decides when to use ML vs. LLM. Writer-critic
   loop ensures quality. **Shows the hybrid architecture in action.**

3. **v3 (Tier 2):** Nivara becomes a local embedding service for RAG.
   Intent routing sends requests to the right specialist. **Shows
   Nivara in multiple ecosystem roles** (classifier, embedding service,
   tool, validator).

4. **v4 (Tier 3):** Nivara trains and serves its own transformer as
   a full IChatClient. Online learning improves models from LLM
   feedback. **The full circle: Nivara is both the ML backbone and
   a first-class citizen in the .NET AI ecosystem.**

Each version is a complete, runnable demo. None requires the others.
But together they make a compelling case that deterministic ML and
stochastic LLMs are complementary, not competing, approaches — and
Nivara is the .NET library that bridges them.
