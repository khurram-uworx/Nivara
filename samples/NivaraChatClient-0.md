# NivaraChatClient — IChatClient Implementation Backed by a Batched Transformer

## Goal

Implement the standard .NET `Microsoft.Extensions.AI.IChatClient` interface
backed by a Nivara-trained transformer model. The example trains a proper
batched causal transformer on TinyShakespeare (or similar small text
corpus), then wraps the trained model in an `IChatClient` that can be
wiredup via dependency injection.

This is the natural "next step" from MicroGpt: MicroGpt proves Nivara can
train a transformer; NivaraChatClient proves it can serve one in a
.ecosystem-compatible way.

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
