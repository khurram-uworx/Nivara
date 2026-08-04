# NivaraChat MiniLM Embedding Demo

## Purpose

Add a MiniLM-based `IEmbeddingGenerator<string, Embedding<float>>` demo to NivaraChat, proving Nivara can serve as a local embedding provider that plugs directly into the .NET AI ecosystem.

This is NEXT.md section F — the bridge between Nivara's trained transformer models and the standard `Microsoft.Extensions.AI` embedding interface.

## Background

### What exists today

| Component | Location | Status |
|-----------|----------|--------|
| `NivaraEmbeddingGenerator<TInput>` | `src/Nivara.Extensions/AI/NivaraEmbeddingGenerator.cs` | Implements `IEmbeddingGenerator<TInput, Embedding<float>>` — takes `Func<TInput, float[]>` |
| `MiniLMDistilled<T>` | `samples/Nivara.Samples/BertModel.cs` | Full transformer encoder, CLS extraction, L2 normalization — produces 384-dim embeddings |
| `MiniLMTokenizer` | `samples/Nivara.Samples/BertModel.cs:437` | Wraps `Microsoft.ML.Tokenizers.BertTokenizer`, `Encode`, `TokenizeWithMask` |
| `BertEncoder<T>`, `BertLayer<T>` | `samples/Nivara.Samples/BertModel.cs` | Building blocks — already promoted to Nivara.Samples |
| `SafeTensorsLoader` | `samples/Nivara.Samples/SafeTensorsLoader.cs` | Generic dtype-aware weight loading |
| `Nivara.Extensions` project | `src/Nivara.Extensions/` | Already references `Microsoft.Extensions.AI.Abstractions` |

### What NivaraChat currently uses

- `TextTokenizer` — word-level tokenizer (core Nivara), different from BertTokenizer
- `TextClassifierModel<float>` / `TokenClassifierModel<float>` — custom Embedding+MLP, not transformer
- `ITextModel` interface — `string → string` classification wrapper
- No `IEmbeddingGenerator` usage
- Chat flow: user text → sentiment/entity → validator → optional LLM — **no context retrieval**

### How Microsoft says it should work

From MS Learn (`dotnet/ai/iembeddinggenerator`):

```csharp
// Standard usage pattern
IEmbeddingGenerator<string, Embedding<float>> generator = /* provider */;

// Batch embedding
GeneratedEmbeddings<Embedding<float>> embeddings =
    await generator.GenerateAsync(["text1", "text2"]);

// Single embedding (extension method)
ReadOnlyMemory<float> vector = await generator.GenerateVectorAsync("text1");

// DI registration
builder.Services.AddEmbeddingGenerator(sp => /* factory */)
    .UseDistributedCache()
    .UseOpenTelemetry();
```

Key requirements from MS docs:
- Thread-safe for concurrent use
- `GenerateAsync` accepts `IEnumerable<TInput>`, returns `Task<GeneratedEmbeddings<TEmbedding>>`
- `EmbeddingGeneratorMetadata` for provider identification
- `GetService` pattern for strongly-typed service resolution
- `IDisposable` support

## How the demo should work

The demo is framed as **"chat with context retrieval"** — the user types a question, the system retrieves relevant context via `IEmbeddingGenerator`, then shows what would be fed to the LLM. This connects directly to NEXT.md section D (RAG) and shows how `IEmbeddingGenerator` powers the retrieval step.

```
User runs: dotnet run --project samples/NivaraChat -- --embed

=== NivaraChat — Embedding Search ===

Loading MiniLM embedding model...
  Provider:     Nivara-MiniLM
  Model:        all-minilm-l6-v2
  Dimensions:   384
  Loaded in:    245 ms

Indexing 8 knowledge documents via IEmbeddingGenerator...
  [0] "The quick brown fox jumps over the lazy dog"
  [1] "Machine learning is a subset of artificial intelligence"
  [2] "The stock market closed at record highs today"
  [3] "Neural networks are inspired by biological brains"
  [4] "The weather forecast predicts rain tomorrow"
  [5] "Deep learning has revolutionized computer vision"
  [6] "Interest rates are expected to rise next quarter"
  [7] "Natural language processing enables text understanding"

Indexed in 180 ms — ready for chat

Type a message and press Enter (or 'quit' to exit):

> what is artificial intelligence?
  Retrieved 4 relevant documents (12 ms)

  Context for LLM:
    #1  0.8234  "Machine learning is a subset of artificial intelligence"
    #2  0.7123  "Neural networks are inspired by biological brains"
    #3  0.6891  "Deep learning has revolutionized computer vision"
    #4  0.6456  "Natural language processing enables text understanding"

  (In a full pipeline, these would be injected into the LLM prompt
   via TextSearchProvider — see NEXT.md section D)

> how does the brain work?
  Retrieved 4 relevant documents (11 ms)

  Context for LLM:
    #1  0.7567  "Neural networks are inspired by biological brains"
    #2  0.6234  "Deep learning has revolutionized computer vision"
    #3  0.5891  "Machine learning is a subset of artificial intelligence"
    #4  0.5234  "Natural language processing enables text understanding"

> quit
```

**What the user gets:**
1. A working semantic retrieval step — the core building block for RAG chat
2. Powered entirely by Nivara's `IEmbeddingGenerator` — no external vector DB, no cloud API
3. Same interface as OpenAI/Ollama embedding providers — can swap providers by changing one line
4. Shows the path to NEXT.md section D: this retrieval + TextSearchProvider + LLM = full RAG chat

---

## Task Breakdown

## Task 1: Add `Nivara.Extensions` reference to `Nivara.Samples`

### Priority

High

### Goal

Enable `Nivara.Samples` to use `NivaraEmbeddingGenerator<TInput>` from `Nivara.Extensions`.

### Why this exists

`NivaraEmbeddingGenerator<TInput>` already implements `IEmbeddingGenerator<TInput, Embedding<float>>` correctly. Duplicating it would be wrong. `Nivara.Samples` is the right home for the MiniLM-specific wiring since it already owns `MiniLMDistilled<T>` and `MiniLMTokenizer`.

### Scope

- Add `<ProjectReference Include="..\..\src\Nivara.Extensions\Nivara.Extensions.csproj" />` to `Nivara.Samples.csproj`
- Verify build succeeds

### Constraints

- `Nivara.Extensions` already references `Microsoft.Extensions.AI.Abstractions` — no new package needed
- `Nivara.Samples` already references `Microsoft.ML.Tokenizers` — already compatible

### Acceptance criteria

- `Nivara.Samples` builds without errors
- `NivaraEmbeddingGenerator<string>` is accessible from `Nivara.Samples` code

### Files likely involved

- `samples/Nivara.Samples/Nivara.Samples.csproj`

---

## Task 2: Add `MiniLMEmbeddingGenerator` factory to `Nivara.Samples`

### Priority

High

### Goal

Create a static factory method in `Nivara.Samples` that loads MiniLM + BertTokenizer and produces a ready-to-use `NivaraEmbeddingGenerator<string>`.

### Why this exists

The tokenization → forward → extract float[] pipeline is ~15 lines but specific to MiniLM. Encapsulating it in `Nivara.Samples` keeps NivaraChat clean and makes the pattern reusable across samples.

### Scope

- Add static class `MiniLMEmbeddingGenerator` to `samples/Nivara.Samples/BertModel.cs` (append to existing file, same file as `MiniLMTokenizer` and `MiniLMDistilled<T>`)
- Factory method: `static NivaraEmbeddingGenerator<string> Create(string modelDir, int maxLen = 128)`
- Loads weights from `model.safetensors`, vocab from `vocab.txt` in `modelDir`
- Returns `NivaraEmbeddingGenerator<string>` with `embeddingDimension: 384`

### Suggested implementation path

```csharp
public static class MiniLMEmbeddingGenerator
{
    public static NivaraEmbeddingGenerator<string> Create(
        string modelDir,
        int maxLen = 128,
        string providerName = "Nivara-MiniLM")
    {
        var tensors = SafeTensorsLoader.Read(Path.Combine(modelDir, "model.safetensors"));
        var config = BertConfig.FromJson(File.ReadAllText(Path.Combine(modelDir, "config.json")));
        var model = MiniLMDistilled<float>.LoadWeights(tensors, config);
        var tokenizer = MiniLMTokenizer.Load(Path.Combine(modelDir, "vocab.txt"));

        Func<string, float[]> embeddingFactory = text =>
        {
            var (input, mask) = MiniLMTokenizer.TokenizeWithMask(tokenizer, text, maxLen);
            var output = mask != null
                ? model.ForwardWithMask(input, mask)
                : model.Forward(input);

            var result = new float[output.Length];
            output.Data.TryGetSpan(out var span);
            if (!span.IsEmpty)
                span.Slice(0, output.Length).CopyTo(result);
            else
                output.Data.CopyTo(result, 0f);
            return result;
        };

        return new NivaraEmbeddingGenerator<string>(
            embeddingFactory,
            config.HiddenSize,
            providerName,
            defaultModelId: "all-minilm-l6-v2");
    }
}
```

### Constraints

- Must call `model.Eval()` after loading
- Lambda captures `model` + `tokenizer` — kept alive by generator
- Thread safety: single-threaded forward; document limitation

### Acceptance criteria

- `MiniLMEmbeddingGenerator.Create("samples/data/minilm")` returns valid `NivaraEmbeddingGenerator<string>`
- `generator.EmbeddingDimension == 384`
- `generator.GenerateAsync(["test"]).Result[0].Vector.Length == 384`

### Files likely involved

- `samples/Nivara.Samples/BertModel.cs`

---

## Task 3: Add `--embed` chat-with-context REPL to NivaraChat

### Priority

High

### Goal

Add a `--embed` mode that indexes knowledge documents, then runs an interactive REPL where the user types messages and sees context retrieved via `IEmbeddingGenerator` — framed as the retrieval step for chat.

### Why this exists

This is the deliverable. It demonstrates a practical use case: retrieving relevant context to feed into an LLM or Nivara classifier. The framing as "chat with context retrieval" connects to NEXT.md sections D (RAG) and F (IEmbeddingGenerator).

### Scope

- Add `--embed` case to the `switch` in `Program.Main`
- Add `RunEmbeddingSearch()` method
- Hardcoded sample documents (8-10 sentences covering diverse topics)
- Flow:
  1. Load model via `MiniLMEmbeddingGenerator.Create("samples/data/minilm")`
  2. Print provider metadata
  3. Embed all documents via `generator.GenerateAsync(documents)` — print indexing time
  4. REPL loop:
     - Read user message
     - Embed query via `generator.GenerateVectorAsync(query)` — print time
     - Compute `TensorPrimitives.CosineSimilarity` against each document
     - Sort descending, print top-4 with similarity scores
     - Print hint: "In a full pipeline, these would be injected into the LLM prompt via TextSearchProvider"

### Constraints

- MiniLM model files must exist at `samples/data/minilm/` — print clear error if missing
- Use `TensorPrimitives.CosineSimilarity` for similarity computation
- No training — inference only
- Show top-4 results (keep output clean)

### Acceptance criteria

- `dotnet run --project samples/NivaraChat -- --embed` runs without error
- Model loads and prints metadata
- Documents indexed with timing
- User can type messages and see ranked context results
- Output frames the results as context for a chat/LLM pipeline

### Files likely involved

- `samples/NivaraChat/Program.cs`

---

## Task 4: Update CLI help and README.md

### Priority

Medium

### Goal

Document the `--embed` mode in CLI help and README.md, explaining how it connects to the chat workflow and NEXT.md roadmap.

### Why this exists

Users need to know the mode exists and understand what it demonstrates in the context of NivaraChat's hybrid architecture.

### Scope

**Program.cs:**
- Add `--embed` row to CLI options table in help output

**README.md:**
- Add `--embed` to "Quick start" section with example command and sample output
- Add "Embedding search" subsection under "Modes of use" explaining:
  - What it does (indexes documents, interactive context retrieval)
  - How it uses `IEmbeddingGenerator` (the .NET AI ecosystem interface)
  - How it connects to NivaraChat (retrieval step for RAG — NEXT.md sections D/F)
- Add `NivaraEmbeddingGenerator` to the "Nivara APIs demonstrated" table
- Add `MiniLMEmbeddingGenerator` to the "Library gaps this example resolved" table

### Acceptance criteria

- `--help` shows `--embed` option
- README.md has quick start command, mode description, and API table entries
- README.md explains the connection to NEXT.md roadmap (sections D/F)

### Files likely involved

- `samples/NivaraChat/Program.cs`
- `samples/NivaraChat/README.md`

---

## Coordination Notes

- Tasks 1 and 2 are sequential (2 depends on 1)
- Task 3 depends on Task 2
- Task 4 is independent of Tasks 2-3 (can run in parallel)
- No shared files that would create merge conflicts
- The `Nivara.Extensions` project reference in Task 1 should be tested against the full solution build

## Suggested Agent Handout Batches

### Batch A: prerequisite

- Task 1 (project reference)

### Batch B: implementation

- Task 2 (MiniLMEmbeddingGenerator factory)
- Task 3 (--embed REPL) — can start after Task 2 completes

### Batch C: polish

- Task 4 (help text + README)

## Final Checklist

- every task has a clear owner-sized scope
- every task has acceptance criteria
- decision-gate tasks are clearly marked (none — all decisions made by NEXT.md F and MS Learn docs)
- likely files are listed to reduce agent search time
- execution order reflects real dependencies (1 → 2 → 3, 4 parallel)
