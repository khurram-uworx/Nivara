# NivaraChat — Confidence, Tools & Critic (B, C, E)

## Purpose

Add three new modes to NivaraChat that showcase Nivara models in roles beyond fixed-pipeline classification:

- **B — Confidence handoff:** Nivara decides whether the LLM is needed (`--handoff`)
- **C — AIFunction tools:** LLM decides when to call Nivara models
- **E — Writer-critic:** Nivara judges LLM quality, triggers re-generation

All three use the same trained models already in NivaraChat (`TextClassifierModel<float>`, `TokenClassifierModel<float>`) — different composition, same weights.

## What exists today

| Component | File | Current behavior |
|-----------|------|-----------------|
| `SentimentExecutor` | `SentimentExecutor.cs` | Returns label string only ("Positive") — no confidence score |
| `ModelInferenceHelper.SoftmaxConfidence` | `ModelInferenceHelper.cs` | **Already computes softmax confidence** — exists but unused by executors |
| `ValidatorExecutor` | `ValidatorExecutor.cs` | Rule-based — checks if entities/sentiment exist, hardcodes confidence |
| `LlmExecutor` | `LlmExecutor.cs` | Calls `OllamaApiClient.GetResponseAsync` — no tool support |
| `NivaraChatClient` | `NivaraChatClient.cs` | Wraps `ITextModel` as `IChatClient` — used in `--agents` mode |
| Agent Framework | `Program.cs` | Fan-out/fan-in only — no conditional edges |

### Key observation

`ModelInferenceHelper.SoftmaxConfidence()` already exists and works. The gap is that **executors don't expose confidence scores**, and the workflow **has no conditional routing**. The primitives are there; the composition is missing.

---

## Task Breakdown

## Task 1: Add confidence extraction to SentimentExecutor and EntityExtractor

### Priority

High (prerequisite for Task 2)

### Goal

Make both executors return structured JSON with `{ label, confidence }` instead of plain strings, so downstream executors can route on confidence.

### Why this exists

B (confidence handoff) and E (critic scoring) both need confidence scores from Nivara models. Today `SentimentExecutor` returns `"Positive"` — no score. `ModelInferenceHelper.SoftmaxConfidence()` already computes it but nobody calls it from the executor.

### Scope

- Modify `SentimentExecutor.HandleAsync` to return JSON: `{"label":"Positive","confidence":0.94}`
- Modify `EntityExtractor.HandleAsync` to return JSON: `{"entities":{...},"confidence":0.89}` (average per-token confidence)
- Add `ModelInferenceHelper.RunClassifierWithConfidence` that returns `(int bestClass, float confidence)` — reusable by both executors
- Update `ValidatorExecutor` to parse the new JSON format (it currently checks for `{` prefix to detect entities)
- Verify existing `--workflow` and `--agents` modes still work (they parse executor output as strings)

### Constraints

- Existing modes must not break — `ValidatorExecutor` already handles JSON from `EntityExtractor`
- Confidence is max softmax probability — same as `SentimentTextModel.Process()` uses

### Suggested implementation path

```csharp
// New helper in ModelInferenceHelper.cs
public static (int bestClass, float confidence) RunClassifierWithConfidence(
    TextClassifierModel<float> model, TextTokenizer tokenizer,
    string input, int maxSeqLen, int numClasses)
{
    var tensorInput = ToTensor(tokenizer, input, maxSeqLen);
    var logits = model.Forward(tensorInput);
    int bestClass = ArgMax(logits.Data, 0, numClasses);
    float confidence = SoftmaxConfidence(logits.Data, 0, numClasses, bestClass);
    return (bestClass, confidence);
}

// SentimentExecutor returns JSON
public override ValueTask<string> HandleAsync(string text, IWorkflowContext context, ...)
{
    var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
        _model, _tokenizer, text, _maxSeqLen, numClasses: 3);
    var result = JsonSerializer.Serialize(new { label = Classes[bestClass], confidence });
    return ValueTask.FromResult(result);
}
```

### Acceptance criteria

- `SentimentExecutor` returns `{"label":"Positive","confidence":0.94}` format
- `EntityExtractor` returns `{"entities":{...},"confidence":0.89}` format
- `--workflow` mode still works (ValidatorExecutor parses both formats)
- `--agents` mode still works (NivaraChatClient receives string output)

### Files likely involved

- `samples/NivaraChat/ModelInferenceHelper.cs`
- `samples/NivaraChat/SentimentExecutor.cs`
- `samples/NivaraChat/EntityExtractor.cs`
- `samples/NivaraChat/ValidatorExecutor.cs`

---

## Task 2: Add ConfidenceRouter executor and `--handoff` mode (Feature B)

### Priority

High

### Goal

Add a `--handoff` mode that routes to the LLM only when Nivara models are uncertain, demonstrating the confidence-based handoff pattern.

### Why this exists

This is the core hybrid thesis: deterministic ML owns the fast path, stochastic LLM handles the uncertain tail. Today every query goes through the full pipeline — there's no decision about whether the LLM is needed.

### Scope

- Add `ConfidenceRouter` executor that reads confidence from sentiment + entity results
- Add `NivaraResultFormatter` executor that formats confident Nivara results for display
- Wire conditional edges in `--handoff` mode:
  - Both confident (>= 0.8) → `NivaraResultFormatter` → output (skip LLM)
  - Either uncertain (< 0.8) → `LlmExecutor` → output
- Add `--handoff` case to `Program.Main` switch
- Add `RunHandoff()` method that builds the conditional workflow
- Update `PrintUsage()` and README.md

### Constraints

- Confidence threshold: 0.8 default (configurable via `--threshold`)
- "Both confident" means sentiment AND entity extraction are both above threshold
- LLM receives Nivara partial results as context when it fires
- Existing modes unchanged

### Suggested implementation path

```csharp
// ConfidenceRouter — reads two JSON inputs, decides path
internal sealed class ConfidenceRouter : Executor<string, string>
{
    private readonly float _threshold;
    private readonly List<string> _pending = [];

    public ConfidenceRouter(float threshold = 0.8f) : base("ConfidenceRouter")
    {
        _threshold = threshold;
    }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, ...)
    {
        _pending.Add(input);
        if (_pending.Count < 2) return ValueTask.FromResult("");

        // Parse both results
        var sentiment = JsonSerializer.Deserialize<ConfidenceResult>(_pending[0]);
        var entities = JsonSerializer.Deserialize<ConfidenceResult>(_pending[1]);
        _pending.Clear();

        bool confident = sentiment.Confidence >= _threshold
                      && entities.Confidence >= _threshold;

        return ValueTask.FromResult(JsonSerializer.Serialize(new
        {
            confident,
            sentiment,
            entities,
            threshold = _threshold
        }));
    }
}

// NivaraResultFormatter — formats confident results for display
internal sealed class NivaraResultFormatter : Executor<string, string>
{
    public NivaraResultFormatter() : base("NivaraResult") { }

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, ...)
    {
        var data = JsonSerializer.Deserialize<RouterOutput>(input);
        var result = $"Sentiment: {data.Sentiment.Label} (confidence: {data.Sentiment.Confidence:F2})\n"
                   + $"Entities: {JsonSerializer.Serialize(data.Entities.Entities)}\n"
                   + $"(Handled by Nivara in <1ms — no LLM needed)";
        return ValueTask.FromResult(result);
    }
}

// Program.cs — RunHandoff()
// Workflow:
//   TextRouter → fan-out → [SentimentExecutor, EntityExtractor]
//                      → fan-in → ConfidenceRouter
//                      → conditional(confident) → NivaraResultFormatter → output
//                      → conditional(!confident) → LlmExecutor → output
```

### Acceptance criteria

- `dotnet run --project samples/NivaraChat -- --handoff --text "I love this product!"` returns Nivara result (no LLM)
- `dotnet run --project samples/NivaraChat -- --handoff --text "This product is interesting but I'm not sure"` calls LLM
- Confidence threshold printed in output
- `--threshold` flag adjusts the cutoff (default 0.8)
- `--ollama` required for LLM path — clear error if missing

### Files likely involved

- `samples/NivaraChat/ConfidenceRouter.cs` (new)
- `samples/NivaraChat/NivaraResultFormatter.cs` (new)
- `samples/NivaraChat/Program.cs`

---

## Task 3: Add Nivara AIFunction tools and `--tools` mode (Feature C)

### Priority

High

### Goal

Wrap Nivara models as `AIFunction` tools that the LLM can call, demonstrating the LLM-as-orchestrator pattern.

### Why this exists

The current architecture has Nivara feeding results *into* the LLM pipeline. This flips it: the LLM *decides* when to call Nivara. Same models, different composition — proves Nivara models are composable tools, not just pipeline nodes.

### Scope

- Create `NivaraToolFunctions` static class with three tool methods:
  - `AnalyzeSentiment(string text)` → `{"label":"positive","confidence":0.94}`
  - `ExtractEntities(string text)` → `{"person":["John"],"org":["Acme Corp"],...}`
  - `ValidateResponse(string original, string response)` → `{"consistent":true,"confidence":0.91}`
- Wrap each with `AIFunctionFactory.Create(...)` with `[Description]` attributes
- Add `ToolOrchestrator` executor that calls `chatClient.AsAIAgent(tools: nivaraTools)` and runs the agent
- Add `--tools` case to `Program.Main` switch
- Add `RunTools()` method
- Update `PrintUsage()` and README.md

### Constraints

- Requires `--ollama` — clear error if not provided
- Each tool is a static method that loads its own model (or accepts injected models)
- Tool descriptions must be clear enough for the LLM to choose correctly
- Single-threaded model inference (document limitation)

### Suggested implementation path

```csharp
// NivaraToolFunctions.cs — static methods wrapped as AIFunction
public static class NivaraToolFunctions
{
    private static TextClassifierModel<float>? _sentimentModel;
    private static TextTokenizer? _sentimentTokenizer;
    // ... similar for entity, validator

    [Description("Analyze sentiment of text. Returns positive/negative/neutral with confidence score.")]
    public static string AnalyzeSentiment(
        [Description("The text to analyze")] string text)
    {
        EnsureSentimentModel();
        var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
            _sentimentModel!, _sentimentTokenizer!, text, maxSeqLen: 20, numClasses: 3);
        return JsonSerializer.Serialize(new { label = Classes[bestClass], confidence });
    }

    [Description("Extract named entities (person, organization, date, location) from text.")]
    public static string ExtractEntities(
        [Description("The text to extract entities from")] string text)
    {
        EnsureEntityModel();
        var entities = ModelInferenceHelper.RunTokenClassifier(
            _entityModel!, _entityTokenizer!, text, maxSeqLen: 20, EntityClasses);
        return JsonSerializer.Serialize(entities);
    }

    [Description("Validate whether a response is consistent with the original text.")]
    public static string ValidateResponse(
        [Description("The original text")] string original,
        [Description("The response to validate")] string response)
    {
        EnsureValidatorModel();
        var input = $"{original} || {response}";
        var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
            _validatorModel!, _validatorTokenizer!, input, maxSeqLen: 40, numClasses: 2);
        return JsonSerializer.Serialize(new { consistent = bestClass == 1, confidence });
    }
}

// ToolOrchestrator.cs
internal sealed class ToolOrchestrator : Executor<string, string>
{
    private readonly OllamaApiClient _chatClient;

    public override async ValueTask<string> HandleAsync(string input, IWorkflowContext context, ...)
    {
        var tools = new[]
        {
            AIFunctionFactory.Create(NivaraToolFunctions.AnalyzeSentiment),
            AIFunctionFactory.Create(NivaraToolFunctions.ExtractEntities),
            AIFunctionFactory.Create(NivaraToolFunctions.ValidateResponse),
        };

        var agent = _chatClient.AsAIAgent(
            name: "NivaraOrchestrator",
            instructions: "You are an analyst. Use the provided Nivara tools to analyze text. "
                        + "Always call tools before generating your response. "
                        + "Present a clear summary of all tool results.",
            tools: tools);

        var result = await agent.RunAsync($"Analyze this text using available tools: {input}");
        return result.Message.Text;
    }
}
```

### Acceptance criteria

- `dotnet run --project samples/NivaraChat -- --tools --ollama --text "John Smith from Acme Corp reported great work"` calls all three Nivara tools
- LLM response includes sentiment, entities, and validation results
- Tool calls are visible in output (print each tool call + result)
- Without `--ollama`, prints clear error
- Tool descriptions are accurate — LLM chooses correct tools

### Files likely involved

- `samples/NivaraChat/NivaraToolFunctions.cs` (new)
- `samples/NivaraChat/ToolOrchestrator.cs` (new)
- `samples/NivaraChat/Program.cs`

---

## Task 4: Add CriticExecutor and `--critic` mode (Feature E)

### Priority

High

### Goal

Add a writer-critic loop where the LLM generates a response, a Nivara model scores it, and the LLM re-generates if quality is below threshold.

### Why this exists

Demonstrates that Nivara models can *evaluate* LLM output, not just generate their own. The validator model is already trained to check consistency — reusing it as a critic shows the same architecture serving different roles.

### Scope

- Add `CriticExecutor` that scores LLM response quality using the validator model
- Add `WriterCriticLoop` executor that orchestrates the bounded retry loop internally
- Wire the loop: Writer → Critic → (score >= 0.8 ? done : re-prompt writer)
- Max 3 iterations with structured feedback
- Add `--critic` case to `Program.Main` switch
- Add `RunCritic()` method
- Update `PrintUsage()` and README.md

### Constraints

- Requires `--ollama` — clear error if not provided
- Validator model is reused as critic (same `TextClassifierModel<float>`, different input format)
- Max 3 iterations — print each attempt with score
- Bounded loop inside executor (simpler than workflow-level feedback topology)

### Suggested implementation path

```csharp
// CriticExecutor.cs — scores LLM response quality
internal sealed class CriticExecutor : Executor<string, string>
{
    private readonly TextClassifierModel<float> _model;
    private readonly TextTokenizer _tokenizer;
    private readonly int _maxSeqLen;

    public override ValueTask<string> HandleAsync(string input, IWorkflowContext context, ...)
    {
        // input format: "original || response"
        var (bestClass, confidence) = ModelInferenceHelper.RunClassifierWithConfidence(
            _model, _tokenizer, input, _maxSeqLen, numClasses: 2);
        // class 0 = inconsistent/poor, class 1 = consistent/good
        return ValueTask.FromResult(JsonSerializer.Serialize(new
        {
            score = confidence,
            verdict = bestClass == 1 ? "GOOD" : "POOR",
            acceptable = bestClass == 1 && confidence >= 0.8f
        }));
    }
}

// WriterCriticLoop.cs — bounded retry loop
internal sealed class WriterCriticLoop : Executor<string, string>
{
    private readonly OllamaApiClient _chatClient;
    private readonly CriticExecutor _critic;
    private const int MaxIterations = 3;
    private const float QualityThreshold = 0.8f;

    public override async ValueTask<string> HandleAsync(string query, IWorkflowContext context, ...)
    {
        string? response = null;
        string feedback = "";

        for (int i = 0; i < MaxIterations; i++)
        {
            // Writer
            var prompt = string.IsNullOrEmpty(feedback)
                ? $"Answer this question clearly and concisely: {query}"
                : $"Answer this question. Previous attempt scored poorly. Feedback: {feedback}\n\nQuestion: {query}";

            response = await _chatClient.GetResponseAsync(prompt);
            var responseText = response.ToString();

            // Critic
            var critiqueInput = $"{query} || {responseText}";
            var critiqueJson = await _critic.HandleAsync(critiqueInput, context);
            var critique = JsonSerializer.Deserialize<CritiqueResult>(critiqueJson);

            if (critique.Acceptable)
                return $"Attempt {i + 1} — Score: {critique.Score:F2} (PASS)\n\n{responseText}";

            feedback = $"Previous attempt scored {critique.Score:F2} ({critique.Verdict}). "
                     + "Improve on: clarity, accuracy, relevance to the query.";
        }

        // Return best attempt after max iterations
        return $"Attempt {MaxIterations} — Score: below threshold (max iterations reached)\n\n{response}";
    }
}
```

### Acceptance criteria

- `dotnet run --project samples/NivaraChat -- --critic --ollama --text "Explain quantum computing to a5-year-old"` shows iteration loop
- Each attempt printed with score and verdict
- Pass threshold: score >= 0.8
- Max 3 iterations — if all fail, returns last attempt with notice
- Without `--ollama`, prints clear error

### Files likely involved

- `samples/NivaraChat/CriticExecutor.cs` (new)
- `samples/NivaraChat/WriterCriticLoop.cs` (new)
- `samples/NivaraChat/Program.cs`

---

## Task 5: Update CLI help and README.md

### Priority

Medium

### Goal

Document all three new modes in CLI help and README.md.

### Why this exists

Users need to know the modes exist and understand what each demonstrates.

### Scope

**Program.cs:**
- Add `--handoff`, `--tools`, `--critic` rows to CLI options table

**README.md:**
- Add quick start commands for all three modes
- Add "Hybrid mode" subsection under "Modes of use" with architecture diagram
- Add "Tool calling" subsection with example showing LLM calling Nivara tools
- Add "Writer-critic" subsection with example showing iteration loop
- Add `AIFunctionFactory`, `ConfidenceRouter`, `CriticExecutor` to "Nivara APIs demonstrated" table
- Add `NivaraToolFunctions`, `ConfidenceRouter`, `CriticExecutor` to "Library gaps" table

### Acceptance criteria

- `--help` shows all three new options
- README.md has quick start commands, mode descriptions, and architecture diagrams
- Each mode explains what Nivara feature it showcases

### Files likely involved

- `samples/NivaraChat/Program.cs`
- `samples/NivaraChat/README.md`

---

## Coordination Notes

- **Task 1 is prerequisite** for Tasks 2, 3, 4 — all need confidence scores
- Tasks 2, 3, 4 are independent of each other (can run in parallel after Task 1)
- Task 5 depends on Tasks 2-4 being complete
- No shared files between Tasks 2, 3, 4 — no merge conflicts
- All tasks modify `Program.cs` — but different switch cases and methods
- All tasks need `--ollama` for LLM path — consistent error handling

## Suggested Agent Handout Batches

### Batch A: prerequisite

- Task 1 (confidence extraction)

### Batch B: implementation (parallel)

- Task 2 (confidence handoff — `--handoff`)
- Task 3 (AIFunction tools — `--tools`)
- Task 4 (writer-critic — `--critic`)

### Batch C: polish

- Task 5 (help text + README)

## Final Checklist

- every task has a clear owner-sized scope
- every task has acceptance criteria
- decision-gate tasks are clearly marked (none — all decisions made)
- likely files are listed to reduce agent search time
- execution order reflects real dependencies (1 → 2/3/4 parallel → 5)
