# Intent Classification Router — NivaraChat Item G

## Purpose

Implement a 5-class intent classifier that routes user input to specialist executors using the Agent Framework's `AddSwitch` pattern. This is Item G from `samples/NivaraChat/NEXT.md`.

The intent router demonstrates Nivara as the "triage" layer — a small deterministic ML model that decides which specialist (ML or LLM) should handle each request. Uses `SwitchBuilder` for clean multi-branch routing, not conditional edges (which have exclusivity problems).

## Suggested Execution Order

1. Task 1: Synthetic intent data generator
2. Task 2: Intent classifier training pipeline
3. Task 3: IntentClassifier executor
4. Task 4: Specialist executors (5 total)
5. Task 5: AddSwitch workflow wiring + CLI integration

## Coordination Notes

- Task 1 must complete before Task 2 (training needs data)
- Task 2 must complete before Task 3 (executor needs trained model)
- Tasks 4a–4e can run in parallel (independent executors)
- Task 5 depends on Tasks 3 and 4 (wiring needs all executors)
- Shared file: `Program.cs` — only Task 5 modifies it; no merge conflicts with other tasks
- Model files go to `samples/NivaraChat/models/` — same directory as existing models

## Task 1: IntentDataGenerator

### Priority

High

### Goal

Create `samples/NivaraChat/Data/IntentDataGenerator.cs` that generates 5-class synthetic intent classification data.

### Why this exists

The intent classifier needs labeled training data. No intent data exists in the codebase. The existing `SyntheticDataGenerator` provides the pattern to follow.

### Scope

- New file `samples/NivaraChat/Data/IntentDataGenerator.cs`
- Static method `GenerateIntentData(int count, int seed)` returning `(string[] texts, int[] labels)`
- 5 intent classes: factual (0), question (1), command (2), complaint (3), chitchat (4)
- Template-based generation with random substitutions (same pattern as `GenerateSentimentData`)
- ~20–30 templates per class with fill slots (`{person}`, `{org}`, `{topic}`, etc.)
- 1000+ samples default, reproducible via seed

### Constraints

- Follow existing code style (no comments, `camelCase` private fields)
- Use same data shape as other generators: `(string[] texts, int[] labels)`
- Labels must be contiguous ints 0–4

### Suggested implementation path

- Copy structure from `SyntheticDataGenerator.GenerateSentimentData`
- Define template arrays per intent class
- Define substitution vocabularies (names, orgs, topics, etc.)
- Generate samples by randomly selecting templates and filling slots
- Shuffle with the provided seed for reproducibility

### Acceptance criteria

- `GenerateIntentData(1000, 42)` returns 1000 texts and 1000 labels
- Labels are evenly distributed across 5 classes (~200 each)
- Each class has varied phrasings (not just one template repeated)
- No duplicate exact strings across the dataset
- Same seed produces same output (reproducibility)

### Files likely involved

- `samples/NivaraChat/Data/IntentDataGenerator.cs` (new)
- `samples/NivaraChat/Data/SyntheticDataGenerator.cs` (reference pattern)

## Task 2: IntentTrainer

### Priority

High

### Goal

Create `samples/NivaraChat/Training/IntentTrainer.cs` that trains a 5-class intent classifier and saves the model + tokenizer.

### Why this exists

The intent classifier executor needs a trained model. The training pipeline must follow the same pattern as `SentimentTrainer` for consistency.

### Scope

- New file `samples/NivaraChat/Training/IntentTrainer.cs`
- Static method `Train(int numSamples = 1000, int epochs = 20)`
- Uses `IntentDataGenerator.GenerateIntentData` for data
- Model: `TextClassifierModel<float>(vocabSize, embedDim=32, hiddenDim=64, numClasses=5, maxSeqLen=20)`
- Optimizer: `Adam<float>(learningRate: 0.001f)`
- Loss: `CrossEntropyLoss<float>`
- Save to `models/intent_model.json` and `models/intent_tokenizer.json`
- Print test accuracy after training

### Constraints

- Follow `SentimentTrainer` structure exactly (same framing, same serializer calls)
- maxSeqLen = 20 (matches existing models)
- Must be callable from `Program.cs` via `--intent-train` or `--intent` flags

### Suggested implementation path

- Copy `SentimentTrainer.cs` as skeleton
- Replace `SyntheticDataGenerator.GenerateSentimentData` → `IntentDataGenerator.GenerateIntentData`
- Change `numClasses: 3` → `numClasses: 5`
- Change model/tokenizer save paths to `intent_model.json` / `intent_tokenizer.json`
- Update accuracy logging to show per-class breakdown

### Acceptance criteria

- Running `IntentTrainer.Train()` produces `models/intent_model.json` and `models/intent_tokenizer.json`
- Training completes in <60 seconds on CPU
- Test accuracy >70% (5-class random baseline is 20%)
- Saved model can be loaded back via `ModelSerializer.Load` without errors

### Files likely involved

- `samples/NivaraChat/Training/IntentTrainer.cs` (new)
- `samples/NivaraChat/Training/SentimentTrainer.cs` (reference pattern)
- `samples/NivaraChat/models/intent_model.json` (output)
- `samples/NivaraChat/models/intent_tokenizer.json` (output)

## Task 3: IntentClassifier Executor

### Priority

High

### Goal

Create `samples/NivaraChat/IntentClassifier.cs` — an executor that classifies user input into one of 5 intents using the trained model.

### Why this exists

This is the routing node in the `AddSwitch` workflow. It receives raw user text and outputs a typed intent result that the switch-case conditions evaluate.

### Scope

- New file `samples/NivaraChat/IntentClassifier.cs`
- Class `IntentClassifier : Executor<string, string>`
- Constructor takes `TextClassifierModel<float>` and `TextTokenizer`
- `HandleAsync` method: tokenize input → `ModelInferenceHelper.RunClassifierWithConfidence` → map class index to intent string → return JSON
- Output format: `{"intent":"factual","confidence":0.92}`
- Intent string mapping: 0→"factual", 1→"question", 2→"command", 3→"complaint", 4→"chitchat"

### Constraints

- Inference must be no-grad (`requiresGrad: false`) — same as existing executors
- Use `ModelInferenceHelper.RunClassifierWithConfidence` for consistency
- Follow `Executor<string, string>` pattern from `SentimentExecutor`

### Suggested implementation path

- Copy `SentimentExecutor.cs` as skeleton
- Change model type from 3-class to 5-class
- Change JSON output from `{"label":..., "confidence":...}` to `{"intent":..., "confidence":...}`
- Add intent string mapping (switch expression on class index)

### Acceptance criteria

- `IntentClassifier` compiles and runs against trained intent model
- Input "What is the capital of France?" → `{"intent":"factual","confidence":...}`
- Input "Hello, how are you?" → `{"intent":"chitchat","confidence":...}`
- All 5 intent strings are valid JSON values
- Confidence is a float between 0 and 1

### Files likely involved

- `samples/NivaraChat/IntentClassifier.cs` (new)
- `samples/NivaraChat/SentimentExecutor.cs` (reference pattern)
- `samples/NivaraChat/ModelInferenceHelper.cs` (shared inference API)

## Task 4a: FactualExecutor (RAG)

### Priority

Medium

### Goal

Create `samples/NivaraChat/FactualExecutor.cs` — a RAG executor that retrieves relevant context and generates a grounded response.

### Why this exists

The "factual" intent needs retrieval-augmented generation. The `--rag` mode already has this logic; this executor wraps it as a workflow node.

### Scope

- New file `samples/NivaraChat/FactualExecutor.cs`
- Class `FactualExecutor : Executor<string, string>`
- Constructor takes `NivaraVectorStore`, `IChatClient` (Ollama), `NivaraEmbeddingGenerator`, `TextTokenizer`
- `HandleAsync` method: embed query → vector search (top 3) → inject context into prompt → call LLM → return response
- Reuse `DocumentChunker` and `InMemoryVectorStore` from existing RAG code

### Constraints

- Must handle empty vector store gracefully (no crashes)
- LLM prompt must include retrieved context
- Follow `Executor<string, string>` pattern

### Suggested implementation path

- Extract RAG logic from `--rag` mode in `Program.cs` into the executor
- Use `NivaraEmbeddingGenerator` for query embedding
- Use `TensorPrimitives.CosineSimilarity` for search (same as existing)
- Format prompt: "Answer based on this context: {context}\n\nQuestion: {input}"

### Acceptance criteria

- With indexed documents, returns grounded response referencing retrieved context
- With empty store, returns LLM response without context (graceful degradation)
- Response is a single string, not JSON-wrapped

### Files likely involved

- `samples/NivaraChat/FactualExecutor.cs` (new)
- `samples/NivaraChat/DocumentChunk.cs` (existing, reuse)
- `samples/NivaraChat/Program.cs` (reference RAG logic from `--rag` mode)

## Task 4b: QuestionExecutor

### Priority

Medium

### Goal

Create `samples/NivaraChat/QuestionExecutor.cs` — general Q&A via Ollama.

### Why this exists

The "question" intent needs a plain LLM Q&A path. `LlmExecutor` already does this; this executor wraps it with intent-appropriate framing.

### Scope

- New file `samples/NivaraChat/QuestionExecutor.cs`
- Class `QuestionExecutor : Executor<string, string>`
- Constructor takes `IChatClient` (Ollama)
- `HandleAsync` method: prepend system prompt → call `_chatClient.GetResponseAsync` → return response
- System prompt: "You are a helpful assistant. Answer the user's question clearly and concisely."

### Constraints

- Must handle Ollama being unavailable (return error message, not crash)
- Follow `LlmExecutor` pattern

### Suggested implementation path

- Copy `LlmExecutor.cs` as skeleton
- Change system prompt to question-answering focused
- Add Ollama availability check (same pattern as `ConfidenceRouter`)

### Acceptance criteria

- Returns LLM-generated answer for question inputs
- Handles Ollama unavailable gracefully (returns fallback message)
- Response is a single string

### Files likely involved

- `samples/NivaraChat/QuestionExecutor.cs` (new)
- `samples/NivaraChat/LlmExecutor.cs` (reference pattern)

## Task 4c: CommandExecutor

### Priority

Medium

### Goal

Create `samples/NivaraChat/CommandExecutor.cs` — LLM with AIFunction tools for action-oriented requests.

### Why this exists

The "command" intent needs tool-calling capability. `NivaraToolFunctions` already wraps the models; this executor wires them into a `ChatClientAgent`.

### Scope

- New file `samples/NivaraChat/CommandExecutor.cs`
- Class `CommandExecutor : Executor<string, string>`
- Constructor takes `IChatClient` (Ollama) and `AIFunction[]` tools
- `HandleAsync` method: create `ChatClientAgent` with tools → `agent.RunAsync(input)` → return response text
- System prompt: "You are an action assistant. Use the provided tools to fulfill the user's request."

### Constraints

- Must handle tool-calling failures gracefully
- Follow `NivaraToolFunctions` tool pattern
- Must not throw if Ollama is unavailable

### Suggested implementation path

- Create `ChatClientAgent` with tool definitions from `NivaraToolFunctions`
- Call `agent.RunAsync(input)` for automatic tool invocation
- Extract response text from `AgentResponseEvent`
- Handle case where LLM doesn't call any tools (returns plain text)

### Acceptance criteria

- For command inputs like "Analyze sentiment of this text", LLM calls the appropriate tool
- For inputs that don't need tools, LLM responds directly
- Tool results are incorporated into the final response
- No unhandled exceptions when tools fail

### Files likely involved

- `samples/NivaraChat/CommandExecutor.cs` (new)
- `samples/NivaraChat/NivaraToolFunctions.cs` (existing tools)

## Task 4d: EscalationExecutor

### Priority

Medium

### Goal

Create `samples/NivaraChat/EscalationExecutor.cs` — human-in-the-loop escalation for complaints.

### Why this exists

The "complaint" intent should flag the input for human review rather than trying to resolve it with an LLM.

### Scope

- New file `samples/NivaraChat/EscalationExecutor.cs`
- Class `EscalationExecutor : Executor<string, string>`
- No external dependencies (no LLM, no model)
- `HandleAsync` method: format escalation message with complaint text and timestamp → return it
- Output format: `"[ESCALATION] Complaint received at {time}: {input}. A human agent will follow up."`

### Constraints

- Must not call any LLM or model
- Must return a string (not void) so the workflow can output it
- Follow `Executor<string, string>` pattern

### Suggested implementation path

- Simple executor with no constructor dependencies
- Format string with complaint text and `DateTime.UtcNow`
- Optionally write to a log file for persistence

### Acceptance criteria

- Returns formatted escalation message
- Includes original complaint text
- Includes timestamp
- No external service calls

### Files likely involved

- `samples/NivaraChat/EscalationExecutor.cs` (new)

## Task 4e: ChitchatExecutor

### Priority

Medium

### Goal

Create `samples/NivaraChat/ChitchatExecutor.cs` — casual conversation via Ollama.

### Why this exists

The "chitchat" intent (and the default fallback) needs a casual chat path. Similar to `QuestionExecutor` but with a different system prompt and personality.

### Scope

- New file `samples/NivaraChat/ChitchatExecutor.cs`
- Class `ChitchatExecutor : Executor<string, string>`
- Constructor takes `IChatClient` (Ollama)
- `HandleAsync` method: prepend casual system prompt → call `_chatClient.GetResponseAsync` → return response
- System prompt: "You are a friendly chatbot. Have a casual, warm conversation with the user. Keep responses short and engaging."

### Constraints

- Must handle Ollama being unavailable gracefully
- Must be the default case in the switch (catches everything)
- Follow `LlmExecutor` pattern

### Suggested implementation path

- Copy `LlmExecutor.cs` as skeleton
- Change system prompt to casual/friendly tone
- Add Ollama availability check

### Acceptance criteria

- Returns friendly, casual response
- Handles Ollama unavailable gracefully
- Works as default fallback (no crash on unrecognized intents)

### Files likely involved

- `samples/NivaraChat/ChitchatExecutor.cs` (new)
- `samples/NivaraChat/LlmExecutor.cs` (reference pattern)

## Task 5: AddSwitch Workflow + CLI Integration

### Priority

High

### Goal

Wire up the intent classifier and 5 specialists into a `AddSwitch` workflow, add `--intent` CLI mode to `Program.cs`.

### Why this exists

This is the integration point that ties all previous tasks together into a runnable feature.

### Scope

- Modify `samples/NivaraChat/Program.cs`:
  - Add `--intent` CLI flag handling
  - Add `--intent-train` flag to train intent model
  - Add `BuildIntentWorkflow()` factory method
  - Add `RunIntentMode(string input)` for interactive REPL
  - Wire up intent model loading (from `models/intent_model.json`)
- Workflow topology:
  ```
  IntentClassifier → AddSwitch → [FactualExecutor, QuestionExecutor, 
                                   CommandExecutor, EscalationExecutor, 
                                   ChitchatExecutor]
  ```
- Condition factory: `IntentCondition(string intent)` checks JSON property
- Register all output sources via `WithOutputFrom`

### Constraints

- Must not break existing modes (`--workflow`, `--handoff`, `--tools`, etc.)
- `--intent` mode is standalone (doesn't combine with other modes)
- Workflow objects are single-use per run (use factory lambda)
- Must handle missing intent model (prompt to train first)

### Suggested implementation path

- Add `--intent` and `--intent-train` cases to the CLI switch in `Program.Main`
- Add `BuildIntentWorkflow()` that creates all executors and builds the switch-case graph
- Add `RunIntentMode()` for interactive REPL (same pattern as `--handoff`)
- Add `IntentCondition` helper method for switch-case predicates
- Load intent model/tokenizer via `ModelSerializer.Load`

### Acceptance criteria

- `--intent-train` trains and saves intent model
- `--intent` starts interactive REPL with intent routing
- Input "What is machine learning?" routes to FactualExecutor
- Input "Can you explain transformers?" routes to QuestionExecutor
- Input "Summarize this document" routes to CommandExecutor
- Input "I'm unhappy with the service" routes to EscalationExecutor
- Input "Hello!" routes to ChitchatExecutor
- Each specialist returns a response
- Workflow output shows which executor handled the request
- Existing modes (`--workflow`, `--handoff`, etc.) still work unchanged

### Files likely involved

- `samples/NivaraChat/Program.cs` (modify)
- `samples/NivaraChat/IntentClassifier.cs` (from Task 3)
- `samples/NivaraChat/FactualExecutor.cs` (from Task 4a)
- `samples/NivaraChat/QuestionExecutor.cs` (from Task 4b)
- `samples/NivaraChat/CommandExecutor.cs` (from Task 4c)
- `samples/NivaraChat/EscalationExecutor.cs` (from Task 4d)
- `samples/NivaraChat/ChitchatExecutor.cs` (from Task 4e)

## Suggested Agent Handout Batches

### Batch A: Data + Model (must run sequentially)

- Task 1: IntentDataGenerator
- Task 2: IntentTrainer

### Batch B: Executors (can run in parallel)

- Task 3: IntentClassifier
- Task 4a: FactualExecutor
- Task 4b: QuestionExecutor
- Task 4c: CommandExecutor
- Task 4d: EscalationExecutor
- Task 4e: ChitchatExecutor

### Batch C: Integration (depends on A + B)

- Task 5: AddSwitch workflow + CLI integration

## Final Checklist

- [ ] Every task has a clear owner-sized scope
- [ ] Every task has acceptance criteria
- [ ] Decision-gate tasks are clearly marked (none — all decisions made in planning)
- [ ] Likely files are listed to reduce agent search time
- [ ] Execution order reflects real dependencies (data → model → executor → wiring)
- [ ] 5 intent classes match NEXT.md spec (factual, question, command, complaint, chitchat)
- [ ] Uses `AddSwitch` pattern (not conditional edges) for routing
- [ ] Follows existing code conventions (no comments, camelCase, Executor<string, string>)
- [ ] All specialist executors handle Ollama unavailability gracefully
- [ ] Existing CLI modes remain unchanged
