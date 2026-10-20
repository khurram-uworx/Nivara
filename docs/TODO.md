# Phase 3 — Real Nivara-backed AIFunction tool + agent surface (#383)

## Problem
Phase 2 (`--qwen tools-weather`) proves the native function-calling loop with a fake
`GetWeather`. Phase 3 must replace/supplement it with a real Nivara capability invoked
as an `AIFunction`, feeding the Nivara result back so Qwen closes with a clean final
answer. Today the only Nivara-backed tools live in sample code
(`NivaraChat.Tools.NivaraToolFunctions`) and are not reusable building blocks.

## Proposed changes

### 1. Promote library building blocks (src/Nivara.Extensions/AI)
- New `NivaraAIFunctions.cs` (namespace `Nivara.AI`) exposing:
  - `CreateSentimentTool(TextClassifierModel<float>, TextTokenizer, ...)`
  - `CreateEntityTool(TokenClassifierModel<float>, TextTokenizer, ...)`
  - `CreateValidatorTool(TextClassifierModel<float>, TextTokenizer, ...)`
  - `CreateRegistered(..., toolsCsv = "all")` returning `IReadOnlyList<AIFunction>`
- Move float inference math (tokenize→forward→argmax→softmax) from
  `samples/NivaraChat/Helpers/ModelInferenceHelper.cs` into an internal
  `NivaraInference` helper in Extensions so the factory is self-contained.
- Each factory captures model/tokenizer (trusted state) and calls
  `AIFunctionFactory.Create` over a `[Description]`-decorated delegate; default
  names `analyzeSentiment`/`extractEntities`/`validateResponse`; returns JSON string.

### 2. Sample (NivaraChat)
- `NivaraToolFunctions` becomes a thin adapter delegating to `NivaraAIFunctions`
  (or removed in favor of calling the factory directly).
- Qwen mode: add `tools-nivara` sub-mode + `--nivara-tools` subset (default `all`).
  Load all three models from `samples/data/nivarachat`, `Initialize`, register
  selected subset, `knownToolNames` = registered names, cap 3.
- Keep `tools-weather`, `chat`, `plain` intact.

### 3. Tests
- Unit test calling `NivaraAIFunctions.CreateSentimentTool(...).InvokeAsync(...)` directly.
- `QwenToolsNivaraLoopTests` end-to-end (sentiment semantic pin; others smoke).

### 4. Docs
- `docs/research/AGENT-FRAMEWORK.md`, `samples/NivaraChat/README.md`, `docs/QWEN.md`.

## Verification
- `dotnet build Nivara.slnx`
- targeted tests: `dotnet test ... --filter "FullyQualifiedName~NivaraAIFunctions|QwenToolsNivara|QwenToolsWeather"`

## Planned commits
1. docs: plan #383 in TODO.md
2. feat(extensions): add NivaraAIFunctions building blocks
3. refactor(sample): NivaraToolFunctions delegates to factory
4. feat(sample): --qwen tools-nivara sub-mode + subset selection
5. test: NivaraAIFunctions unit + QwenToolsNivaraLoop
6. docs: update AGENT-FRAMEWORK/QWEN/README
7. docs: remove TODO.md — plan executed

## Blast radius
- New public API in `Nivara.Extensions` (`Nivara.AI.NivaraAIFunctions`) — additive.
- `NivaraToolFunctions` adapter keeps Ollama modes working.
- No core `src/Nivara` changes.

## GitHub issues log
- (none yet)
