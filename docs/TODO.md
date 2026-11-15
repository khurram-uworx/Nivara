# Plan: #916 close-out + NivaraChat OpenAI-compatible transport

## Problem

#916 (Nivara Adaptive Intelligence umbrella) has all four children (#980-#983) closed and the
core loop implemented (`--rag-adaptive`, `--rag-adaptive-eval`). Two things block an honest close:

1. The umbrella acceptance criterion "Evaluation: ... training time, CPU vs Silk" is not met.
   `RagAdaptiveEvalMode.cs:73-76` stops a stopwatch immediately (`cpuBuildMs = 0`), the Silk build
   time is never recorded, and `AdaptiveReportTiming("head updates", saves, 0)` reports 0 ms.
   The committed smoke artifact (`samples/data/nivarachat/adaptive/rag-adaptive-report.md`) shows
   exactly those zeros. `docs/RAG.md:241` claims timings split head updates and persistence, which is
   false today.
2. The #983 slice shipped with dead code and a no-op flag:
   `RagAdaptiveEvalMode.cs:207-209` (empty temp-file loop), `213-224` (scratch head with
   `headLayerCount: 0` then disposed), `369-380` (unused `accepted` + a "-> change signature"
   planning comment), and `--no-persist` is parsed into `ModeContext` but never read.

Plus, evidence for the umbrella's decision metrics requires judge labels from an LLM, and the repo
only supports Ollama. The user's environment no longer has Ollama Cloud, so an OpenAI-compatible
transport is required before a real run is possible.

## Scope decisions

- Close-out fixes are samples + docs only. No `src/Nivara` changes. The adaptive work already
  added one additive Extensions API (`LayaDecisionHead.ForwardChoiceLogits`, `d8d1b45d`); this plan
  adds none.
- OpenAI transport is its own issue/PR (separate commit stream), so #916's close-out stays
  scoped to honest timing/docs.
- Silk remains MiniLM-embeddings-only (no hand-written weight step) — filed as a follow-up,
  not a blocker.
- ONNX export stays deferred (umbrella Phase 7 stretch).

## Work items

### A. OpenAI-compatible transport for NivaraChat (new issue, prerequisite for evidence)

New issue title: "NivaraChat: OpenAI-compatible chat transport alongside Ollama".

- `samples/NivaraChat/Shared/OpenAiCompatibleChatClient.cs`: `IChatClient` over a shared
  `HttpClient`, POST `{base}/chat/completions`, non-streaming, temperature + `ModelId` from
  `ChatOptions`, errors map to status + body snippet. No silent fallbacks.
- `samples/NivaraChat/Shared/LlmClientFactory.cs`: single seam replacing the ten
  `new OllamaApiClient(...)` sites.
- Flags: `--llm ollama|openai` (default `ollama`; `--ollama` keeps its meaning),
  `--openai-url <base>`, `--openai-model <name>`, `--openai-auth bearer|api-key`
  (default `bearer`; `api-key` for Azure OpenAI). Key only from `NIVARA_CHAT_API_KEY` env var —
  never a CLI flag, never logged.
- Wire plain-chat modes: `--rag`, `--rag-adaptive`, `--rag-eval`, `--rag-agent`,
  `--handoff`, `--intent`, `--online-learning`, `--critic`, `--workflow`.
  `--agents`/`--tools` reject `--llm openai` with an explicit message (function-calling and
  streaming are a follow-up).
- `RagEvalMode` judge path accepts either provider.
- Tests `OpenAiCompatibleChatClientTests`: fake `HttpMessageHandler` asserts request URI/body
  (model, messages, temperature), Bearer vs `api-key` header, response parsing, non-2xx and
  missing-key errors. No network.
- Docs: `README.md` + `PrintUsage()` rows.

### B. #916 close-out fixes (samples + docs)

1. Timing honesty (`AdaptiveFeedbackCollector`, `RagAdaptiveEvalMode`, `AdaptiveReport`):
   - Collector records `UpdateMilliseconds`/`PersistMilliseconds`.
   - Eval mode times both `MultiViewIndex.BuildAsync` calls.
   - Report stages: `index build (cpu)`, `index build (silk)`, `test feature extraction (before)`,
     `train feature extraction + decisions`, `head updates (AutoDiff)`, `head persistence`,
     `test feature extraction (after)`. Drop the duplicate `total test-pass before` row.
2. Delete dead code + planning chatter in `RagAdaptiveEvalMode.cs`
   (empty temp-file loop, scratch head, unused `accepted`, `-> change signature` comment).
3. Honour `--no-persist` in the collector (skip save; report shows `Persistence: disabled`).
4. `try/finally` dispose of the engine in both adaptive modes.
5. Docs: `docs/RAG.md` (merge duplicate `## 10.`, adaptive before References, flags + run recipe +
   artifacts); `README.md` + `PrintUsage()` rows for adaptive flags and `--llm` flags.
6. Tests: collector timing monotonicity (`>= 0`, accumulates across batches), no-persist writes
   nothing, report renders every stage + disabled-persistence line; keep byte-stability test.
   Structural asserts only — no wall-clock bounds, no `Performance` category.

### C. Evidence run (after A lands, using the user's endpoint)

1. `--rag-eval --regenerate judge --llm openai --openai-url ... --openai-model ... --force`
   -> `samples/data/nivarachat/eval/relevance.tsv` (gitignored).
2. `--rag-adaptive-eval --labels judge --no-persist --head-weights <temp>` at default caps
   (40 queries / 128 examples), CPU.
3. Optional `--silk` rerun for parity + real build timings if OpenCL is available.
4. Quote held-out retrieval table, decisions before/after, calibration, timings, persistence state
   in the #916 close comment. Auto-label run stays labelled a leakage diagnostic, not evidence.

### D. Close-out

- Comment on #916: child map -> commits, AC -> evidence table, stated limits (Silk scoped to MiniLM
  embeddings, feedback simulated from judge labels, `ForwardChoiceLogits` Extensions API, ONNX deferred).
- Close #916; update roadmap #1016 (drop/mark the #916 block); close milestone 1.6.
- File follow-ups: ONNX export of trained head; hand-written Silk weight step; interactive
  human-feedback path; tool/streaming support for the OpenAI transport.

## Grounding (Resistance G1)

- Azure OpenAI v1 API (verified via microsoft-learn MCP): endpoint shape
  `https://<resource>.openai.azure.com/openai/v1/chat/completions`, auth via `api-key:` or
  `Authorization: Bearer`, `api-version` optional. Generic OpenAI-compatible servers use
  `<base>/v1/chat/completions`. One client covers both via configurable base URL + auth mode.
- `Microsoft.Extensions.AI` 11.0.0 is already a dependency; `IChatClient` is the existing seam
  used by every mode, so no new package pins.
- Existing chat clients (`SmolLMChatClient`, `QwenChatClient`, `BatchedChatClient`) establish the
  hand-rolled pattern; the new client follows it.
- Code-memory MCP confirmed: `AdaptiveFeedbackCollector`, `RagAdaptiveEvalMode`, `AdaptiveReport`,
  `ModeContext`, and the ten `OllamaApiClient` construction sites.
- Blast radius: samples/NivaraChat only. `src/Nivara` untouched. Tests under
  `tests/Nivara.Tests/Samples`. No DocsCoveragePins movement expected (no new doc files, no
  `File.cs::Symbol` citations, no new doc cross-links).

## Planned commits (one logical change each)

1. `docs: plan #916 close-out + OpenAI transport in TODO.md`
2. `feat(samples): add OpenAiCompatibleChatClient and LlmClientFactory`
3. `feat(samples): wire --llm openai into plain-chat modes`
4. `feat(samples): accept openai provider in RagEval judge path`
5. `test(samples): cover OpenAiCompatibleChatClient request/response/error handling`
6. `docs(samples): document --llm openai flags and usage`
7. `fix(samples): record real head-update and index-build timings in adaptive eval`
8. `refactor(samples): remove dead code and planning comments from RagAdaptiveEvalMode`
9. `feat(samples): honour --no-persist in adaptive feedback collector`
10. `fix(samples): dispose adaptive engines in try/finally`
11. `test(samples): cover adaptive timing and no-persist behaviour`
12. `docs: fix RAG.md adaptive section numbering and run recipe`
13. `docs(samples): document adaptive flags in README and usage`
14. `docs: remove TODO.md — plan executed` (only after G2 review)

## GitHub issues log

- [ ] New issue — NivaraChat OpenAI-compatible chat transport alongside Ollama (created during A)
- [ ] #916 — parent umbrella (close after C/D)
- [ ] Follow-up — ONNX export of trained Laya head (deferred from #916)
- [ ] Follow-up — hand-written Silk weight step for head training (deferred from #916)
- [ ] Follow-up — interactive human-feedback path for adaptive loop (deferred from #916)
- [ ] Follow-up — tool-call/streaming support on OpenAI-compatible transport (deferred from A)

## Verification steps

- `dotnet build samples/NivaraChat`
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
- `dotnet test tests/Nivara.Tests -c Release --filter "FullyQualifiedName~Nivara.Tests.Samples"`
- `dotnet test tests/Nivara.Tests -c Release --filter "FullyQualifiedName~Nivara.Tests.Docs"` (docs touched)
- Evidence run C (human-assisted, needs endpoint) before closing #916
