# Plan: NivaraChat cleanup — issues #1307, #1308, #1309

Umbrella: #1270 (TinyShakespeare transformer has no Silk/GPU path). Phase A cleanup.

## Problem

- `BatchedTransformerBlock<T>` accepts a `dropout` constructor argument and stores nothing for it; no dropout is applied anywhere in `ForwardBlock`. The parameter is misleading and complicates the upcoming Silk runner.
- `BatchedChatClient.Sample` hand-implements temperature softmax + argmax, duplicating `TokenSampling.Sample` (used by SmolLM/Qwen), which also supports top-p nucleus filtering and a seeded RNG.
- Top-p is not exposed on the `--tinyshakespeare` CLI.

## Proposed changes

### #1307 — remove unused `dropout` parameter
- `samples/NivaraChat/Transformer/BatchedTransformer.cs`: drop `double dropout` from `BatchedTransformer` and `BatchedTransformerBlock` constructors; update `new BatchedTransformerBlock<T>(...)` call.
- `samples/NivaraChat/Transformer/TransformerMode.cs`: remove `--dropout` parse case, `dropout:` arg, `Dropout` field, help line, and `dropout=` status text.
- `samples/NivaraChat/ARCHITECTURE.md`: remove `--dropout` CLI line.

### #1308 — replace hand-rolled sampling with `TokenSampling`
- `samples/NivaraChat/Transformer/BatchedChatClient.cs`: add `topP` field (default 1.0f); add `topP` constructor param; replace `Sample` body with `TokenSampling.Sample(logits, logits.Length, temperature, topP, rng)`; remove `ArgMax`. Add `using NivaraChat.Shared;`.

### #1309 — add `--top-p` support
- `samples/NivaraChat/Transformer/TransformerMode.cs`: add `TopP` field (default 1.0f); parse `--top-p`; pass to both `BatchedChatClient` call sites; add help line.
- `samples/NivaraChat/ARCHITECTURE.md`: add `--top-p` CLI line.
- Optional: `Program.cs` top-level help list.

## Blast radius
- Only `samples/NivaraChat` consumers; no core library or tests reference these constructors directly (grep confirmed). Test host compiles the sample project, so build must stay green.

## Verification steps
- `dotnet build samples/NivaraChat/NivaraChat.csproj`
- `dotnet build tests/Nivara.Tests/Nivara.Tests.csproj`
- `dotnet run --project samples/NivaraChat -- --tinyshakespeare --help` (no `--dropout`, lists `--top-p`)
- Smoke run with `--top-p 0.9`.

## Planned commits
1. docs: plan cleanup in TODO.md
2. #1307 remove dropout parameter
3. #1308 replace sampling with TokenSampling
4. #1309 add --top-p support
5. docs: remove TODO.md (after G2)

## GitHub issues log
- (none created yet; deferred work recorded here if found)
