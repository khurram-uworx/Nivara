# Plan: issue #1324 — NivaraChat SilkBatchedChatClient for --tinyshakespeare

## Problem

`--smollm` and `--qwen` each ship a `Silk*ChatClient : IChatClient` (streaming, KV-cached, top-p) that lets the Silk path plug into `--agents`, `--interactive`, `--workflow`, and the DI demo. `--tinyshakespeare --silk` does not: `TransformerMode.RunSilk` drives `SilkTinyShakespeareSession` by hand and only prints. As a result the Silk path cannot be used as a chat client, has no streaming output, and duplicates the overflow/cache logic already present in `SilkSmolLMChatClient`.

## Proposed changes

### New file `samples/NivaraChat/Transformer/SilkBatchedChatClient.cs`

`IChatClient` over `SilkTinyShakespeareSession`, mirroring `SilkSmolLMChatClient`:
- Constructor `(SilkTinyShakespeareSession session, TextTokenizer tokenizer, int maxSeqLen, int maxNewTokens=96, float? temperature=null, float? topP=null, int? seed=null, bool useKvCache=true)`.
- `GetResponseAsync` (single-shot) and `GetStreamingResponseAsync` (token streaming).
- KV-cached generation by default, `--no-kv-cache` opt-out (non-cached path re-projects full window via `ForwardPrefill`).
- top-p / temperature via shared `TokenSampling`.
- cache-overflow handling: keep tail window of `maxSeqLen-1` when `position >= maxSeqLen`.
- `Dispose` disposes the session.
- `FormatConversation` identical to `BatchedChatClient` for CPU parity.

### `samples/NivaraChat/Transformer/TransformerMode.cs`

- `ParseArgs`: add `--no-kv-cache` (sets `UseKvCache = false`); keep `--kv-cache`.
- `Execute`: when `UseSilk`, call new `RunSilkChat` instead of `RunSilk`.
- `RunSilkChat`: build `TinyShakespeareConfig`, open session, construct `SilkBatchedChatClient`; if `--text` provided stream via `GetStreamingResponseAsync`; otherwise run samples + DI demo (DI demo registers client through `AddChatClient`).
- `PrintHelp`: document `--no-kv-cache` / `--kv-cache`.
- Keep old `RunSilk` method (unused fallback) or remove; prefer replacing call site.

### No changes to CPU `BatchedChatClient`, `SilkTinyShakespeareSession`, runner.

## Verification steps

- `dotnet build Nivara.slnx`
- `dotnet run --project samples/NivaraChat -- --tinyshakespeare --silk --text "ROMEO:"`
- `dotnet run --project samples/NivaraChat -- --tinyshakespeare --silk`
- `dotnet run --project samples/NivaraChat -- --tinyshakespeare --silk --no-kv-cache --text "ROMEO:"`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~SilkTinyShakespeare"`

## Planned commits

1. docs: plan issue #1324 in TODO.md
2. feat(nivarachat): add SilkBatchedChatClient
3. feat(nivarachat): wire --tinyshakespeare --silk to SilkBatchedChatClient
4. docs: help/args for --no-kv-cache
5. docs: remove TODO.md — plan executed

## Blast radius

- New client depends on `SilkTinyShakespeareSession` (ForwardPrefill/ForwardCached/ResetCache) and `TextTokenizer`.
- `TransformerMode` changes only Silk branch; CPU path untouched.
- No public library API changes (sample only).
- DocsCoveragePins: new .cs file not a doc; no pin changes expected. Re-check after.

## GitHub issues log

- (none yet)
