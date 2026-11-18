# Plan: Issue #1325 — align KV-cache default (true) for `--tinyshakespeare`

## Problem

`BatchedChatClient` (CPU path) defaults `useKvCache = false`, and `TransformerOptions.UseKvCache`
defaults `false`, while the SmolLM/Qwen/Silk clients default to `true`. A plain
`--tinyshakespeare` invocation therefore runs the slow non-cached path, inconsistent with
`--smollm`/`--qwen`.

## Proposed changes

1. `samples/NivaraChat/Transformer/BatchedChatClient.cs` — ctor default `useKvCache = false` → `true`.
2. `samples/NivaraChat/Transformer/TransformerMode.cs` — `TransformerOptions.UseKvCache` default `false` → `true`;
   help text updated to cached-by-default style; `--kv-cache`/`--no-kv-cache` both still accepted.
3. `samples/NivaraChat/ARCHITECTURE.md` — CLI list updated.
4. `samples/NivaraChat/README.md` — tested-examples note (optional).

No parsing/execution logic changes; `RunSilk`/`RunSamples`/`RunDiDemo` already forward `UseKvCache`.

## Blast radius

- `BatchedChatClient` ctor default (internal, sample assembly). Callers in `TransformerMode.cs`
  pass explicitly, so runtime behavior for `--tinyshakespeare` flips to cached by default.
- `TransformerOptions` default field. Interactive menu (`RunInteractive`) constructs options without
  setting `UseKvCache`, so it now defaults cached (consistent with SmolLM interactive menu).
- Tests: none reference these defaults directly; `InferenceCliArgumentTests` (NivaraInference) unaffected.

## Verification

- `dotnet build Nivara.slnx`
- `dotnet test tests/Nivara.Tests/Nivara.Tests.csproj -c Release --filter "FullyQualifiedName~InferenceCliArgument"`
- `dotnet run --project samples/NivaraChat -- --tinyshakespeare --help`

## Planned commits

1. `docs: plan #1325 in TODO.md`
2. `fix: default KV cache on for BatchedChatClient and TransformerMode`
3. `docs: update help/ARCHITECTURE/README for cached-by-default tinyshakespeare`
4. `docs: remove TODO.md — plan executed`

## GitHub issues log

(none created yet)
