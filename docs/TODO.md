# Plan: NivaraChat shared causal-LM generation loop (issue #1327)

## Problem
Six chat clients (`BatchedChatClient`, `SilkBatchedChatClient`, `SmolLMChatClient`,
`SilkSmolLMChatClient`, `QwenChatClient`, `SilkQwenChatClient`) reimplement the
same autoregressive loop: prefill -> sample -> append -> (on cache overflow) reset
cache with tail window -> cached step, plus a non-cached fallback. Overflow handling
is copy-pasted and a drift risk.

## Proposed changes
Shared, sample-only (no core API change):

1. `samples/NivaraChat/Shared/CausalLmOps.cs` — `readonly record struct` bundle of
   delegates + scalars: `Prefill`, `Cached`, `LastLogits`, `ResetCache`, `VocabSize`,
   `EosTokenId`, `MaxSequenceLength`.
2. `samples/NivaraChat/Shared/CausalLmGenerator.cs` — single static
   `Generate(CausalLmOps, ...)` covering cached/non-cached paths, sampling, overflow
   (`overflowKeep`), and Qwen's context-limit guard (`breakOnContextLimit`).
3. `samples/NivaraChat/Shared/CausalLmChatClientBase.cs` — abstract `IChatClient`
   base holding common state and plumbing, with hooks for prompt render/encode/decode/
   ops-build/post-process.
4. Migrate the six clients to subclass the base and delegate generation to
   `CausalLmGenerator`.

## Per-client knobs
| Client | windowPrompt | windowNonCached | overflowKeep | breakOnContextLimit |
|---|---|---|---|---|
| BatchedChatClient | true | true | MaxSeqLen-1 | false |
| SilkBatchedChatClient | true | true | maxSeqLen-1 | false |
| SmolLMChatClient | false | false | 0 | false |
| SilkSmolLMChatClient | false | false | 0 | false |
| QwenChatClient | false | false | 0 | true |
| SilkQwenChatClient | false | false | 0 | true |

## Verification
- Build sample: `dotnet build samples/NivaraChat/NivaraChat.csproj -c Release`
- Smoke-run Transformer/SmolLM/Qwen (and Silk variants).
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`

## Planned commits
1. docs: plan #1327 in TODO.md
2. feat: add CausalLmOps + CausalLmGenerator
3. feat: add CausalLmChatClientBase
4. refactor: migrate BatchedChatClient + SilkBatchedChatClient
5. refactor: migrate SmolLMChatClient + SilkSmolLMChatClient
6. refactor: migrate QwenChatClient + SilkQwenChatClient
7. docs: remove TODO.md — plan executed

## Blast radius
- Only `samples/NivaraChat`. No core `Nivara` API change.
- Chat clients consumed by `TransformerMode`, `SmollmMode`, `QwenMode`.

## GitHub issues log

- (none yet)
