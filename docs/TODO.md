# Plan — Issue #375: SmolLM IChatClient follow-ups (KV cache + temperature/top-p sampling)

## Problem

Stage A of the SmolLM-as-`IChatClient` work in `samples/NivaraChat` (`--smollm chat/plain`)
serves SmolLM-135M-Instruct through `Microsoft.Extensions.AI.IChatClient`. Two known
follow-ups would noticeably improve the chat REPL:

1. **No KV cache.** Each generated token re-runs the full `LlamaForCausalLM.Forward()` over
   the entire token sequence — ~L² attention work per token across 30 layers.
2. **No sampling.** Greedy argmax only. Temperature/top-p sampling would reduce repetitive
   multi-turn replies (SmolLM instruct recommends `temperature=0.6`-ish).

Keep the plain **greedy** path intact (used by `NivaraInference` numeric diffing) and keep
Stage B/C (separate branches) unaffected.

## Proposed changes

### Core (minimal, backward-compatible)
- `src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs`
  - Add optional `int positionOffset = 0` to `Forward(ReverseGradTensor<T> input)`.
    The RoPE cos/sin cache currently indexes by absolute position (`p * (headDim/2)`).
    Offset slices the cache to start at `positionOffset` so a single new token at absolute
    position `P` gets RoPE for position `P`. Default `0` preserves existing behavior.
- `src/Nivara/AutoDiff/Nn/LlamaCausalAttention.cs`
  - Add a cache-aware single-token forward (default `Forward` untouched):
    - Take `input` `[1, D]`, `positionOffset`, and per-layer K/V caches (arrays of rows).
    - Q/K/V projections for the one new position → RoPE Q and K with the offset.
    - Append RoPE'd K (`[1, numKVHeads*headDim]` and V) to the layer caches.
    - GQA-repeat new K/V, run
      `MultiHeadAttention(newQ[1,D], fullK[kvLen,D], fullV[kvLen,D], numHeads, scale, openMask)`
      where `openMask = ModuleHelpers<T>.CreateCausalMask(1, kvLen)` (fully-open `[1, kvLen]`).
  - Keep values numerically identical to the non-cached full `Forward`.

### Sample (primary home of the KV-cache path, per the issue)
- `samples/Nivara.Samples/LlamaForCausalLM.cs`
  - Add a KV-cache generation API:
    - A small cache holder (per-layer `T[]` K/V buffers, growing by `numKVHeads*headDim` per
      token, or `ReverseGradTensor<T>` per row).
    - An initial prompt pass (full `Forward`) to seed the cache and produce the first logits.
    - Per-token forward reusing the new cache-aware attention path.
    - Public flag/method to enable the cache; the plain greedy `Forward(int[])` stays.
- `samples/NivaraChat/SmolLM/SmolLMChatClient.cs`
  - Add optional `float? temperature`, `float? topP`, `int? seed` ctor args.
  - Default greedy when `temperature` null/`0`; otherwise temperature softmax + optional
    top-p filtering + cumulative sampling over the last-row logits (reuse the proven
    `BatchedChatClient.Sample` pattern, `lock (rng)` for thread safety).
    - If `temperature` is null/undefined → **greedy**; `>0` → sampling.
  - Route generation through the KV-cache path when enabled.
- `samples/NivaraChat/SmolLM/SmollmMode.cs`
  - Add `--temperature <float>`, `--top-p <float>`, `--kv-cache` / `--no-kv-cache` options;
    wire into `SmollmOptions` + `ParseArgs` + `PrintHelp`; pass through to
    `SmolLMChatClient<T>`.

### Tests
- `tests/Nivara.Tests/AutoDiff/` (existing `LlamaCausalLMTests.cs` + new
  `LlamaCausalKVCacheTests.cs`):
  - KV-cache logits per step == non-cached `Forward(int[])` logits (greedy) — the key
    correctness guarantee mirroring `NivaraInference` diffing.
  - Greedy default unchanged.
  - Sampling: varied tokens across different seeds; deterministic for same seed.
  - top-p never selects a token past the nucleus cutoff.
- These live in `tests/Nivara.Tests` which already references `Nivara.Samples`.

### Docs
- Update `samples/NivaraChat/README.md` SmolLM section (new options, KV cache, sampling).

## Blast radius

- **Core:** `RotaryEmbedding.Forward` (new optional param — default 0, backward compatible)
  and `LlamaCausalAttention` (new method — existing `Forward` untouched).
  - Downstream callers of `RotaryEmbedding.Forward`: `LlamaCausalAttention.Forward`,
    `tests/Nivara.Tests/.../RotaryEmbeddingTests.cs`, NivaraTorch fixtures.
  - Tests covering these: `AutoDiff/RotaryEmbeddingTests.cs`, `NivaraTorch/RotaryEmbeddingTests.cs`,
    `AutoDiff/LlamaCausalLMTests.cs`, `NivaraTorch/LlamaDecoderBlockTests.cs`.
- **Sample:** `LlamaForCausalLM.cs`, `SmolLMChatClient.cs`, `SmollmMode.cs` — these are
  standalone (NivaraChat sample + Nivara.Samples).
- **Tests:** additive — new file + a couple of additions to `LlamaCausalLMTests`.
- No change to the existing greedy public `Forward(int[])` contract → `NivaraInference`
  numeric diffing unaffected.

## Verification steps

- `dotnet build Nivara.slnx` (after each logical step).
- Targeted tests for RotaryEmbedding / LlamaCausalLM / KV cache (ask before `dotnet test`).
- Manual: `--smollm chat --temperature 0.6` with and without KV cache; compare tokens/sec
  and reply variety.

## Planned commits

1. `docs: plan #375 (SmolLM KV cache + sampling) in TODO.md`
2. `Add positionOffset to RotaryEmbedding.Forward`
3. `Add cache-aware forward to LlamaCausalAttention`
4. `Add KV-cache generation path to LlamaForCausalLM`
5. `Add temperature/top-p sampling to SmolLMChatClient`
6. `Add --temperature/--top-p/--kv-cache options to --smollm mode`
7. `Add KV-cache and sampling tests`
8. `docs: SmolLM KV-cache + sampling options in NivaraChat README`
9. `docs: remove TODO.md — plan executed`

## GitHub issues log

- [ ] #375 — SmolLM IChatClient follow-ups: KV cache + temperature/top-p sampling (this plan)
