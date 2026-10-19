# Plan: Issue #658 Milestone 3 — cache-friendly SmolLM decode hygiene

Branch: `khurram/658-milestone-3`
Base: `khurram/658-milestone-2`

## Problem

Milestones 1 and 2 improved GEMM and attention, but cached decode still grows
KV buffers by doubling and repeats weight-span lookups. Milestone 3 is hygiene:
optional up-front cache sizing and reduced per-step churn, without changing
numerical behavior.

## Proposed changes

1. Add optional `maxCapacity` to `LlamaKVCache<T>`:
   - If > 0, allocate all layer buffers up front and set fixed capacity.
   - If <= 0, retain existing doubled-growth behavior.
   - `Ensure` validates bounds in fixed mode.
   - Expose `Capacity`.
2. Store `maxPositionEmbeddings` in `LlamaForCausalLM<T>` and add a
   `PrepareCache` helper for opt-in up-front sizing.
3. Optionally cache weight spans in `LlamaDecoderBlock` fused scratch if
   parameter storage is stable; otherwise keep `WeightSpan`.
4. Audit fused paths for `FlattenTo`, allocation churn, and virtual dispatch.

## Blast radius

- `samples/Nivara.Samples/LlamaForCausalLM.cs`
- `src/Nivara/AutoDiff/Nn/LlamaDecoderBlock.cs`
- Llama KV/decoder/prefill/LM tests.

## Verification steps

1. `dotnet build Nivara.slnx`
2. `dotnet test -c Release --filter "FullyQualifiedName~LlamaCausalKVCacheTests"`
3. `dotnet test -c Release --filter "FullyQualifiedName~LlamaDecoderBlockTests|LlamaForCausalLMPrefillTests|LlamaCausalLMTests|LlamaCausalAttentionTests"`
4. `dotnet test -c Release --filter "FullyQualifiedName~EveryGatedBlock_CompilesWithoutErrors"`
5. `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`
6. Human-run SmolLM benchmark/generate.

## Planned commits

1. `docs: plan #658 milestone 3 decode hygiene in TODO.md`
2. `feat(llama): add opt-in preallocated LlamaKVCache capacity`
3. `refactor(llama): cache fused weight spans in decoder blocks`
4. `test(llama): cover preallocated cache behavior`

## GitHub issues log

- #658 — parent: CPU tiled multithreaded GEMM and cache-friendly SmolLM decode
- #458 — CPU-side performance deficit tracker