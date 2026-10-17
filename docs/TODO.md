# Plan: SmolLM KV-cached decode on GPU (Silk) — issue #445 Slice 4

## Problem
The Silk SmolLM runner (`SilkSmolLMGpuRunner`) is prefill-only: every generated token re-runs the full
prefix, so generation is O(n²). Issue #445 Slice 4 replaces this with a KV-cached decode path on the
proven Silk/OpenCL backend, mirroring the CPU `LlamaForCausalLM` `ForwardPrefill`/`ForwardCached`
contract and the Qwen `GenerateCore` flow.

## Proposed changes

1. **Decode-attention kernel** (`src/Nivara.Extensions/Silk/SilkKernels.cs`): add
   `decode_gqa_attention` — single query row attending j=0..kvLen-1 over cached K/V, GQA repeat.
   Add constant, source, `KernelNames` entry, and `SilkLauncher.DecodeGqaAttention`.
2. **Runtime copy** (`SilkRuntime`): add `CopyBuffer(src,dst,srcOff,dstOff,count)` via
   `cl.EnqueueCopyBuffer` for device-side cache population.
3. **Runner cache state** (`SilkSmolLMGpuRunner`): per-layer `kCache`/`vCache` (`maxPos*kvDim`),
   `cachedLen`; `ResetCache`, `Prefill(int[])` (populates cache, returns last-row logits),
   `Decode(int tokenId,int position)` (append + attend, returns logits).
4. **Session API** (`SilkSmolLMSession`): `ForwardPrefill`, `ForwardCached`, `ResetCache`.
5. **Generation loop** (`Program.cs`): rewrite `RunSmolLMGenerationSilk` to prefill once then decode;
   keep EOS-append semantics identical to CPU `RunSmolLMGeneration`.
6. **Stage A (CPU)**: switch `RunSmolLMGeneration<T>` to `ForwardPrefill`/`ForwardCached` for parity/perf.

## Verification
- Build Extensions + Inference (Release).
- Parity: cached Silk token stream matches CPU cached token stream on the fixed prompt.
- `dotnet format ... --verify-no-changes`.

## Planned commits
1. docs: plan #445 in TODO.md
2. feat(silk): add decode_gqa_attention kernel + launcher
3. feat(silk): add device-side CopyBuffer to SilkRuntime
4. feat(silk): add KV cache state + Prefill/Decode to SilkSmolLMGpuRunner
5. feat(silk): expose ForwardPrefill/ForwardCached on SilkSmolLMSession
6. feat(sample): switch smollm silk generation to cached path
7. feat(sample): switch CPU smollm generation to cached path
8. docs: remove TODO.md — plan executed

## Blast radius
- SilkKernels.cs, SilkLauncher.cs, SilkRuntime.cs, SilkSmolLMGpuRunner.cs, SilkSmolLMSession.cs,
  Program.cs (smollm silk + CPU generation). Tests: InferenceCliArgumentTests (no mode changes expected).

## GitHub issues log
- (none yet)