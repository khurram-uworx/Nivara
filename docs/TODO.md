# Plan: NivaraChat KV cache for BatchedTransformer (#1310)

## Problem

`BatchedChatClient.Generate` re-runs the full `[1, L]` forward for every generated token (`ForwardLastRow`). Add a cached generation path to `BatchedTransformer<T>` mirroring SmolLM/Qwen: `ForwardPrefill`, `ForwardCached`, and `TransformerKVCache<T>`, plus a `--kv-cache` flag (default off).

## Proposed changes

### 1. `PositionEncoding.cs`
- Add `Build(batch, startPosition, seqLen, embedDim)` overload; original delegates to `startPosition=0`.

### 2. `BatchedTransformer.cs`
- Add `NumLayers`, `NEmbd` properties.
- Add `KVCache`/`Cache`/`LastKVCache` properties.
- Add `ForwardPrefill(ReverseGradTensor<T> tokens)` and `ForwardPrefill(tokens, cache)` (also `int[]`/`float[]` overloads).
- Add `ForwardCached(int tokenId, int position)` and `ForwardCached(tokenId, position, cache)`.
- Add `ForwardPrefill`/`ForwardCached` methods to `BatchedTransformerBlock` that capture K/V per layer and run causal (prefill) / full-cached (decode) attention.

### 3. New `TransformerKVCache<T>`
- Fixed `[numLayers][maxSeqLen * nEmbd]` K/V buffers.
- Constructors, `Reset`, `Ensure`, `Capacity`, `MaxCapacity`, `IDisposable`.
- Expose lowercase `keys`/`values` (Llama-style) and PascalCase aliases.

### 4. `BatchedChatClient.cs`
- Add `useKvCache` ctor param (default false).
- Cached path in `Generate`/`GenerateStreaming`.

### 5. `TransformerMode.cs`
- Parse `--kv-cache`; thread through client construction; update help.

## Verification
- Build NivaraChat (Release).
- Run targeted tests if added.
- Manual smoke with `--tinyshakespeare --kv-cache`.

## Planned commits
1. docs: plan #1310 in TODO.md
2. feat: position encoding offset overload
3. feat: TransformerKVCache
4. feat: cached forward path in BatchedTransformer
5. feat: --kv-cache in BatchedChatClient
6. feat: --kv-cache CLI flag
7. docs: remove TODO.md

## GitHub issues log
- (none yet)