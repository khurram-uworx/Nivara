# Plan: Issue #1313 — NivaraChat: add Silk/OpenCL inference runner for `BatchedTransformer`

## Problem
There is no GPU path for `--tinyshakespeare`. Need an inference-only Silk runner mirroring `SilkSmolLMGpuRunner`/`SilkSmolLMSession`: upload weights, run prefill + cached decode, argmax/logits. Training stays on CPU AutoDiff.

## Proposed changes

### New: `src/Nivara.Extensions/Silk/SilkTinyShakespeareGpuRunner.cs`
F32 only. `SilkTinyShakespeareGpuRunner(SilkRuntime, TinyShakespeareConfig, Dictionary<string,(float[],int[])>)`.

Weight upload (keys from `BatchedTransformer.ToSilkWeights()`):
- `tok_emb.weight` + transposed copy (`tokEmbTransposed`) for tied LM head.
- Sinusoidal position table `posEmb` (`maxSeqLen * hidden`), computed on CPU, uploaded once.
- `final_norm.weight`/`bias`.
- Per layer (`layers.{i}.*`): `ln1/ln2` weight+bias, bias-free transposed `q/k/v/o` weights, `mlp_fc1` transposed weight+bias, `mlp_fc2` transposed weight+bias.

Workspaces sized to `seqLen`: x, normed, q, k, v, context, projected, gate, posBuf, idsBuf, posIdsBuf.
Device KV cache per layer: `kCache/vCache` (`maxSeqLen * hidden`).

Methods:
- `ForwardPrefill(int[])`: gather tok emb + add pos emb; per layer LayerNorm×2, bias-free GEMMs, `CausalGqaAttention`, GELU MLP (`GemmGelu`+`Gemm`/`GemmBias`), residual adds, capture K/V, final LayerNorm, tied LM head GEMM on last row. Returns last-row logits (`vocab`).
- `ForwardCached(int,int)`: single row, append K/V at position, `DecodeGqaAttentionOproj`, manual MLP, final norm + LM head. Returns logits.
- `ResetCache`, `CachedLength`, `DeviceName`, `Dispose`.

### New: `src/Nivara.Extensions/Silk/SilkTinyShakespeareSession.cs`
Mirrors `SilkSmolLMSession`: constructor + `Open`, `DeviceName`, `ForwardPrefill`, `ForwardCached`, `ResetCache`, `Dispose`.

### Edit: `samples/NivaraChat/Transformer/TransformerMode.cs`
- Add `UseSilk` to `TransformerOptions`; parse `--silk`.
- In `Execute`, if `UseSilk`: build `TinyShakespeareConfig`, `model.ToSilkWeights()`, open session, print device, run KV-cache-aware generation loop using `TokenSampling.Sample`, print text + timing. Else unchanged CPU path (`RunSamples`/`RunDiDemo`).

### Edit: `samples/NivaraChat/Program.cs`
- Add `--silk` to `--tinyshakespeare` help line (optional).

## Config
`TinyShakespeareConfig` record: `HiddenSize`, `NumHiddenLayers`, `NumAttentionHeads`, `VocabSize`, `MaxSeqLen`, `LayerNormEps` (1e-5).

## Verification
- `dotnet build Nivara.slnx` (Release) including samples.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Manual: `--tinyshakespeare --silk --load <model>` produces text and reports device name.
- Keep CPU path unchanged.

## Planned commits
1. docs: plan issue #1313 in TODO.md
2. feat(silk): add SilkTinyShakespeareGpuRunner (f32)
3. feat(silk): add SilkTinyShakespeareSession
4. feat(nivarachat): wire --tinyshakespeare --silk
5. docs: remove TODO.md — plan executed

## Blast radius
- New files in `src/Nivara.Extensions/Silk/` (no core changes).
- `samples/NivaraChat/Transformer/TransformerMode.cs` (parse + silk branch).
- `ToSilkWeights()` already exists on `BatchedTransformer`; reused.
- No change to `BatchedChatClient` or DI demo.

## GitHub issues log
- (none yet)
