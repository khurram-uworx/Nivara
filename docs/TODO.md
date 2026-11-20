# Plan: TinyShakespeare Silk training (Gap S6 / #1342)

**Branch:** khurram/tiny-shakespeare
**Status:** in progress

## Problem

`--tinyshakespeare` training runs entirely on the CPU AutoDiff path
(`BatchedTransformer`, `ReverseGradOperations.BatchedMultiHeadAttention`, `Adam<float>`).
Silk is inference-only for this mode (`SilkTinyShakespeareSession`,
`SilkBatchedChatClient` via `ToSilkWeights()`). With Silk promoted to first-class support,
training should gain a device path while the CPU AutoDiff path stays the reference parity gate.
f32 only this release.

## Decisions (confirmed)

- Reuse `--silk` to mean "train on device when training" (and infer on Silk when generating).
- Trainer ships primitives + convenience `Train(...)` loop.
- Parity tolerance 1e-5 (logits / gradients / weights); argmax exact.
- Caps maxBatch=32, maxSeqLen=64.

## Proposed changes

### 1. Core trainer — `src/Nivara.Extensions/Silk/SilkTinyShakespeareTrainer.cs` (new)

Mirror `SilkLayaHeadTrainer` / `SilkDistilBertTrainer`. No new OpenCL kernels required.

- Constructor: `SilkRuntime`, `TinyShakespeareConfig`, weight dict (`ToSilkWeights()` shape), `maxBatch=32`, `maxSeqLen=64`. Allocates f32 weight/grad/Adam m,v buffers for `tok_emb`, `final_norm` (w,b), and per-layer `ln1/ln2` (w,b), `q/k/v/o`, `fc1/fc2` (+bias). Precompute sinusoidal `posEmb`. `EnsureAttentionLocalMemory(headDim)`.
- Tied weights: one canonical `tokEmb` buffer; `tokEmbTransposed` refreshed on host after each Adam step. Accumulate both `EmbeddingBackwardRows` (input-token grad) and LM-head weight grad (`GemmTn`) into the same `tokEmb` grad buffer before `AdamUpdate`.
- Per-sample batching loop inside Forward/Backward (causal-attention kernel is single-sequence); gradients scaled by `1/(B*L)`. Activations saved per layer.
- Surface: `Forward(int[] inputTokens, int seqLen)` → logits (host); `Backward(int[] targetTokens, int seqLen, float scale)`; `ZeroGradients()`; `AdamUpdateAll(lr, wd, beta1, beta2, eps)`; `ExportCheckpoint()`; `ExportGradients()`; `Train(...)` convenience loop; `Dispose()`.

### 2. Sample import — `samples/NivaraChat/Transformer/BatchedTransformer.cs`

Add `LoadFromSilkWeights(Dictionary<string,(float[] Data,int[] Shape)>)` copying exported f32 arrays into CPU params (f32→T via `float.CreateChecked`), validating keys/shapes.

### 3. Wire `--silk` into training — `TransformerMode.cs`

When `UseSilk` and not loading: instantiate runtime+trainer, `Train(...)`, `ExportCheckpoint()` → `model.LoadFromSilkWeights(...)`, dispose. Otherwise CPU `Train`. Update backend reporting + help text (f32).

### 4. Parity tests — `tests/Nivara.Tests/Gpu/SilkTinyShakespeareTrainerTests.cs` (new)

Seeded tiny fixture (vocab 32, hidden 16, 1 layer, 2 heads, seq 8, batch 4). Compare CPU AutoDiff one step vs Silk: logits, loss, gradients, post-step weights; save/load round-trip. Tolerance 1e-5, argmax exact. `[Category("Gpu")]`, ignore when no device.

### 5. Docs

- `ARCHITECTURE.md` D-016 → inference + training (f32).
- `docs/plan/NIVARACHAT-SILK-GAPS.md` → mark Gap S6 implemented.
- `README.md` → `--silk` covers training.

## Verification steps

- Build `Nivara.slnx` (Release).
- Targeted tests: new parity tests + existing `SilkTinyShakespeareParityTests`.
- `dotnet format Nivara.slnx whitespace --verify-no-changes --no-restore`.
- Snippet gate if docs change.

## Planned commits

1. `SilkTinyShakespeareTrainer` primitives (weights, forward, backward, adam, export, dispose)
2. `Train(...)` convenience loop + host CE-loss reporting
3. `BatchedTransformer.LoadFromSilkWeights` + round-trip helper
4. Wire `--silk` training path in `TransformerMode` + backend reporting + help
5. Parity + round-trip tests
6. Docs (D-016, gaps, README)

## Blast radius

- New core trainer in `src/Nivara.Extensions/Silk/` (public API addition).
- Sample `BatchedTransformer` gains import method.
- `TransformerMode` training branch.
- Tests under `tests/Nivara.Tests/Gpu/`.
- Docs pins: document count unchanged; citation count may change if citations added.

## GitHub issues log

- (none yet)
