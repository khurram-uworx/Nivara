# NivaraChat: TinyShakespeare transformer training on Silk (AutoDiff → device) — design note

**Status:** implemented (PR #1359). Written 2026-11-19; updated 2026-11-20.
**Scope:** `samples/NivaraChat` (training path) and a new `SilkTinyShakespeareTrainer` in
`src/Nivara.Extensions/Silk`.
**Release constraint:** Silk training is **f32 only** this release. No bf16/fp16 training on
Silk is required; inference keeps its existing f32 path.
**Tracking:** issue #1342 (Gap S6 in `docs/plan/NIVARACHAT-SILK-GAPS.md`).

## 1. Motivation

`--tinyshakespeare` training runs entirely on the CPU AutoDiff path
(`BatchedTransformer<T>`, `ReverseGradOperations.BatchedMultiHeadAttention`, `Adam<T>`).
Silk is inference-only for this mode (`SilkTinyShakespeareSession`, `SilkBatchedChatClient`
via `ToSilkWeights()`). With Silk promoted to first-class support, training should gain a
device path while the CPU AutoDiff path stays the **reference parity gate**. This is a large,
open-ended item; this note scopes kernels, memory, precision, save/load, and parity before any
code.

## 2. Reference (CPU) semantics to match

Source: `samples/NivaraChat/Transformer/BatchedTransformer.cs`.

- Model: token embedding (tied LM head) + sinusoidal position encoding + `NLayer` pre-norm
  blocks (LN1 → Q/K/V/O GEMMs → causal multi-head attention → residual → LN2 → GELU MLP
  (fc1 → GELU → fc2) → residual) → final LN → tied LM head
  `MatMul(x, tok_emb^T)`.
- Forward: input `[B, L]` tokens → logits `[B*L, V]`.
- Training loop (`TransformerMode.Train`): per batch sample `B` random windows of length `L`;
  forward; `CrossEntropyLoss`; `Backward`; `Adam.Step`; `ZeroGrad`. `using Grad()` scope.
  No gradient checkpointing, no clipping, constant learning rate.
- `ForwardCached` / KV-cache are inference-only and never used in training.
- Weights exported via `ToSilkWeights()` → `Dictionary<string,(float[] Data,int[] Shape)>`
  with keys `tok_emb.weight`, `final_norm.weight/bias`, `layers.{i}.*`.

## 3. Proposed device component: `SilkTinyShakespeareTrainer`

Mirrors `SilkDistilBertTrainer` / `SilkLayaHeadTrainer`:

- Owns device buffers for all trainable params (f32), gradients, Adam m/v moments, and saved
  activations.
- Constructor takes `SilkRuntime`, `TinyShakespeareConfig`, tensors (or weights), caps
  `maxBatch`, `maxSeqLen`, `trainableLayerCount` (default all layers).
- Surface: `Forward(batch, seqLen)`; `Backward(batch, seqLen, dLogits)`;
  `AdamUpdateAll(lr, wd, beta1, beta2, eps, step)`; `ExportCheckpoint()`; `ExportGradients()`;
  `Dispose()`.
- A convenience `Train(...)` loop may be added later; primitives first (matches the
  DistilBERT trainer surface).

## 4. Kernels needed (all already exist — no new kernels required)

Forward:

- `Gather` (token + position embedding lookup), `Add` (residual/positional).
- `LayerNorm` (eps 1e-5).
- `Gemm` (Q/K/V/O, fc1/fc2), `GemmEpilogue` with `GemmGelu` for fused fc1+GELU.
- `CausalGqaAttention` (self-attention, `numKvHeads == numHeads`).
- `LayerNorm` for final norm.

Backward (chain, mirroring `BatchedTransformerBlock.ForwardBlock`):

- `SoftmaxCrossEntropyBackward` (logits → dLogits, mean reduction, scale `1/(B*L)`).
- Tied LM head: `GemmNt` / `GemmTn` for activation/weight gradients; accumulate into
  `tok_emb` gradient.
- Final LN: `LayerNormBackward` (dgamma/dbeta accumulators).
- Per block reverse: residual add (identity grad split), LN2 backward, fc2 backward
  (`GemmNt`/`GemmTn`), `GeluBackward`, fc1 backward, LN1 backward, attention backward
  (`AttentionBackwardPrep`/`Dq`/`Dkv`), O-proj backward, plus residual gradient accumulation.
- Q/K/V projection backward (`GemmNt`/`GemmTn`).
- Token embedding backward: `EmbeddingBackwardRows` (input tokens) **plus** tied LM-head
  gradient accumulation into the same `tok_emb` grad buffer.
- Position encoding: no gradient (fixed).

Optimizer: `AdamUpdate` (f32) over all trainable params.

No activation-level bf16; kernels stay f32.

## 5. Tied-weight gradient accumulation (key non-trivial point)

CPU AutoDiff ties `tok_emb.weight` to the LM head automatically (same parameter, gradient
accumulates from both uses). On device, `tok_emb` and `tok_embTransposed` are separate
buffers (the runner uploads a transposed copy). The trainer must:

- Keep a single canonical `tok_emb` buffer and derive the transposed view for the LM-head
  GEMM (or maintain `tokEmbTransposed` and accumulate its grad back into the `tok_emb` grad
  buffer).
- Accumulate embedding-input gradient (`EmbeddingBackwardRows`) **and** LM-head weight
  gradient into the same grad buffer before `AdamUpdate`.
- Document that the transposed buffer is a view/alias, not an independent parameter, to avoid
  double-counting or missed gradients.

## 6. Memory budget

Caps: `maxBatch` (default 32), `maxSeqLen` (default 64) → `rows = maxBatch * maxSeqLen`
(default 2048). Device allocations:

- Weights + grads + Adam m/v: ~4× param bytes. Params ≈ `tok_emb` 8000·96 + 2 layers·
  (~6·96² + biases) + final norm ≈ 0.8M floats → ~3.2 MB per buffer × 4 (weights/grads/m/v)
  ≈ ~13 MB.
- Activations (per forward, saved for backward): x, normed×2, q/k/v, context, projected,
  gate, posBuf, idsBuf, plus per-layer saved tensors (pre-norm, H, QKV, attn probs, F1
  pre/post) ≈ ~10–12 buffers × 2048 × 96 × 4 B ≈ ~10–12 MB per distinct buffer; saved per
  layer ×2 ≈ ~40–60 MB.
- KV cache not allocated for training (inference-only).
- Total estimated peak: **< 120 MB device**, well within the 1 GB default budget and typical
  discrete/integrated GPU memory. Larger configs (vocab 8000, embd 256, 4 layers) scale
  roughly with `rows·hidden·layers`; still bounded by caps. Document a `--max-batch`/
  `--max-seq` override and an OOM guard.

## 7. Precision lane

- f32 only this release. No narrow (bf16/fp16) matmul weights, no `AdamUpdateBf16`.
  Inference runner stays f32.
- bf16/fp16 training tracked separately (consistent with the release constraint; not a goal
  here).

## 8. Save/load round-trip

- `ExportCheckpoint()` returns `Dictionary<string,(float[] Data,int[] Shape)>` with the **same
  keys/shapes** as `ToSilkWeights()`, so a trained checkpoint is structurally identical to an
  inference weight dict.
- Import path: add `BatchedTransformer.LoadFromSilkWeights(dict)` (or `FromSilkWeights`) that
  copies device-exported f32 arrays into `tokenEmb.Weight`, `finalNorm`, and each
  `BatchedTransformerBlock`'s params, converting f32→T (`float.CreateChecked`), validating
  keys/shapes.
- Persistence: after import, `ModelSerializer.Save(model, path)` writes the canonical JSON
  (same format as CPU-trained models), and `ModelSerializer.Load` reads it back. Thus
  device-trained models are indistinguishable from CPU-trained ones on disk.
- Round-trip parity: device train → export → import → CPU `Forward` logits match device
  forward within f32 exactness (bit-for-bit where kernels are deterministic; note OpenCL
  reduction order may introduce ~1e-6 differences — gate on tolerance ≤ 1e-5 for logits, exact
  for argmax).

## 9. Parity gate

- F32 Silk trainer vs CPU AutoDiff on a **tiny seeded fixture**: small vocab (e.g. 32),
  hidden 16, 1 layer, 2 heads, seq 8, batch 4, fixed seed for weights and sampled windows.
- Compare: (a) forward logits, (b) loss, (c) parameter gradients after one step,
  (d) post-step weights after one Adam step. Seeded weights/data so baselines are stable
  (lesson from `SilkLayaHeadTrainerBf16ParityTests`).
- Gate tolerance: logits/gradients ≤ 1e-5 relative; weights ≤ 1e-5. One failure reason per
  assertion.
- Also a save/load round-trip parity test (section 8).
- CPU path remains the reference; Silk is never the baseline.

## 10. AutoDiff semantics: matched vs divergent

Matched:

- Per-batch forward → CE loss → backward → Adam step → zero grad; `Grad()` scope equivalent
  (device trainer is always in "grad mode").
- Constant LR, no clipping, no schedule (matches CPU).
- Causal mask (upper-triangular -inf), sinusoidal positions, pre-norm, weight tying.

Documented as divergent / not supported on Silk:

- Gradient checkpointing: CPU `Module` supports none here anyway; device trainer does not add
  it. If added to CPU later, Silk must follow or be documented divergent.
- Mixed precision / bf16/fp16 training: f32 only this release.
- KV-cache path is inference-only on both sides.
- OpenCL reduction order may differ from CPU SIMD/sequential order → tiny numeric
  differences (covered by tolerance gate).
- No per-op `OperationDiagnostics`/kernel selection on device; diagnostics record
  trainer-level step metrics only.

## 11. Non-goals

- Replacing the CPU AutoDiff path (it stays the reference).
- bf16/fp16 training on Silk (f32 only this release).
- BPE migration (issue #1328).
- KV-cache training.
- Multi-GPU / distributed training.
- Encoder/other models (SmolLM, Qwen, vision) — separate issues (#1283/#1284).
- New OpenCL kernels if existing ones suffice (reuse the listed set).
- Changing the default training backend (CPU stays default).

## 12. Proposed CLI surface

- Add `--silk-train` flag to `TransformerMode` (default off) to train on device; `--silk`
  keeps its current meaning (inference backend). Training on Silk exports via
  `ExportCheckpoint()` then imports into the CPU `BatchedTransformer` so save/load and
  inference paths are unchanged.
- Alternative (simpler): reuse `--silk` to mean "train on device when training". Decision
  needed before implementation; recommend `--silk-train` to avoid ambiguity with the inference
  backend.

## 13. Milestones

1. This design note (done).
2. `SilkTinyShakespeareTrainer` f32 primitives + tied-weight grad accumulation.
3. Parity tests (F32 vs CPU) on seeded fixture.
4. `BatchedTransformer.LoadFromSilkWeights` + save/load round-trip parity test.
5. Wire `--silk-train` in `TransformerMode` + docs/help.
6. Update `NIVARACHAT-SILK-GAPS.md` Gap S6 status and `ARCHITECTURE.md` decisions.

## 14. Open decisions to confirm before implementation

- `--silk-train` vs reusing `--silk`.
- Whether to ship a convenience `Train(...)` loop in v1 or just primitives (recommend
  primitives first, matching DistilBERT).
- Max caps and OOM guard defaults.
- Tolerance threshold for the parity gate (recommend 1e-5).
- Whether position encoding should be a device buffer (yes, precomputed) or host-uploaded
  per step (recommend precomputed buffer, matching the runner).

## 15. References

- `samples/NivaraChat/Transformer/BatchedTransformer.cs` (`ToSilkWeights`,
  `ForwardBlock`, `ForwardCached`).
- `samples/NivaraChat/Transformer/TransformerMode.cs` (`Train`, `RunSilk`/`RunSilkChat`).
- `src/Nivara.Extensions/Silk/SilkTinyShakespeareSession.cs`,
  `SilkTinyShakespeareGpuRunner.cs`.
- `src/Nivara.Extensions/Silk/SilkDistilBertTrainer.cs`,
  `SilkLayaHeadTrainer.cs`, `SilkPartialTrainingPolicy.cs`.
- `src/Nivara/AutoDiff/ReverseGradOperations.cs` (`BatchedMultiHeadAttention`).
- `docs/SILK.md`, `docs/adr/001-autodiff-nonnullable-domain.md`,
  `docs/adr/002-autodiff-span-boundary.md`.
- `docs/plan/LAYA-HEAD-TRAINING.md` (format template).

## 16. Follow-up issues (post-PR #1359)

Remaining work tracked as GitHub issues:

- #1360 — TinyShakespeare Silk: add save/load round-trip + loss parity tests.
- #1361 — TinyShakespeare Silk: update CLI help and docs/SILK.md.
- #1362 — TinyShakespeare Silk: enforce maxBatch/maxSeqLen OOM guard.
- #1363 — TinyShakespeare Silk: support trainableLayerCount (partial/frozen training).
