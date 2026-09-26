# TODO — ModernBERT encoder + Laya decision head in Nivara

Research, tensor maps, and grounded architecture semantics live in [`docs/LAYA.md`](LAYA.md).
This file is the executable plan for branch `khurram/modernbert`.

> As each task executes, if you find deferred work or a concern, create a GitHub issue
> **immediately** (`gh issue create --repo khurram-uworx/Nivara`) and record the number in the
> log at the bottom — don't rely on memory or wait until the end of the plan.

## Problem

Laya (`convaiinnovations/laya`) is a 421M non-autoregressive decision model whose backbone is
**ModernBERT-large**, and the user wants Nivara to run it for inference. Nivara today supports
DistilBERT/MiniLM (Post-LN BERT) and the Llama/SmolLM/Qwen causal family, but nothing
ModernBERT-shaped.

Two phases, in order:

1. **Phase 1 — ModernBERT encoder.** Get `answerdotai/ModernBERT-large` running under Nivara
   with a last-hidden-state parity gate against HuggingFace. This is the load-bearing work;
   Laya's encoder is a ModernBERT-large fine-tune of the same architecture.
2. **Phase 2 — Laya head.** Port Laya's from-scratch decision head + prompt format
   (`rl_common.py`) on top of the Phase 1 encoder, gated on a PyTorch parity fixture.

Phase 1 does **not** require Phase 2 and does not require the Laya checkpoint.

## What already exists (no new core op needed)

Verified against the sources — see `docs/LAYA.md` §5:

- `RotaryEmbedding<T>` — ModernBERT uses the same HF `rotate_half` half-split convention.
  Per-layer-type theta is just two instances (160000 full / 10000 sliding).
- `LayerNorm<T>(n, eps, affine)` — ModernBERT's norm is LayerNorm with `bias=False`; an
  all-zero Beta is numerically identical, so `affine: true` + no Beta load is exact.
- `ReverseGradOperations.MultiHeadAttention` takes an arbitrary additive `[qLen, kvLen]` mask
  → the sliding window is a **band mask**, no new op.
- Gated MLP house style (`LlamaDecoderBlock.cs:105-107`) is `Linear` + activation + `Multiply`,
  so GeGLU is `GeluExact` in place of `Silu` — no new op.
- `Embedding<T>.Forward(int[])`, `SafeTensorsLoader.Read<float>` (handles Laya's F16-on-disk),
  `StateDictLoader.LoadLinear/LoadLayerNorm/LoadRMSNorm`.

**`src/Nivara` needs no changes in Phase 1.** The two real gaps are sample-side.

## Two corrections from grounding (recorded so nobody re-derives them)

- **Not RMSNorm, not layer scale.** ModernBERT's norm is bias-free **LayerNorm**
  (`modeling_modernbert.py:319`, `norm_bias: false` in both configs), and the residual is a
  plain add — `answerdotai/ModernBERT src/bert_layers/layers.py:304,323` has no `gamma`
  anywhere. Layer scale is a common misattribution; building it would have been wasted work
  *and* wrong.
- **`Wi` gate is the second half.** `input, gate = Wi(x).chunk(2, dim=-1)` then
  `act(gate) * input` → rows `[0:2624]` = up, `[2624:5248]` = gate.
- **Sliding window is symmetric and inclusive**: `abs(i - j) <= local_attention / 2` (band 129),
  matching answerdotai's flash `window_size=(64, 64)`.

---

## Phase 1 — ModernBERT encoder

### 1.1 Data + gitignore

- [ ] `.gitignore`: add `samples/data/modernbert/` and `samples/data/laya/` (one line each,
      following the `samples/data/distilbert/` pattern at `.gitignore:360`).
- [ ] Document the download in `samples/NivaraInference/README.md`:
      `hf download answerdotai/ModernBERT-large --local-dir samples/data/modernbert`
      (single 1.6 GB F32 `model.safetensors` + `config.json` + `tokenizer.json`).
- [ ] Download it locally (outside the repo tree it lands in the gitignored dir, so this is safe).

### 1.2 `samples/Nivara.Samples/ModernBertModel.cs` (new)

- `ModernBertConfig` — parse `config.json` with `JsonDocument` (NOT the string-search
  `BertConfig.FromJson` idiom: ModernBERT has nested `rope_parameters` and a `layer_types`
  array). Fields: `HiddenSize`, `NumAttentionHeads`, `NumHiddenLayers`, `IntermediateSize`,
  `VocabSize`, `MaxPositionEmbeddings`, `NormEps`, `LocalAttention`, `LayerTypes[]`,
  `RopeThetaFull`, `RopeThetaSliding`, `PadTokenId`, `ClsTokenId`, `SepTokenId`, `MaskTokenId`.
  Derived: `HeadDim`, `SlidingWindow => LocalAttention / 2`, per-layer
  `(bool IsFull, float RopeTheta)`.
- `ModernBertAttention<T>` — `qProj/kProj/vProj/oProj` all `bias: false`; a `RotaryEmbedding<T>`
  sized `HeadDim`; `Forward(hidden, mask?)` = project → rope(Q), rope(K) →
  `MultiHeadAttention(Q, K, V, numHeads, 1/sqrt(HeadDim), mask)` → `oProj`.
- `ModernBertMlp<T>` — `upProj`/`gateProj`/`downProj`, all `bias: false`;
  `Forward` = `Multiply(GeluExact(gate), up)` → `downProj`.
- `ModernBertLayer<T>` — `attnNorm` (**null / Identity for layer 0**) + `mlpNorm`, both
  `LayerNorm<T>(hidden, normEps, affine: true)` with Beta left at zero;
  `h = h + attn(attnNorm(h))`; `h = h + mlp(mlpNorm(h))`.
- `ModernBertEncoder<T>` — `tokEmbed`, `embedNorm`, `layers[]`, `finalNorm`;
  `Forward(int[] tokenIds, int[]? validLengths)` → `[L, HiddenSize]`;
  `ForwardBatched(int[][], int[][] validLengths)` via `BatchedMultiHeadAttention`.
- `ModernBertMasks` (static, same file) — additive `[B, L, L]` / `[L, L]` builder:
  `-inf` where `Math.Abs(i - j) > band` (band = -1 → no band) **or** `j >= validLength`.
  Band is per-*layer*, not per-head, so one mask serves the layer. Fail loudly (not silently
  full-attention) when `seqLen` exceeds the dense-mask budget — see issue log.
- `ModernBertLoader` — `LoadWeights<TModel, TWeight>(tensors, config, prefix)` where `prefix` is
  `"model"` (stock HF) or `"encoder"` (Laya) so **one class serves both checkpoints**;
  per-layer `attn_norm`/`mlp_norm` weight load, with layer 0's `attn_norm` skipped.

### 1.3 `samples/Nivara.Samples/StateDictLoader.cs` (additive)

- [ ] `LoadLinearSlice<TModel, TWeight>(Linear<TModel> target, tensors, key, rowOffset, rowCount)`
      — bind one row block out of a fused weight. Required because `Wqkv` is `[3H, H]` and `Wi`
      is `[2I, H]`. Validate the fused shape and fail with a clear message.
- [ ] `LoadLayerNorm` already tolerates a missing `.bias` — no change (only the weight is loaded
      for ModernBERT's bias-free norms).

### 1.4 `samples/Nivara.Samples/Gpt2BpeTokenizer.cs` (additive)

- [ ] `LoadFromTokenizerJson(string path)` — ModernBERT ships **no `vocab.json`/`merges.txt`**,
      only `tokenizer.json` with inline `model.vocab` (dict) + `model.merges` (list). Reuse the
      existing `added_tokens` merge path.
- [ ] NFC normalization: ModernBERT declares `normalizer: {"type": "NFC"}`; apply
      `text.Normalize(NormalizationForm.FormC)` before pretokenizing. Keep it opt-in via the new
      entry point so the SmolLM/Qwen paths are byte-unchanged.
- [ ] `EncodeIds(string, bool addSpecialTokens)` helper returning `[CLS] … [SEP]` for the
      ModernBERT convention (`[CLS]`=50281, `[SEP]`=50282, `[PAD]`=50283, `[MASK]`=50284).

### 1.5 `samples/NivaraInference/ModernBert.cs` (new) + `Program.cs`

- [ ] `RunModernBertInference` — tokenize a fixed sentence, forward, print shape/stats
      (mirrors `RunDistilBertInference`).
- [ ] `BenchmarkModernBert` — 3 warmup + 10 timed, avg/min/max ms + params + weight MB
      (mirrors `BenchmarkDistilBert`).
- [ ] `RunModernBertCompare` — load `last_hidden_state_py.bin` + `input_ids_py.bin` when present;
      report maxAbs, maxRel, cosine, violation count against the established bound
      `|cs − py| ≤ 1e-3·(1 + |py|)`; also assert **tokenizer id agreement**. Print
      "reference not found; skipping diff" otherwise (existing convention).
- [ ] `Program.cs`: add `case "modernbert":` next to the other text models.

### 1.6 `samples/NivaraInference/Python/modernbert_compare.py` (new)

- [ ] Mirror `distilbert_compare.py`: `AutoModel` + `AutoTokenizer`, one fixed sentence,
      `padding="max_length", truncation=True, max_length=128`, save
      `samples/data/modernbert/last_hidden_state_py.bin` (and the ids, so the tokenizer is gated
      independently of the model).

### 1.7 Tests (`tests/Nivara.Tests`)

- [ ] `AutoDiff/ModernBertMaskTests.cs` — band boundaries (`|i-j| == band` visible, `band+1`
      suppressed), pad interaction (padded `j` suppressed, valid `j` not), full-attention layer
      has no band, batched vs single-sequence agreement.
- [ ] Extend `AutoDiff/Gpt2BpeTokenizerTests.cs` — `tokenizer.json`-only load produces the same
      ids as `vocab.json`+`merges.txt` for the same vocab; NFC input.
- [ ] `StateDictLoader.LoadLinearSlice` — row-block extraction from a fused weight, and a
      clear throw on a shape mismatch.
- [ ] Weight-mapping test: synthetic `model.`-prefixed and `encoder.`-prefixed tensor dicts both
      bind a tiny ModernBERT config, and the two loaders agree (proves the prefix parameter).
- [ ] `StateDictLoader` already has no dedicated test file → cover it in the ModernBERT tests
      rather than creating a third file for one method.

### 1.8 Docs

- [ ] `samples/NivaraInference/README.md` — `modernbert` row in Supported models, quick-start
      commands, architecture section (and the two corrections, since a reader will assume layer
      scale), the core/sample reuse table, the PyTorch-vs-Nivara benchmark row, and the
      `hf download` command. Note the F32-only choice for Phase 1 and why.

## Phase 2 — Laya head (after Phase 1's gate passes)

- [ ] `LayaDecisionHead<T>` — `type_emb` add → 2 × pre-norm transformer layer
      (16 heads, FFN 4096, **ReLU**, `LayerNorm` **with bias**, biased fused `in_proj` +
      `out_proj`) → gather marker rows → `scorer` (LayerNorm → Linear 1024×1024 → GELU → Linear
      → 1) → `masked_fill(~markerMask, -1e4)`. The biased fused in-proj is
      `MultiHeadAttention` + `ReverseGradOperations.AddBias`; `head.dense`/`head.norm` of the
      backbone are **not** needed (Laya does not use the MLM/classification head).
- [ ] `LayaPromptBuilder` — C# port of `render_options` + `build_sequence` (option rendering per
      qtype, the 48-token per-option cap, the `opt_budget < 16` even-shrink fallback, the
      `head_ids[:max(8, opt_budget)]` cap, marker positions, state truncation to
      `max_len - len(ids) - 1`). **Byte-exact prompt parity is the gate** — the prompt is the
      API contract, and a silently different render still "runs".
- [ ] `LayaCalibration` — `temp_bucket(qtype, k)` + `temperature_by_options` override, falling
      back to the per-qtype `temperature` vector.
- [ ] `laya` mode (`default` / `benchmark` / `compare`) + `Python/laya_compare.py` reference
      (a ~60-line port of `rl_common.DecisionModel`/`build_sequence`, Apache-2.0, reading the
      shipped `model.safetensors`). Cross-check against `pip install laya` if that install is
      acceptable — see open question below.
- [ ] `act_head` — **optional, and the model card says not to build it first** (AUROC 0.30,
      reads 1.0 almost always). Gate on answer confidence instead.
- [ ] Out of scope unless asked: `laya-multilingual` (mmBERT-base, different tokenizer +
  8k context), `laya-typed-decisions` subfolder, training/RLCD, temperature refitting.

## Verification steps

1. `dotnet build Nivara.slnx` — clean, no new warnings.
2. `hf download answerdotai/ModernBERT-large --local-dir samples/data/modernbert`.
3. `python samples/NivaraInference/Python/modernbert_compare.py` — fixture written.
4. `dotnet run --project samples/NivaraInference -c Release -- modernbert` — forward runs.
5. `dotnet run --project samples/NivaraInference -c Release -- modernbert compare` —
   **the gate**: token ids match, `maxRel` at DistilBERT class (~1e-5), 0 violations.
   A cosine below ~0.999 means a structural bug, not precision.
6. `dotnet run --project samples/NivaraInference -c Release -- modernbert benchmark` — record
   Nivara ms.
7. `python samples/NivaraInference/Python/modernbert_benchmark.py` — same methodology, record
   PyTorch ms, compute the ratio in the same session.
8. **Ask before** `dotnet test`; then run the new test files, then the AutoDiff suite as the
   regression guardrail.
9. Smoke-check that the SmolLM/Qwen/DistilBERT modes still run (the `Gpt2BpeTokenizer` and
   `StateDictLoader` edits are shared code).

## Debug ladder (if the gate fails)

Parity failures in this architecture are almost always one of exactly five things, in
descending likelihood:

1. `Wi` gate/up halves swapped (or Q/K/V thirds swapped in `Wqkv`).
2. Gate omitted entirely — some ports silently use a plain 2-layer MLP.
3. Layer-0 `attn_norm` skipped/duplicated (or a `mlp_norm` applied post-residual).
4. Sliding-window band off by one (64 vs 65) or applied to full-attention layers.
5. RoPE theta swapped between the full (160000) and sliding (10000) layer types.

Bisect by comparing `last_hidden_state` after layer 0, 1, 2, 3… — the first diverging layer
identifies the bug. Add a temporary `compare_diag`-style mode only if the ladder is not enough.

## Blast radius

- `src/Nivara`: **no changes planned in Phase 1.** All new code is `samples/` + tests + docs.
- `samples/Nivara.Samples/Gpt2BpeTokenizer.cs` — shared by SmolLM, Qwen, and their tests.
  Additive only: a new factory + an opt-in NFC flag. The existing ctor and encode path must stay
  byte-identical; verified by the existing `Gpt2BpeTokenizerTests` and `Qwen/*` parity suites.
- `samples/Nivara.Samples/StateDictLoader.cs` — shared by DistilBERT/MiniLM/Llama. Additive
  method only; `LoadLinear`/`LoadLayerNorm`/`LoadRMSNorm` untouched.
- `samples/NivaraInference/Program.cs` — one new `switch` case; the existing models' behaviour is
  unchanged (guard with a smoke run of one text model).
- `.gitignore` — two new lines, no existing rule modified.
- Memory: a 1.6 GB F32 checkpoint read into `float[]` tensors is ~1.7 GB managed heap (the same
  shape as the Qwen 989 MB BF16 load, which peaked at ~1.88 GB). Fine on this machine, worth
  stating in the README.
- Dense-mask memory: `[512, 512]` additive mask is 1 MB per layer per forward. **Never** build it
  at 8192 tokens (would be ~1 GB) — the builder must throw, not fall back.

## Planned commits

1. `docs: add LAYA.md — ModernBERT/Laya research reference and plan` (this file + `LAYA.md`)
2. `feat: add ModernBERT encoder (GeGLU, RoPE, banded attention) to the inference sample`
3. `feat: add modernbert inference mode with HuggingFace parity gate`
4. `test: cover ModernBERT band masks, fused weight splits, tokenizer.json-only BPE load`
5. `docs: document the modernbert mode in NivaraInference README`

## Open questions for the human

- **Laya reference source.** Port `rl_common.py` into `Python/laya_compare.py` (self-contained,
  no new deps) vs `pip install laya` and diff against the real package (strongest ground truth,
  adds an install). Default to the port, cross-check if the install is cheap.
- **Phase 2 checkpoint.** 808 MB F16 for Laya on top of the 1.6 GB ModernBERT download. Confirm
  both are wanted locally, or whether Phase 2 should reuse Laya's encoder for its own parity gate
  and skip the stock ModernBERT download.
- **`samples/data/modernbert/` vs reusing `samples/data/laya/encoder/`** for the Phase 1 gate.
  Default: separate dirs, since Phase 1 stands alone.

## GitHub issues log

- [ ] #NNN — `LayerNorm<T>` has no bias-free mode (`bias: false`); ModernBERT's `norm_bias: false`
      is emulated by a zero Beta, which is exact for inference but wrong for fine-tuning.
- [ ] #NNN — banded/sparse attention kernel: a dense `[L, L]` mask cannot serve ModernBERT at
      `max_position_embeddings = 8192` (≈1 GB), and makes sliding layers cost the same as full ones.
- [ ] #NNN — ILGPU/OpenCL GPU path for ModernBERT (`BertEncoderGpuRunner` is Post-LN BERT only).
- [ ] #NNN — `LayerNorm` norm-after-residual vs ModernBERT pre-norm fused block: a
      ModernBERT-specific fused forward (per-layer band + theta) is a perf follow-up.
- [ ] #NNN — Laya `act_head` / escalate signal is documented as unusable (AUROC 0.30);
      investigate or explicitly close.
- [ ] #NNN — NFC normalization is only implemented on the new `tokenizer.json` entry point;
      consider promoting it to the shared byte-level BPE path.
