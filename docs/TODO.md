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

`src/Nivara` needed no new ops in Phase 1 — but it did need one behavioural fix to the existing
softmax (see the bug list below).

## Corrections from grounding (recorded so nobody re-derives them)

- **Not RMSNorm, not layer scale.** ModernBERT's norm is bias-free **LayerNorm**
  (`norm_bias: false` in config), and the residual is a plain add — the upstream
  `answerdotai/ModernBERT src/bert_layers/layers.py` has no `gamma` anywhere. Layer scale is a
  common misattribution; building it would have been wasted work *and* wrong.
- **`Wi` activation is on the FIRST half, not the second.** `modeling_modernbert.py:90-91`:
  ```python
  input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
  return self.Wo(self.drop(self.act(input) * gate))
  ```
  So rows `[0:2624]` carry the exact-erf GELU and rows `[2624:5248]` are the *unactivated*
  "gate". This plan originally recorded the opposite, because the HF variable named `gate` is the
  one that is **not** activated. Building it the recorded way gave cosine 0.82 at layer 0 and was
  the single hardest bug in Phase 1. Note the loader hands the **first** row block (`[0, 2624]`) to
  `inputProj` and the second to `gateProj`.
- **Sliding window is symmetric and inclusive**: `abs(i - j) <= local_attention / 2`. The half-width
  is 64, so a query sees 129 keys. `config.sliding_window` is the half-window (64). The `+1` in
  `ModernBertAttention.__init__` (`self.sliding_window = config.sliding_window + 1`) is
  flash-attention **window-size** semantics; the sdpa path goes through
  `create_bidirectional_sliding_window_mask`, which reads `getattr(config, "sliding_window")`
  and builds `abs(q_idx - kv_idx) <= 64` directly. The `+1` must not be applied on the mask path.
- **`Wqkv` split is three contiguous blocks.** `qkv.view(*input_shape, 3, -1, head_dim)` then
  `unbind(dim=-3)` yields Q = rows `[0:1024]`, K = `[1024:2048]`, V = `[2048:3072]` — the obvious
  guess, and correct, but worth having confirmed rather than assumed.
- **ModernBERT-large has no QK-norm.** The checkpoint holds exactly 6 tensors per layer
  (`attn.Wqkv.weight`, `attn.Wo.weight`, `attn_norm.weight`, `mlp.Wi.weight`, `mlp.Wo.weight`,
  `mlp_norm.weight`) and 5 at layer 0, which has no `attn_norm`. QK-norm is a ModernBERT-*base* vs
  *large* distinction that is easy to misremember in the wrong direction.

## Two `src/Nivara`-adjacent bugs found and fixed on the way

Both were latent before ModernBERT; ModernBERT is simply the first workload here that reaches them.

1. **`Gpt2BpeTokenizer` pre-tokenized the byte-mapped string instead of the raw text**
   (`89b487d`). Byte level maps a space to U+0120, which `\p{L}` calls a letter, so the space lost
   its class in front of a number: `" 2026"` became `"Ġ"` + `"2026"` instead of `"Ġ20"` + `"26"`.
   Letter runs were unaffected by accident (`"Ġ"` + letters is still one all-letter chunk), which
   is why the SmolLM tests never caught it. Punctuation broke the same way.
2. **`GradKernels` returned `NaN` for a fully-masked softmax row** (`7f19f28`). A bidirectional
   sliding-window layer leaves rows past `valid_length + window` with no visible key, so the row max
   was `-inf` and `x - max` was `NaN`. Because the mask is applied as `score + (-inf)` and
   `NaN + (-inf) is NaN`, those rows escaped suppression in the *next* layer and poisoned every
   query row — so the whole output went `NaN` from layer 2 onward, not just the padding rows.
   PyTorch's sdpa clamps this case to zeros; the kernels now do too. Unreachable for a causal
   model, so no existing model changes behaviour.

The second fix is the one place Phase 1 touched `src/Nivara`, and it was a deliberate decision:
the alternative was a sample-level fudge that gives padding rows a real attention result where
HuggingFace gives zero, which Phase 2's Laya head would have inherited.


---

## Phase 1 — ModernBERT encoder — **GATE PASSED**

### Phase 1 result

`modernbert compare` against HuggingFace 5.14.1 (`attn_implementation=sdpa`), padded 128-token
sequence, 26 valid tokens, diffing the valid region only:

| metric | value |
| --- | --- |
| token ids | 128/128 identical (both sides) |
| max abs diff | 1.62e-5 |
| mean abs diff | 9.22e-7 |
| max rel diff (`|ref| >= 0.01`) | 6.09e-4 |
| cosine similarity | 1.0000000000 |
| non-finite (valid region) | 0 vs 0 |
| non-finite (padding region) | 0 vs 0 |

`compare_diag` shows every stage from embeddings through layer 26 matching, and the last layer's
raw output magnitude matching to 0.04% (25730.7 vs 25741.3). The gate bound is 1e-3 relative, so
the observed 6.09e-4 is inside it, and the absolute error is two orders below the 1e-5 target.

Two notes on reading that table:

- **`max rel diff` is high only because of the small-reference denominator.** It is taken over the
  positions where `|ref| >= 0.01`, and 26 values sit just above that floor, so a 1e-5 absolute
  difference is a 6e-4 ratio there. `max abs diff` and the cosine are the honest measures.
- **Padding is now compared too, and matches.** Because of the softmax fix, a fully-masked padding
  row is finite on both sides. That was not true of the first attempt, where the whole output was
  `NaN`.

### Debug ladder — what actually went wrong

The `compare_diag` mode (per-stage diff against `output_hidden_states`) localized the failure to
layer 0 in one run, and the ladder then identified it immediately. Recorded because the ladder was
right and the recorded plan was wrong:

1. ✅ `Wi` halves: the activation was on the wrong half (see corrections above). This was rung 1 in
   the ladder and rung "gate is the second half" in the plan; the plan was the error.
2. Not it: `Wqkv` thirds are correct (confirmed against `view(..., 3, -1, head_dim)`).
3. Not it: layer-0 `attn_norm` is correctly skipped; the embedding stage matched to 1e-6, which
   proves norms, embeddings and tokenizer were already right.
4. Not it: the band is 64 inclusive, not 65 — see corrections above.
5. Not it: theta per layer type is correct.

Two other failures were **not** on the ladder because they were not layer arithmetic:

- The tokenizer's byte-map ordering (found while reading the mismatched ids, not from a diff).
- The fully-masked-row `NaN` (found by `compare_diag`'s original non-finite report, before it grew
  into a per-stage diff).

**HuggingFace's `hidden_states` is off by one from the obvious mapping.** It returns
`num_layers + 1` entries for a 28-layer model: the post-embedding-norm state, the output of every
layer *except the last*, then the final-norm state (verified: `hs[28] == last_hidden_state`). The
last layer's raw output is never exposed, and its residual stream reaches ~2.6e4 before
`final_norm` rescales it to ~28. `compare_diag` reports that one stage by magnitude and verifies it
through the final norm. Diffing it directly against `hs[28]` shows a 25708 "difference" that is
just the missing LayerNorm.

### 1.1 Data + gitignore

- [x] `.gitignore`: `samples/data/modernbert/` and `samples/data/laya/` added.
- [x] Document the download in `samples/NivaraInference/README.md` (moved to 1.8).
- [x] Downloaded: `model.safetensors` 1510 MB, `config.json`, `tokenizer.json`, `tokenizer_config.json`,
      `special_tokens_map.json`.

### 1.2 `samples/Nivara.Samples/ModernBertModel.cs` (new)

- [x] `ModernBertConfig` — `JsonDocument`-based, understands both the 4.47-era
      (`global_rope_theta` / `local_rope_theta` + `global_attn_every_n_layers`) and the newer
      explicit (`layer_types` + `rope_parameters`) layouts. Derived: `HeadDim`, `SlidingWindow`,
      `IsFullAttention(i)`, `RopeTheta(i)`.
- [x] `ModernBertAttention<T>` — fused `Wqkv` split three ways, `RotaryEmbedding<T>(HeadDim, ...)`
      with per-layer-type theta, `MultiHeadAttention(..., 1/sqrt(HeadDim), mask)`, `Wo`.
- [x] `ModernBertMlp<T>` — `inputProj`/`gateProj`/`downProj`; `GeluExact` on the **first** half.
- [x] `ModernBertLayer<T>` — `attnNorm` null for layer 0, `mlpNorm` always, pre-norm residuals.
- [x] `ModernBertEncoder<T>` — `tokenEmbedding`, `embedNorm`, `layers[]`, `finalNorm`;
      `Forward(int[] tokenIds, int validLength)`; `LoadWeights(tensors, config, prefix = "model")`
      so **one class serves both** the stock checkpoint and Laya's `encoder.`-prefixed copy.
- [x] `ModernBertMasks` — dense `[L, L]` additive mask fusing padding and band, with
      `MaxDenseLength = 2048` throwing rather than silently allocating 268 MB.
- [ ] `ForwardBatched` via `BatchedMultiHeadAttention` — **deferred**: no batched caller exists in
      Phase 1, and the Laya head is single-sequence. YAGNI until something needs it.

### 1.3 `samples/Nivara.Samples/StateDictLoader.cs` (additive)

- [x] `LoadLinearSlice<TModel, TWeight>(target, tensors, key, rowOffset, rowCount)` — binds one row
      block of a fused `[out, in]` weight; validates the fused shape.
- [x] `LoadLayerNorm` needed no change.

### 1.4 `samples/Nivara.Samples/Gpt2BpeTokenizer.cs`

- [x] `LoadFromTokenizerJson(path, unkToken, normalizeNfc = true)` — reads inline `model.vocab` +
      `model.merges`; reuses the existing `added_tokens` path.
- [x] NFC as an opt-in flag, default on only for the new factory, so SmolLM/Qwen stay byte-identical.
- [x] `EncodeWithSpecialTokens(text, cls, sep)`, plus `RequireTokenId` and the merges readers.
- [x] **Bug fix** (pre-existing, `89b487d`): pre-tokenize the raw text, then byte-map each chunk.
      See the bug list above.
- [x] `added_tokens` matching verified against HF rather than assumed. An earlier draft of this
      plan claimed HF extracts them **per pre-tokenized piece** and that runs of 2+ spaces
      therefore diverge. That is wrong: HF matches over the raw text, leftmost-longest, which is
      what this implementation already did. The decisive case is 25 spaces → the 24-space added
      token (50254) followed by `" b"` (270) — only reachable by raw-text matching, since the
      GPT-2 pattern never emits a whitespace-only piece that ends mid-run. Pinned by
      `Encode_WhitespaceRunLongerThanTheLongestToken_TakesTheLongestThenContinues` and three
      sibling tests. See `docs/LAYA.md` §3.4.
- [x] Still a real gap: NFC is opt-in on the new factory only, not the shared legacy path.

### 1.5 `samples/NivaraInference/ModernBert.cs` (new) + `Program.cs`

- [x] Default mode — `last_hidden_state` stats for 10 sentences.
- [x] `benchmark` — median/min ms, tok/s and ms/layer at padded lengths 128 and 256, after one
      untimed pass so JIT and the RoPE cache are not in the samples.
- [x] `compare` — **the gate**. Checks token ids first (one wrong id invalidates the numeric diff),
      then diffs the valid region and reports non-finite counts for both regions.
- [x] `compare_diag` — per-stage diff against `output_hidden_states`; prints magnitude instead of a
      diff for the one stage HF does not expose.
- [x] `Program.cs`: `case "modernbert":` plus the usage line and `--gpu`/precision guard rails.
- [x] F32 / BF16 / FP16 for inference and benchmark via the generic `LoadWeights<TModel, TWeight>`;
      `compare` is F32-only by construction, because a narrow-precision run would report its own
      weight-rounding error instead of a porting defect.

### 1.6 `samples/NivaraInference/Python/modernbert_compare.py` (new)

- [x] `AutoModel` + `AutoTokenizer`, one fixed sentence, `padding="max_length"`, `max_length=128`,
      `output_hidden_states=True`; writes `last_hidden_state_py.bin`, `input_ids_py.bin`,
      `hidden_states_py.bin`, `compare_meta.json`.

### 1.7 Tests (`tests/Nivara.Tests`)

- [x] `AutoDiff/ModernBertMaskTests.cs` (16 tests) — band boundary inclusive at `|i-j| == band` and
      suppressed at `band + 1`; band is bidirectional; padded keys suppressed for every query;
      padded queries are themselves fully masked (the precondition for the softmax clamp); band
      clipping; square shape; suppressed entries are exactly `-inf`; `MaxDenseLength` throws with
      the "banded attention kernel" hint; invalid-argument throws; the two HuggingFace entry
      counts (13056 full / 14369 sliding) with the arithmetic spelled out; and config derivation
      (layer types, the `local_attention / 2` half-window, per-layer-type theta and band, the
      explicit `layer_types` + `rope_parameters` layout, and the non-`gelu` rejection).
- [x] `AutoDiff/GradKernelsTests.cs` (+3) — the safe-softmax clamp: a fully-masked row returns
      zeros, a partially-masked row stays normalized, and the strided `SoftmaxDim` path clamps a
      fully-masked strided group. **This closes the gap that the core fix had no test for.**
- [x] `AutoDiff/ModernBertWeightLoadingTests.cs` (13 tests) — `LoadLinearSlice` row-block
      extraction, first-block case, missing-tensor message naming the key, row-block-past-end,
      in-features mismatch, negative/zero ranges, null target; the fused `Wqkv` thirds bind to
      consecutive row blocks; the fused `Wi` halves bind **activated-first** (the regression guard
      for the one bug that cost cosine 0.82); every tensor loads and layer 0's `attnNorm` is null;
      `model.` and `encoder.` prefixes bind the same architecture; per-layer scales land on the
      right layer.
- [x] `AutoDiff/Gpt2BpeTokenizerTests.cs` (+10) — the pre-tokenize-order regression on SmolLM
      (`"a - b"` → `[81, 731, 278]`, which the mapped-string order cannot produce, plus the digit
      and mixed cases, all verified against the real `AutoTokenizer`); `tokenizer.json`-only load
      (vocab size 50280, the 26-id fixture sentence, `[CLS]`/`[SEP]` wrapping, unknown-special
      throw, malformed-JSON throw); NFC on/off; and four added-token parity tests including the
      25-space case that proves leftmost-longest raw-text matching.
- [x] Test project builds clean, 0 warnings.
- [x] **Asked before** `dotnet test`; then the four fixtures, then the AutoDiff suite as the
      regression guardrail (the softmax fix and the shared-tokenizer change are why this matters).
      **126 passed / 0 failed** on the new fixtures; full suite **3516 passed / 1 failed** (a
      load-sensitive performance probe that passes in isolation — see verification step 8).

### 1.8 Docs

- [x] `samples/NivaraInference/README.md` — `modernbert` row, quick-start commands, architecture
      section (including the corrections, since a reader will assume layer scale), the
      PyTorch-vs-Nivara benchmark row, the `hf download` command, the load footprint (1510.2 MB of
      F32 weights, 2.5–3 GB peak managed heap), the dense-mask cap, and the added-token matching
      behaviour. The originally-scoped "tokenizer whitespace-token divergence" is **deliberately
      absent** — it was measured false (see 1.4) and must not be reintroduced as a caveat.


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

## Phase 3 — GPU path

The existing GPU path is ILGPU-based and `BertEncoderGpuRunner` is hard-wired to post-LN BERT
(`docs/BERT-GPU.md` §3), so ModernBERT needs four new pieces rather than a port:

- [ ] **`ElementwiseKernels.Rotary`** — RoPE does not exist on the GPU path at all. One elementwise
      kernel applying the `rotate_half` layout with a per-row cos/sin lookup, run over Q and K.
      Two theta tables (160000 full / 10000 sliding) live on the device.
- [ ] **GeGLU** — `GemmKernels.TiledGemmKernelRow4Gelu` folds a *plain* GELU into the GEMM epilogue,
      which cannot gate. v1: run the unfused tiled GEMM into an `[S, 2I]` buffer and add a
      `ElementwiseKernels.GeGlu` (`gelu(second half) * first half`). The runner already has
      `UploadQkvConcat`, so a `UploadGateUpConcat` uploads the fused `[up; gate]` weight directly.
      Fusing the gate into the epilogue is a later optimisation, not a v1 requirement.
- [ ] **Pre-norm restructure** — `LayerNorm1D` and `Add` both exist, so pre-norm is
      `LN → GEMM → attn → Add → LN → GEMM → GeGLU → Add`. `LayerNormResidual1D` (post-norm fused)
      is not reusable; the two primitives already in the file cover it.
- [ ] **Sliding-window band** — `AttentionKernels.BatchedAttention` takes a `[B, S]` 0/1 *padding*
      mask, not a dense matrix, so the band is one extra condition (`|qPos - j| > band`) at the
      three places the mask is applied. A `band` parameter of -1 means global.
- [ ] **Fused-QKV weight upload** — the runner re-concatenates separately-stored q/k/v; ModernBERT
      ships `Wqkv` already fused, so add the mirror path (or split on the CPU: 3×1M floats).
- [ ] **`ModernBertGpuRunner`** (~500 lines, following `BertEncoderGpuRunner`) plus
      `modernbert --gpu` dispatch, and the gate below.
- [ ] **Gate: GPU vs CPU, not GPU vs PyTorch.** The CPU encoder is already pinned to PyTorch in
      Phase 1, so diffing the GPU runner against the CPU runner pins it transitively at the cost of
      one fixture. Hold the same bound (`maxRel ~1e-5`) so an F32-vs-F32 divergence is still caught.
- [ ] Expect the naive `BatchedAttention` (it recomputes each score three times and materialises a
      `[B, H, S, S]` score buffer) to dominate at ModernBERT-large's 28×1024 depth/width. A tiled
      or banded GPU attention kernel is the follow-up if the numbers justify it.


## Verification steps

1. ✅ `dotnet build Nivara.slnx` — clean, 0 warnings, 0 errors.
2. ✅ Downloaded the checkpoint (see 1.1).
3. ✅ `python samples/NivaraInference/Python/modernbert_compare.py` — fixture written.
4. ✅ `modernbert` — forward runs; stats are stable across all 10 sentences
   (std 0.986–1.016, min ≈ −24, max ≈ +16), which is the expected shape for a
   final-LayerNorm'd encoder and a good smoke test.
5. ✅ `modernbert compare` — **the gate passed** (table above).
6. ✅ `modernbert benchmark` — record Nivara ms (below).
7. ✅ `python samples/NivaraInference/Python/modernbert_benchmark.py` — written and run in the
   **same session** as the C# side, so the ratio is honest. PyTorch: 813.6 ms median at seq 128,
   1164.4 ms at seq 256.
8. ✅ **Asked before** `dotnet test`. New/changed fixtures: **126 passed, 0 failed**. Full
   `Nivara.Tests` suite: **3516 passed, 1 failed, 14 skipped, 7m46s** — the one failure is
   `TensorsHelperTests.Transpose_PerformanceProbe_TiledKernelBeatsBclViewMaterialization`
   (`[Category("Performance")]`, a timing comparison between the tiled transpose kernel and the
   BCL view+flatten route). It **passes in isolation** (423 ms) and failed only because the full
   run shared the machine. Unrelated to this branch: no softmax, tokenizer, or ModernBERT code is
   involved. Treated as machine-load flake, not a regression.
9. ✅ Smoke-check that the SmolLM/Qwen/DistilBERT modes still run — the `Gpt2BpeTokenizer` and
   `StateDictLoader` edits are shared code, and the tokenizer fix changes byte-level BPE behaviour
   for every legacy-path caller. The existing SmolLM test uses letter-only text and is therefore
   **not** a sufficient guardrail for that change; a real SmolLM `compare` run is. (The new
   punctuation test `"a - b"` → `[81, 731, 278]` pins the fixed order against HF, but it pins the
   *vocab*, not the SmolLM *checkpoint*, so a live run is still worth doing.)
   - **Result**: `smollm compare --dtype float32` on this branch produces output **byte-identical**
     to the base commit `73d0035`, verified by building that commit in a scratch worktree
     (`git worktree add --detach … 73d0035`) and running the same command there. Prompt ids match
     the HF fixture exactly (`[504, 3575, 282, 4649, 314]`) and the 32-token stream is the same.
     No blast radius.
   - **But it surfaced a pre-existing defect, not a regression**: SmolLM greedy generation diverges
     from PyTorch at generated token 30 (`argmax match 25/32`, final-logits `cosine 0.243`,
     `max abs diff 31.9`). Identical on `73d0035`, so it predates this branch. Already disclosed in
     the README, so nothing to correct there; the root cause is filed in the issues log. The first
     25 tokens match exactly, and the same README table shows BF16 matching *better* than F32
     (22/32 but 0.94 cosine vs 25/32 and 0.24), which is the part worth explaining.
   - Qwen could **not** be smoke-checked: `samples/data/qwen2.5-0.5b-instruct` is absent, so its
     parity tests skip and its `Split`-pretokenizer path is covered only by unit tests. Worth doing
     on a machine that has the 989 MB checkpoint. DistilBERT and MiniLM use WordPiece
     (`Microsoft.ML.Tokenizers`), not the byte-level BPE path, so they are untouched by construction.

### CPU timings — PyTorch vs Nivara, same session (2026-09-27, F32, Release, .NET 11)

| padded length | valid tokens | PyTorch median | Nivara median | Slowdown | Nivara tok/s | Nivara ms/layer |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 26 | 813.6 ms (min 751.9) | 2589.5 ms (min 2526.4) | **~3.2×** | 10.0 | 92.48 |
| 256 | 26 | 1164.4 ms (min 1155.7) | 4853.3 ms (min 4663.9) | **~4.2×** | 5.4 | 173.33 |

**Parameter count corrected.** The checkpoint file holds **173** tensors summing to 395,881,664,
but three of them are an MLM head that `ModernBertModel` never instantiates:
`head.dense.weight` (1,048,576), `head.norm.weight` (1,024), `decoder.bias` (50,368) —
**1,099,968** in total. The encoder's own **170 tensors / 394,781,696** is the number that
matches `sum(p.numel() for p in model.parameters())` exactly. The earlier "395,881,664 params"
figure in this plan was the file total, not the encoder.

1510.2 MB as F32; safetensors parse ≈ 2.7 s (all 173 tensors), weight load into modules ≈ 10.7 s
(170 bound). So a cold `benchmark` run is load-dominated: ~13.4 s of setup against a 2.59 s
seq-128 forward.

**The most useful number in the table is not the ratio — it is the disagreement between the two
length columns.** Doubling the padded length costs Nivara **1.87×** but PyTorch only **1.43×**.
PyTorch's sliding-window SDPA skips out-of-band blocks, so 18 of its 28 layers get *cheaper* per
token as the sequence grows; Nivara builds a dense `[L, L]` mask and does the full product
regardless, so its cost tracks `L²` and the window buys nothing. That is the banded-kernel issue in
the log below, now quantified rather than asserted — it is worth ~1.4× at seq 256 and much more
near the 8192-token context, where the dense mask is 268 MB and the cap throws.

## Debug ladder (superseded — see the Phase 1 result section)

The ladder is kept because it was accurate; only its item 1 was applied to a *plan* that was
wrong, not to the code.

1. ✅ `Wi` gate/up halves swapped (this was it).
2. Gate omitted entirely.
3. Layer-0 `attn_norm` skipped/duplicated, or `mlp_norm` applied post-residual.
4. Sliding-window band off by one (64 vs 65) or applied to full-attention layers.
5. RoPE theta swapped between the full (160000) and sliding (10000) layer types.

`compare_diag` (per-stage diff against `output_hidden_states`) is the bisect tool and is now
permanent, so future parity work does not have to rebuild it.

## Blast radius — as executed

- `src/Nivara`: **one behavioural change**, `GradKernels.SoftmaxSingle` / `SoftmaxSingleStrided`
  clamping a fully-masked row to zeros. Reachable only where every key is suppressed, which a
  causal mask never produces, so no shipped model changes behaviour. Now covered by three tests in
  `GradKernelsTests` (the gap this branch originally had).
- `samples/Nivara.Samples/Gpt2BpeTokenizer.cs` — shared by SmolLM, Qwen and their tests. Three
  changes: the additive `LoadFromTokenizerJson` / `EncodeWithSpecialTokens` / opt-in NFC, the
  pre-tokenize-order fix (**a behaviour change for every legacy-path caller**), and the
  `ModernBertConfig`-independent `VocabSize` now including added tokens. The fix makes them match
  HuggingFace, and all pre-existing tests still pass, but see verification step 9.
- `samples/Nivara.Samples/StateDictLoader.cs` — additive `LoadLinearSlice` only;
  `LoadLinear`/`LoadLayerNorm`/`LoadRMSNorm` untouched.
- `samples/Nivara.Samples/ModernBertModel.cs` — `LayerTypes` changed from a plain `init` property
  defaulting to `[]` to a lazily **derived** one. Found by the new tests: a hand-built
  `ModernBertConfig` previously threw `ArgumentOutOfRangeException` from `IsFullAttention` because
  the array was empty, so the type was only usable via `FromJson`. `FromJson` behaviour is
  unchanged (it always set `LayerTypes` explicitly).
- `samples/NivaraInference/Program.cs` — one new `switch` case plus the usage string.
- `.gitignore` — two new lines, no existing rule modified.
- Memory: the 1510 MB F32 checkpoint read into `float[]` tensors plus module copies peaks around
  2.5–3 GB managed heap. Higher than the Qwen BF16 load (which peaked at ~1.88 GB) because F32
  doubles the element size. Worth stating in the README.
- Dense-mask memory: `[L, L]` F32 is 4·L² bytes — 1 MB at 512, 16 MB at 2048, and **268 MB** at
  8192. `MaxDenseLength = 2048` throws above that. (An earlier draft of this plan said ~1 GB at
  8192; that was wrong by ~4×.)


## Planned commits — as executed

1. `cc3fb2f` `docs: add LAYA.md — ModernBERT/Laya research reference and branch plan`
2. `89b487d` `fix(samples): pre-tokenize raw text before byte-level mapping`
3. `7f19f28` `fix(autodiff): clamp fully-masked softmax rows to zero`
4. `6fc0465` `feat(samples): ModernBERT-large encoder with a HuggingFace parity gate`
5. `5697832` `docs: record Phase 1 gate result and correct the Wi activation note`
6. `0bf10ff` `test(samples): cover the ModernBERT encoder, tokenizer order, and masked softmax`
7. `27fbf81` `docs: document ModernBERT, and correct a false tokenizer-divergence claim`
8. ⬜ `docs: Phase 1 close-out` — this file, then deleted by the two-gate review

Fixes 2 and 3 are separate from the feature on purpose: each builds and stands on its own, so a
bisect points straight at whichever one broke a model.

## Open questions for the human

- **Laya reference source.** Port `rl_common.py` into `Python/laya_compare.py` (self-contained,
  no new deps) vs `pip install laya` and diff against the real package (strongest ground truth,
  adds an install). Default to the port, cross-check if the install is cheap.
- **Phase 2 checkpoint.** 842.6 MB for Laya on top of the 1510 MB ModernBERT download. Confirmed
  wanted in G1; download after Phase 1's gate passed, which it now has.
- ~~`samples/data/modernbert/` vs reusing `samples/data/laya/encoder/`~~ — resolved in G1: separate
  directories, Phase 1 stands alone.
- **Is the safe-softmax clamp enough, or should the mask also be applied as a select?** With the
  clamp, ModernBERT matches PyTorch exactly. But a `NaN` already present in q/k/v still is not
  suppressed, because `NaN + (-inf) = NaN`. Nothing in the current model suite produces that, so it
  is filed rather than fixed — but it is a latent trap for any future model that can emit `NaN`
  upstream. Decided in G1: clamp only, file the select variant.

## GitHub issues log

Numbers are filled in as they are created; per the instruction at the top of this file, each is
raised **when the work is deferred**, not at the end.

- [#446](https://github.com/khurram-uworx/Nivara/issues/446) — `LayerNorm<T>` has no bias-free mode (`bias: false`); ModernBERT's `norm_bias: false`
      is emulated by a zero Beta, which is exact for inference but wrong for fine-tuning
      (the zero Beta would take a gradient instead of staying fixed).
- [#447](https://github.com/khurram-uworx/Nivara/issues/447) — banded/sparse attention kernel. A dense `[L, L]` mask cannot serve ModernBERT at
      `max_position_embeddings = 8192` (268 MB per mask), and it makes sliding layers cost the same
      as full ones, which is the dominant cost in the Phase 1 timings. This is the highest-value
      follow-up of the lot.
- [#448](https://github.com/khurram-uworx/Nivara/issues/448) — `GradKernels`: apply the attention mask as a select (forced `-inf`) rather than an add,
      so a `NaN` in q/k/v cannot escape suppression via `NaN + (-inf) = NaN`. The safe-softmax clamp
      removes the known trigger; this removes the class.
- [#449](https://github.com/khurram-uworx/Nivara/issues/449) — ILGPU/OpenCL GPU path for ModernBERT (`BertEncoderGpuRunner` is Post-LN BERT only, and
      there is no RoPE kernel on the GPU path at all). See Phase 3 for the component list.
- [#450](https://github.com/khurram-uworx/Nivara/issues/450) — ModernBERT-specific fused pre-norm block (per-layer band + theta + GeGLU epilogue) as
      a perf follow-up; the CPU path is 28 separate module forwards with two materialized masks.
- [#451](https://github.com/khurram-uworx/Nivara/issues/451) — NFC normalization is only implemented on the new `tokenizer.json` entry point;
      consider promoting it to the shared byte-level BPE path. (The related `added_tokens`
      per-piece concern turned out not to exist — HF matches over raw text, and that is now
      pinned by tests.)
- [ ] #NNN — Laya `act_head` / escalate signal is documented as unusable (AUROC 0.30);
      investigate or explicitly close. **Deliberately not raised yet**: it asserts a measurement
      about a checkpoint this branch has not downloaded. Raise it in Phase 2 once
      `samples/data/laya/` is loaded and the claim is either reproduced or refuted — an issue that
      turns out to be wrong about a model we have never run is worse than no issue.
- [#452](https://github.com/khurram-uworx/Nivara/issues/452) — **pre-existing, surfaced by the Phase 1 blast-radius check**: SmolLM-135M F32 greedy
      generation diverges from PyTorch at generated token 30 (`argmax match 25/32`, final-position
      `cosine 0.243`). Verified byte-identical on `73d0035`, so it predates this branch and is
      unrelated to it. Already **disclosed** in the README's SmolLM diff table (25/32 F32, 22/32
      BF16), and the odd result is that BF16 — the *less* precise dtype — matches far better, which
      suggests the F32 path is accumulating error that BF16's rounding happens to cancel rather than
      a near-tie walk-off. Ask here is a root cause: where do the two implementations' logits begin
      to separate, and why is BF16 closer than F32? It matters because "use BF16 for SmolLM" is
      currently justified by an empirical observation nobody has explained.

