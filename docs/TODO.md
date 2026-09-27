# Laya decision head (Phase 2) - #460

## Problem

`docs/LAYA.md` sequences the Laya port as three phases. Phase 1 (ModernBERT encoder) shipped in
PR #453; Phase 3's encoder (the ILGPU runner) shipped in PR #461 / #449. What is missing is
**Phase 2 - the Laya decision head** - and nothing in the repo implements it. There is no
`LayaDecisionHead<T>`, no prompt builder, no `laya` mode. Without the head the ModernBERT encoder
cannot produce a Laya decision, so the port is not usable.

Phase 3 is folded into Phase 2 here: #449 delivered the GPU encoder, so the remaining GPU work
(the GPU head, plus #440) is part of this scope. The backend was settled as **GPU-only** by
measurement (PR #459, `docs/LAYA.md` "Backend decision"), so the CPU forward this issue builds is
a **parity reference, not a deployment path**.

## Proposed changes

### Unit 1 - `LayaPromptBuilder` + unit tests

`samples/Nivara.Samples/LayaPromptBuilder.cs` - the C# port of the prompt surface in
`laya/common.py` (PyPI `laya` 0.3.20):

- `render_criterion` - strings pass through; structured values become compact JSON.
- `render_options` - `choice` -> `"key: description"` (bare `key` when the description is
  `None`/`""` only; `0` and `False` are real values), `score` -> `"level {i}: {c}"`,
  `noul` -> two options with the resolved labels and the fallback criterion strings.
- `_resolve_noul_labels` - requires exactly `false`/`true` mapped to two distinct non-empty
  strings, stripped; `labels` on a non-noul question raises.
- `build_sequence` - `[CLS] <type> question: <ins> [SEP] [MASK] opt0 ... [SEP] state [SEP]`, with
  the `opt_budget < 16` per-option shrink, the `opt_budget` **recompute**, the
  `head_ids[:max(8, opt_budget)]` clamp, `room = max(0, max_len - len(ids) - 1)`, the
  `truncateLeft` slice, and the `[m for m in markers if m < max_len]` filter.
- `clamp_temperature` (`TEMP_MIN = 0.5`, `TEMP_MAX = 5.0`, NaN/inf/non-numeric -> 1.0) and
  `temp_bucket(qtype, k)`.
- The `agent.py` decode path: `answer_confidence` = `max(p[:k])`, `confidence_from_probs` =
  `1 - H(p)/log(k)`, the `choice` / `score` / `noul` typed answers, the `t_scale` lookup, and the
  4-dp rounding.

Typed input, not raw JSON: a `LayaQuestion` record (`t`, `ins`, ordered `crit`, optional
`labels`) plus a `string stateText`. `crit` is an ordered key/value list for `choice` (Python
dict insertion order is the contract), a list for `score`, and a two-key map for `noul`. A
non-string state throws `NotSupportedException` - see "Out of scope".

### Unit 2 - `LayaDecisionHead<T>` + the biased fused-QKV module

`samples/Nivara.Samples/LayaHeadModel.cs`:

```csharp
// nhead is derived, not hardcoded: a different d changes the head shape and the
// checkpoint would not load, so this must fail loudly rather than mis-load.
int nhead = Math.Max(1, d / 64);
T scale  = T.CreateChecked(1.0 / Math.Sqrt(d / (double)nhead));
```

- **Attention: reuse `BertSelfAttention<T>` (`BertModel.cs:46`), do not write a new one.** G1
  found the plan's planned `LayaHeadAttention<T>` would be a near-duplicate. `BertSelfAttention<T>`
  already has biased q/k/v, a biased `oProj`, the dense `[L,L]` additive mask with
  `T.NegativeInfinity`, and the `1/sqrt(embedDim/numHeads)` scale, and its local `MultiHeadAttention`
  helper (`BertModel.cs:140`) delegates to `ReverseGradOperations.MultiHeadAttention` with the right
  `numHeads` and scale. The checkpoint's *fused* `in_proj` is a **load-time** concern, not a
  runtime one: `LoadLinearSlice` slices the weight row blocks and the new bias-slice helper slices
  `in_proj_bias`. The mask polarity needs no special handling either - the reference's
  `pad = ~attention_mask` (True = ignore) and this convention's "below 0.5 = ignore" are the same
  predicate given a 1/0 `attention_mask`, so the raw mask is passed straight through. AGENTS.md
  rule 8 prefers one authoritative implementation over copies.
- `LayaHeadLayer<T>` - **genuinely new; no existing type fits.** Pre-norm:
  `x + attn(norm1(x))`, then `x + linear2(relu(linear1(norm2(x))))`. Biased `LayerNorm(1024, 1e-5)`
  throughout (PyTorch `nn.LayerNorm`, `eps` default 1e-5). The two candidates were both checked
  and rejected: `BertLayer` is **post**-LN with GELU, and `TransformerBlock` is pre-norm but has a
  GELU MLP, a pre-computed *causal* mask (the head needs a per-call padding mask), and
  `NormType` defaulting to RMSNorm.
- `LayaDecisionHead<T>` - `type_emb` (3x1024) broadcast over the sequence, 2 layers, `scorer`
  (`LayerNorm` -> `Linear` -> `GeluExact` -> `Linear`->1), `act_head`
  (`Linear(1028->256)` -> `GeluExact` -> `Linear(256->2)`), the marker gather, and
  `feats = [top1, top1-top2, ent, k/255]` with `k = max(markers, 2)`, entropy normalised by
  `log(k)`, and the one-option `top2` pad with 0.0.
- `StateDictLoader` - add a bias-slice sibling of `LoadLinearSlice` for the fused QKV bias.

`Forward` takes the encoder output `[L, d]`, `qtype`, and `int[] markerPos`, and returns
`(float[] Logits, float[] ActLogits)`. **Temperature is not applied inside the head** - the
reference registers the buffer and never reads it in `forward`; `agent.py` divides after
`masked_fill`. Expose it to the caller.

### Unit 3 - `laya` mode in `samples/NivaraInference`

`Laya.cs` with the default and `benchmark` sub-modes, wired into `Program.cs` (usage line, model
map, dispatch). Runs the fixture's questions through encoder + head and prints the typed answer,
`answer_confidence`, `confidence`, per-option probabilities, the temperature actually applied, and
the `act_head` reading **with the ~1.0 caveat inline** so a reader is not misled into gating on
it (upstream `NandhaKishorM/laya#185`; the model card documents the same limitation).

### Unit 4 - `Python/laya_compare.py` + the `laya compare` gate

The generator does **not** transcribe the reference. It runs `pip download laya==0.3.20 --no-deps`,
unzips to a temp dir, and loads `common.py` by path with `importlib` (bypassing
`laya/__init__.py`, which pulls in unrelated modules). The reference is therefore the wheel
itself and is structurally incapable of sharing a misreading with the C# port.

Emits into `samples/data/laya/`: `prompts_py.bin` (ids + marker positions per question),
`logits_py.bin`, `act_py.bin`, `answers_py.json` (typed decision, confidences, per-option probs,
the applied temperature and its raw-vs-clamped provenance so the `choice:11+` -> 0.5 clamp is
visible), and `laya_meta.json`.

The C# gate runs **prompt parity first as its own gate**: ids compared byte-exact, a mismatch
fails before any numeric comparison, because a silently different render still "runs". Then the
numeric gate on the marker-scorer logits, then the typed decision and temperature bucket. Skips
cleanly when the wheel is unreachable and the fixtures are absent, matching `modernbert compare`.

### Unit 5 - tests, then documentation

Head composition tests (shapes, and the `~attention_mask` polarity assertion - the single easiest
thing to get backwards), then the doc commit.

- `docs/LAYA.md` rewritten in the `docs/BERT-GPU.md` shape: result and what is gated; what shipped
  (architecture + the decisions below); what we learned; where reality diverged from the
  pre-implementation estimate; what's next. Phase 3 folded into Phase 2, with #440 named as the
  highest-value remaining item because GEMM is ~99.9% of Laya's arithmetic.
- `samples/NivaraInference/README.md` - a Laya section, the fixture commands, and the CPU path
  labelled a reference with the probe's numbers rather than a deployment claim.

## Decisions (settled with the human)

| # | Decision |
|---|---|
| D1 | **Python reference = the real wheel**, loaded via `importlib`. Not a transcription. |
| D2 | **JSON fixture file** for questions + state, read by both sides so the inputs cannot drift. **String states only.** |
| D3 | **Keep the CPU default mode, labelled a reference.** It gates CPU-vs-PyTorch at the 1e-5 class and is the reference the GPU head later diffs. |
| D4 | **B=1, loop per question.** `k = markers.Length`, so `marker_mask` padding slots never exist and the `-1e4` `masked_fill` is a no-op at this batch size. |
| D5 | **Phase 3 folded into Phase 2** - #449 shipped the GPU encoder. |

## G1 grounding outcome (recorded after the plan commit)

Grounded before implementation, as the iterative-work workflow requires.

- **microsoft-learn.** `TensorPrimitives` plus `Vector<T>` / the fixed-width `Vector128/256/512<T>`
  types are the right primitive layer. This work adds **no new kernel** - the head is composed
  from existing `ReverseGradOperations` ops - so the guidance produced no design change. The
  grounding that mattered was reading the wheel and the checkpoint header directly.
- **code-memory.** Confirms the issue: only two `Laya*` symbols exist in the repo, both in the
  probe harnesses (`CpuGemmProbe.ProjectLayaForward`, `GpuAllocProbe.LayaF32Bytes`). The symbol
  index is partial for the generic methods in `ReverseGradOperations.cs`, so those were verified
  against source rather than the index.
- **Reference environment** (checked, not assumed): `torch` 2.13.0+cpu, `transformers` 5.14.1,
  `numpy` 2.2.1, `pip` 26.1.2. `transformers` >= 5 matters - the Laya config carries the newer
  `rope_parameters` key, which the shim path only emulates. `laya` is **not** installed, which is
  precisely why D1 downloads the wheel and loads `common.py` by path instead of installing.
- **Norm placement confirmed in the installed reference.** `transformers`
  `ModernBertEncoderLayer.__init__` is literally `if layer_idx == 0: self.attn_norm =
  nn.Identity()`, matching both the checkpoint (28 layers, `mlp_norm` on all, `attn_norm` on 27)
  and what `ModernBertLayer` hard-codes. The one untested assumption is now a checked fact.
- **Amendment (Unit 2).** The planned `LayaHeadAttention<T>` was dropped as duplication - see
  Unit 2 above. Strict reduction, no behaviour change, no new risk; the plan is amended rather
  than re-litigated.
- **Blast radius** as documented below, with one line added: `StateDictLoader` gains a member
  (additive, no existing caller changes) and `BertSelfAttention<T>` is reused read-only.
  **`src/Nivara` is still untouched.**

## G1 corrections - `docs/LAYA.md` §4 vs `laya/common.py` 0.3.20

Re-derived line by line against `laya/common.py` and then **empirically checked where a claim was
behavioural**. Six stand. Three did not survive - they are struck out below rather than deleted, so
the record shows what a first reading of the reference got wrong.

**Why this mattered enough to re-check:** the first pass produced nine candidate corrections. Three
were wrong, and two of those would have shipped a false claim into a durable reference document
while looking rigorous. #1 and #7 in particular were asserted confidently and are both falsified by
a two-line probe. The lesson is the same one D1 already encodes - a plausible reading of a
reference is not a verified one.

### Stands - real misstatements or omissions in §4

2. **State truncation direction (§4.380).** §4 says "take the *last* `room` tokens of the state".
   The wheel's `build_sequence` default is `truncate_left=False` -> `state_ids[:room]`, the
   **first** `room` (`common.py:118`). `agent.py:568` sets `truncate_left = isinstance(state, list)`,
   so left-truncation is the exception for conversation lists, not the rule. **A port copying §4
   would keep the wrong end of a long state and still run.**
3. **`room` is never defined in §4.** The wheel is `room = max(0, max_len - len(ids) - 1)`
   (`common.py:114`); the `- 1` pays for the trailing `[SEP]`. §4 line 369 uses `room` as if given.
4. **Marker filtering omitted.** The return is `ids[:max_len], [m for m in markers if m < max_len]`
   (`common.py:120`). §4 line 370 returns `marker_positions` unfiltered, so a port can end up with
   `len(markers) != len(opts)` - which is exactly what `agent.py:582` raises on.
5. **Temperature is not applied in the head.** `DecisionModel.__init__` registers the buffer and
   `forward` never reads it (`common.py:136`, `:139-175`); `agent.py:666` divides by `t_scale`
   *after* the logits come back. §4 states the values but not the application point, which is the
   part a port has to match.
6. **Noul criterion fallbacks are prompt text.** `"no, the statement does not hold"` /
   `"yes, the statement holds"` (`common.py:64,66`). §4 line 374 elides them behind `…`, so a
   port that drops them changes the prompt for any `noul` question without a `false`/`true`
   criterion.
7. **Undocumented validation.** `_resolve_noul_labels` strips whitespace and requires two distinct
   non-empty strings (`common.py:43-46`); `render_criterion` JSON-dumps structured values with
   `separators=(", ", ": ")` and `ensure_ascii=False` (`common.py:32`).

### Falsified - struck out, do not propagate

1. ~~**Option tokenization is capped inside the tokenizer call.**~~ The wheel does use
   `tok(..., truncation=True, max_length=48)`, but §4's `tok(...)[:48]` produces **byte-identical
   ids** - verified: `truncation=True, max_length=48` == `[:48]` on a 200-token description
   straddling the boundary (`True len 48`). The wheel's own comment says the slice "is exactly what
   the previous slice produced"; the change was about not tokenizing a long tail, i.e. throughput.
   The earlier claim that it "differs when a token straddles the 48 boundary" was simply wrong.
   §4 line 364 is correct as written.
8. ~~**The state is truncated at `model_max_length` = 8192.**~~ Verified against the real Laya
   tokenizer: tokenizing a 20,001-token state with no `max_length` **returns all 20,001 ids** and
   emits only a warning ("Token indices sequence length is longer than the specified maximum
   sequence length ... will result in indexing errors"). HF does not truncate there. The state is
   bounded solely by `room`, i.e. by `max_len` = 512. **So the C# port must not impose an 8192
   cap** - doing so would have been the divergence I was about to introduce.
9. ~~**`opt_budget` is recomputed after the shrink and §4 uses the stale value.**~~ §4 line 368 is
   `head_ids[:max(8, head_max_len - sum(len(o) for o in opt_ids))]`, evaluated *after* the shrink
   block, so it already uses the recomputed sum and is equivalent to the wheel's two-step
   `opt_budget = ...` / `head_ids[:max(8, opt_budget)]`. §4 line 368 is correct.

## Already free (verified - do not rebuild)

- **Encoder.** `ModernBertEncoder.LoadWeights(tensors, config, prefix: "encoder")` exists and its
  doc comment names Laya. The 170 `encoder.*` keys match exactly: `embeddings.tok_embeddings.weight`
  / `embeddings.norm` / `final_norm`, and per layer `attn.Wqkv` [3072,1024] / `attn.Wo` [1024,1024]
  / `mlp.Wi` [5248,1024] / `mlp.Wo` [1024,2624], with `mlp_norm` on all 28 layers and `attn_norm`
  on layers 1-27 only - exactly what `ModernBertLayer` hard-codes. No biases under `encoder.*`.
- **Config.** `samples/data/laya/encoder/config.json` is architecturally identical to the stock
  ModernBERT-large config (d=1024, 28L, inter=2624, 16 heads, local_attention 128, every-3 global)
  and differs only in using the newer `layer_types` + `rope_parameters` keys, which
  `ModernBertConfig.FromJson` already handles. **This parse path is untested** - Phase 1's gate
  exercised the legacy flat keys against the stock checkpoint.
- **Tokenizer.** `Gpt2BpeTokenizer.LoadFromTokenizerJson` is the ModernBERT path. The Laya
  `tokenizer/tokenizer.json` is structurally identical (NFC normalizer, ByteLevel
  `add_prefix_space:false`, 6878 vocab / 50009 merges / 116 added, same TemplateProcessing
  specials) but a different file hash, so it needs its own id parity check.
- **Head ops.** All present in core, no new op: `LayerNorm.Bias`, `Linear.Bias`,
  `MultiHeadAttention(q,k,v,numHeads,scale,mask)` with an additive `[qLen,kvLen]` mask using
  `T.NegativeInfinity`, `GeluExact` (matches PyTorch `nn.GELU()`'s exact-erf default), `Gather`,
  `Concat`, `Add`, `Softmax`, `LogSoftmax`, `Embedding.Forward(int[])`.
- **Checkpoint header** (read directly): 206 tensors - `encoder.*` 170, `head.*` 24 (2 x 12),
  `scorer.*` 6, `act_head.*` 4, `type_emb.weight` [3,1024], `temperature` [3]. All F16 except
  `temperature`, which is F32.
- **GPU epilogues.** `GemmKernels` already has `TiledGemmKernelRow4Qkv` and
  `TiledGemmKernelRow4Relu`, and the probe timed both on Laya shapes at parity with their
  siblings (`qkv` 202 vs 202 GMAC/s; `head ff2` `Row4Relu` 155 vs `Row4Bias` 155). The GPU head
  is wiring, not kernels.

## Blast radius

**`src/Nivara` is not touched.** Every new type is sample-side. The one pre-existing type this
adds a member to is `StateDictLoader` (in `samples/Nivara.Samples`), and it is additive only.

| Change | Files | Downstream impact |
|---|---|---|
| `LayaPromptBuilder`, `LayaQuestion` | new, `samples/Nivara.Samples` | none - new surface |
| `LayaDecisionHead<T>`, `LayaHeadLayer<T>`, `LayaHeadAttention<T>` | new, `samples/Nivara.Samples` | none - new surface |
| `StateDictLoader.LoadLinearBiasSlice` | additive, `samples/Nivara.Samples` | none - no existing caller changes |
| `Laya.cs` + `Program.cs` dispatch | `samples/NivaraInference` | `Program.cs` gains one `case` + the usage line; the switch is exhaustive over `modelType` and every existing case is untouched |
| `laya compare` | new sub-mode | additive; the fixture-absent path must skip, not throw |
| Tests | new, `tests/Nivara.Tests` | none - additive |
| Docs | `docs/LAYA.md`, `samples/NivaraInference/README.md` | none |

Covered by the existing suite: nothing changes behaviourally, so the existing 1948 tests are the
regression guardrail. The new tests cover the head composition and the prompt edge cases.

**Memory.** Loading the 822 MB F16 checkpoint through `SafeTensorsLoader.Read<float>` widens all
421M params to F32 = **1.685 GB** of managed arrays (the same pattern the stock 1510 MB F32
ModernBERT already uses at 3 GB). Both the encoder and the head need to be resident for a
`laya` run, so the peak is the whole checkpoint, not one half. The probe measured the GPU
allocating that same working set (1.569 GiB) in 64 MiB chunks; the host side has not been measured
at 1.685 GB for this checkpoint specifically.

## Out of scope (decided, recorded not deferred silently)

- **Structured (dict/list) states and structured criteria.** `serialize_state` and
  `render_criterion` JSON-dump with Python's `ensure_ascii=False` and `separators=(", ", ": ")`.
  Matching that byte-for-byte from C# is its own port with its own float/exponent edge cases, and
  D2 scopes the gate to string states. A structured state throws `NotSupportedException` naming
  this decision.
- **Batched multi-question forward.** Each question has its own sequence and valid length, and
  the encoder is 2D `[L, d]` only. D4 loops at B=1.
- **Multilingual / typed-decisions subfolders.** Noted optional in §9, untouched.
- **`act_head` as a signal.** Reproduced faithfully (4 tensors, `forward` computes it
  unconditionally) and exposed, but the ~1.0 reading is an upstream model limitation, not a Nivara
  defect - so no Nivara issue is filed. The cheap check is whether `act_probability` reads ~1.0 for
  arbitrary synthetic inputs, which needs no eval data and no labels.

## Verification steps

1. `dotnet build Nivara.slnx -c Release` - 0 errors, 0 warnings, after **each** unit.
2. The `laya` mode runs the encoder + head end to end on CPU. Confirmed against the real
   822 MB checkpoint in `samples/data/laya/`.
3. `laya compare` - prompt ids byte-exact, then marker-scorer logits, then the typed decision and
   temperature bucket. **Requires the wheel download and a PyTorch environment - ask the human
   before running.**
4. `dotnet test` - **ask the human before running.**
5. `git show --stat` against the intended message before moving to the next unit.

## Planned commits

1. `docs: plan the Laya decision head (Phase 2) in TODO.md` - this file.
2. `feat: add the Laya prompt builder as a port of the wheel's build_sequence`
3. `feat: add the Laya decision head with a biased fused QKV projection`
4. `feat: add a laya mode running the encoder and head end to end`
5. `perf: gate the Laya prompt and head against the laya 0.3.20 wheel`
6. `test: cover the Laya head composition and the mask polarity`
7. `docs: rewrite LAYA.md as an implementation reflection and mark Phase 2 complete`

Additive fix/test commits are permitted after any unit. No amend, no rebase, no squash.

## GitHub issues log

- [ ] #460 - this work (Laya decision head, Phase 2)

As each task executes, if you find deferred work or a concern (known limitations, follow-ups,
refactors) outside this plan, create it immediately (`gh issue create --repo khurram-uworx/Nivara`)
and record the number here. Do not rely on memory - compaction during execution can lose it.

## Reference

- `laya/common.py` from PyPI `laya` 0.3.20 is the **maintained** reference. The HF repo's
  `rl_common.py` is **stale** and has three defects, two of which fail silently (`docs/LAYA.md` §4.1).
  Re-verify the source with `pip download laya==0.3.20 --no-deps` - nothing needs installing.
