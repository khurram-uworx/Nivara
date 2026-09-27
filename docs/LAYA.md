# Laya (and ModernBERT) in Nivara — research reference

Durable reference for running [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
and its ModernBERT backbone in Nivara. Everything here was verified against primary sources
(the HF checkpoint's own safetensors header, the `answerdotai/ModernBERT` reference implementation,
and the installed `transformers` 5.14.1 source) rather than assumed.

This file is the "why" that must not be re-derived: architecture semantics, tensor maps, prompt
format, reuse map, and the traps. It is also where the remaining phases are tracked — §9 carries the
Phase 2 and Phase 3 scope and the decisions still open, so the phase-1 working plan
(`docs/TODO.md`) was retired once its two review gates cleared and nothing here needed to move.

---

## 1. What Laya is

- **Non-autoregressive, single-forward-pass decision model.** You give it a *state* (text, email,
  ticket, JSON) plus *typed questions*; it returns typed answers with calibrated probabilities.
  It never generates text, so there is no decoding loop and nothing to hallucinate.
- **Checkpoint family** (all in the one HF repo; only the requested subfolder downloads):

  | Checkpoint | Backbone | Params | Context |
  |---|---|---|---|
  | `convaiinnovations/laya` (repo root) | ModernBERT-large | 421M | 512 |
  | `convaiinnovations/laya` → `multilingual/` | mmBERT-base | 322M | 1024 (8k capable) |
  | `convaiinnovations/laya` → `typed-decisions/` | ModernBERT-large | 421M | 1024 |

- **Training**: RLCD — reinforcement learning against strictly proper scoring rules (log +
  spherical + RPS for ordinal), REINFORCE with a group-mean baseline. So reported probabilities
  are the thing being optimised; logits alone are not the product.
- **The English root checkpoint's encoder is `answerdotai/ModernBERT-large`, fine-tuned.**
  `rl_agent_config.json` says `"encoder": "answerdotai/ModernBERT-large"` and the safetensors
  header matches ModernBERT-large's encoder tensor-for-tensor (same names, `encoder.` prefix,
  same shapes). Laya adds a from-scratch decision head on top.

### Repo file inventory (HF, `convaiinnovations/laya`)

```
model.safetensors            # single file, F16 on disk, 842,609,210 B (842.6 MB / 803.6 MiB), 206 tensors
encoder/config.json          # ModernBERT-large config (the backbone config)
rl_agent_config.json         # head depth, budgets, temperatures, act costs
tokenizer/tokenizer.json     # ModernBERT byte-level BPE (same tokenizer as the backbone)
tokenizer/tokenizer_config.json
rl_common.py                 # prompt rendering + DecisionModel + scoring rules  <-- the spec
rl_agent_api.py, email_utils.py
typed-decisions/…, multilingual/…
```

Note there is **no `model.safetensors.index.json`** — one monolithic F16 file.

---

## 2. The tensor map (read from the safetensors header, not guessed)

206 tensors, all `F16` except `temperature` (`F32`).

Re-verified 2026-09-27 against the downloaded `model.safetensors` (842,609,210 B; 21,536 B
header; 842,587,666 B data; no `__metadata__`). Every shape and dtype above matches, as do the
170/3/167 tensor counts and the 5-vs-6 split across layers 0 and 1–27. Total **421,293,830**
parameters, partitioned:

| Prefix | Tensors | Params |
|---|---|---|
| `encoder.*` | 170 | 394,781,696 |
| `head.*` | 24 | 25,192,448 |
| `scorer.*` | 6 | 1,052,673 |
| `act_head.*` | 4 | 263,938 |
| `type_emb.*` | 1 | 3,072 |
| `temperature` | 1 | 3 |

**Every model weight is F16 on disk, not F32.** Phase 1's parity gate compared a F32 encoder
against HuggingFace, so Phase 2 cannot reuse its tolerance unchanged: the parity gate has to
decide whether to compare in F16 (matching the checkpoint, and so measuring the same thing the
checkpoint will do at inference) or widen to F32 on both sides (isolating the *port* from the
*storage precision*). Widening is the cleaner gate — it tests the port rather than F16 rounding
— but it must be applied identically on both sides or the comparison measures nothing.

### Encoder — ModernBERT-large, `encoder.*` prefix

| Key | Shape | Notes |
|---|---|---|
| `encoder.embeddings.tok_embeddings.weight` | `[50368, 1024]` | |
| `encoder.embeddings.norm.weight` | `[1024]` | LayerNorm, **no bias** |
| `encoder.layers.N.attn.Wqkv.weight` | `[3072, 1024]` | **fused Q‖K‖V**, no bias |
| `encoder.layers.N.attn.Wo.weight` | `[1024, 1024]` | no bias |
| `encoder.layers.N.attn_norm.weight` | `[1024]` | **absent for N=0** (layer 0 = Identity) |
| `encoder.layers.N.mlp.Wi.weight` | `[5248, 1024]` | **fused `input‖gate`**, no bias |
| `encoder.layers.N.mlp.Wo.weight` | `[1024, 2624]` | no bias |
| `encoder.layers.N.mlp_norm.weight` | `[1024]` | LayerNorm, no bias |
| `encoder.final_norm.weight` | `[1024]` | LayerNorm, no bias |

170 encoder tensors = 3 non-layer (`embeddings.norm`, `embeddings.tok_embeddings`, `final_norm`)
+ 167 layer tensors (layer 0 holds 5 — no `attn_norm`; layers 1-27 hold 6 each: 5 + 27×6 = 167).
28 layers, hidden 1024, 16 heads (headDim 64), intermediate 2624.

**There is no position-embedding tensor and no token-type-embedding tensor.** Positions come
purely from RoPE. There are no biases anywhere in the encoder, and no QK-norm.

### Decision head — the from-scratch part

| Key | Shape | Module |
|---|---|---|
| `head.layers.{0,1}.self_attn.in_proj_weight` / `.in_proj_bias` | `[3072, 1024]` / `[3072]` | fused Q‖K‖V **with bias** |
| `head.layers.{0,1}.self_attn.out_proj.weight` / `.bias` | `[1024, 1024]` / `[1024]` | |
| `head.layers.{0,1}.linear1.weight` / `.bias` | `[4096, 1024]` / `[4096]` | FFN 4·d |
| `head.layers.{0,1}.linear2.weight` / `.bias` | `[1024, 4096]` / `[1024]` | |
| `head.layers.{0,1}.norm1.weight` / `.bias` | `[1024]` | LayerNorm **with bias** |
| `head.layers.{0,1}.norm2.weight` / `.bias` | `[1024]` | LayerNorm **with bias** |
| `type_emb.weight` | `[3, 1024]` | question-type embedding (choice/score/noul) |
| `scorer.0.weight` / `.bias` | `[1024]` | LayerNorm |
| `scorer.1.weight` / `.bias` | `[1024, 1024]` / `[1024]` | |
| `scorer.3.weight` / `.bias` | `[1, 1024]` / `[1]` | option-marker scorer |
| `act_head.0.weight` / `.bias` | `[256, 1028]` / `[256]` | input is `d + 4` features |
| `act_head.2.weight` / `.bias` | `[2, 256]` / `[2]` | act / escalate |
| `temperature` | `[3]` | per-qtype calibration, F32 |

`scorer` is a `nn.Sequential` — indices 0/1/3 are the modules, index 2 is `nn.GELU()` (no params).
`act_head` likewise: 0 = Linear, 1 = GELU (no params), 2 = Linear.

---

## 3. ModernBERT semantics (grounded — and two corrections)

I initially assumed ModernBERT used RMSNorm + layer-scale. **Both are wrong.** Verified:

### 3.1 Norm is bias-free LayerNorm, not RMSNorm

`transformers/models/modernbert/modeling_modernbert.py:319` hard-codes
`nn.LayerNorm(config.hidden_size, eps=config.norm_eps, bias=config.norm_bias)` with
`norm_bias: false` in both the backbone config and Laya's `encoder/config.json`. The
`answerdotai` reference exposes `get_norm_layer()` over `{layernorm, triton_layernorm, rmsnorm,
triton_rmsnorm}` (`src/bert_layers/normalization.py:91-114`), but the released
HF-converted config/weights use **LayerNorm with `bias=False`**, and LayerNorm subtracts the
row mean. The `norm_bias` config knob only makes sense for LayerNorm — it is the tell.

### 3.2 There is no layer scale

This is the highest-value correction, because layer scale is widely (wrongly) listed as a
ModernBERT feature. The `answerdotai` pre-norm layer is
`src/bert_layers/layers.py:282-304`:

```python
if config.skip_first_prenorm and config.embed_norm and layer_id == 0:
    self.attn_norm = nn.Identity()
else:
    self.attn_norm = get_norm_layer(config)
...
def compiled_mlp(self, hidden_states):
    return self.mlp(self.mlp_norm(hidden_states))     # line 304
...
attn_out = hidden_states + self.attn(self.attn_norm(hidden_states), ...)   # line 323
```

Plain residual adds. HF agrees (`modeling_modernbert.py:325-340`): `hidden_states +
attn_output`, `hidden_states + self.mlp(self.mlp_norm(hidden_states))`. No `gamma` in either.
The `attn_norm`/`mlp_norm` weights are **ordinary LayerNorm gammas**, not layer-scale gammas —
so there is no "layer scale" op to build. (Corroborating: `model.layers.0` has no
`attn_norm.weight` in the checkpoint, consistent with `attn_norm = Identity` at layer 0.)

### 3.3 The rest of the encoder spec

- **Structure**: `tok_embeddings` → `LayerNorm` (no bias) → 28 × pre-norm layer → `final_norm`.
  Per layer: `h = h + attn(attn_norm(h))`; `h = h + mlp(mlp_norm(h))`.
- **Attention**: fused `Wqkv` `[3072, 1024]` → `view(..., 3, -1, head_dim)` then
  `permute(2, …).unbind(0)` (`modeling_modernbert.py:278-279`), so the split is
  **Q = rows `[0:1024]`, K = `[1024:2048]`, V = `[2048:3072]`** (head-major within each).
  `Wo` has no bias. Scale `1/sqrt(64)`. **Bidirectional** — no causal mask.
- **RoPE**: full head-dim (64), `inv_freq = base^(-arange(0, 64, 2)/64)`,
  `emb = cat(freqs, freqs)`, applied as `(q·cos) + (rotate_half(q)·sin)` — the HF
  **half-split `rotate_half`** layout, i.e. *identical* to Llama/`RotaryEmbedding<T>` in Nivara.
  `theta` is **per layer type**: `160000` for full attention, `10000` for sliding.
- **Sliding window**: `config.sliding_window = local_attention // 2 = 64`, and HF's bidirectional
  overlay is `abs(q_idx - kv_idx) <= sliding_window` (`masking_utils.py:141-151`). So the **half-width
  (`band`) is 64** and a query therefore sees **129 keys** (64 each side plus itself). Do not conflate
  the two: `ModernBertConfig.Band` and `ModernBertMasks.Build` both use 64, and 129 is the
  keys-seen count, not the band.

  HF's two attention backends reach that same 129 by different arithmetic, and it is worth knowing
  both so the numbers are not misread. The SDPA path above uses `config.sliding_window` directly. The
  flash-attention path bumps it first (`modeling_modernbert.py:258-260`), with HF's own comment
  recording that `config.sliding_window` is the "half-window size, e.g. 64 for local_attention=128"
  and that the `+1` "is needed because flash attention sets inclusive boundaries" — giving
  `self.sliding_window = 65`. Then `modeling_flash_attention_utils.py:652-653` emits
  `flash_kwargs["window_size"] = (sliding_window - 1, sliding_window - 1)` = `(64, 64)`. The `+1`
  and the `−1` cancel exactly; flash-attn's `(64, 64)` is inclusive, so it is 64 left + self + 64
  right = the same 129 keys. The `+1` is an inclusive-boundary adjustment, **not** a different band.

  The `answerdotai` reference reaches the same 64 **from a different convention**, which is worth
  stating because the two `sliding_window` fields are *not* the same number:
  `src/bert_layers/configuration_bert.py:208` stores `sliding_window` as the **full** width and
  documents it as "window size `n` … split between the [two sides]", then
  `src/bert_layers/attention.py:311` halves it — `window_size = (config.sliding_window // 2,
  config.sliding_window // 2)` — which is `(64, 64)` when their `sliding_window` is 128. HF instead
  publishes `sliding_window` **already halved** at 64 and never halves it again, so re-halving HF's
  value would wrongly give 32. Both land on a half-width of 64; HF is the authority here.
- **Layer types**: `layer_types` starts `"full_attention"` and repeats every 3
  (`global_attn_every_n_layers = 3`) → layers 0, 3, 6, …, 27 are full (10), the other 18 are
  sliding. **Band mask is per-layer, not per-head** — one `[L, L]` additive mask serves a layer.
- **MLP (GeGLU)**: `modeling_modernbert.py:89-91`
  ```python
  def forward(self, hidden_states):
      input, gate = self.Wi(hidden_states).chunk(2, dim=-1)
      return self.Wo(self.drop(self.act(input) * gate))
  ```
  So **`Wi` rows `[0:2624]` (`input`) carry the activation and rows `[2624:5248]` (`gate`) do
  not** — the product is `act(input) * gate`. The variable HF calls `gate` is the *unactivated*
  one; this reads backwards and was the hardest bug in Phase 1 (cosine 0.82 at layer 0). The
  loader hands the **first** row block to the activated projection.
  `act` for ModernBERT-large is `hidden_activation: "gelu"` = **exact erf GELU** (HF `gelu`, not
  `gelu_new`), matching `ReverseGradOperations.GeluExact` in Nivara.
- Config: `hidden_size 1024`, `intermediate_size 2624`, `28` layers, `16` heads,
  `norm_eps 1e-5`, `local_attention 128`, `global_rope_theta 160000`,
  `local_rope_theta 10000`, `max_position_embeddings 8192`, `vocab_size 50368`,
  `tie_word_embeddings true`, `layer_norm_eps 1e-5`, `position_embedding_type "absolute"`
  (that flag refers to the rotary type, not to a learned position table).
- **Padding**: ModernBERT's native path unpads + FlashAttention, which is mathematically
  identical to a standard additive padding mask on a padded batch. The sdpa path HF uses
  (`build_model(..., attn_implementation="sdpa")`, which is what Laya does) builds exactly
  that padding mask — so a dense additive mask is the correct reference semantics, not a
  approximation.
- **A bidirectional band can leave a query row with no visible key.** With `local_attention 128`
  the half-window is 64, so any query row beyond `valid_len + 64` has its whole valid range
  outside the band and is fully masked. This is unreachable for a causal model and is the first
  thing to check when a *bidirectional* encoder produces `NaN`. Verified in Phase 1: the clamp makes
  those rows finite on the Nivara side (zeros). It does **not** make the two implementations agree
  on those rows — HF masks with `torch.finfo(dtype).min`, so its row max is finite and it produces a
  *uniform* distribution, not zeros. That is why the gate diffs the valid prefix only. Without the
  clamp, the `NaN` compounds across layers because the mask is an *add* and
  `NaN + (-inf) = NaN`.
- **QK-norm: ModernBERT-large has none.** The stock checkpoint holds exactly 6 tensors per layer
  (`attn.Wqkv.weight`, `attn.Wo.weight`, `attn_norm.weight`, `mlp.Wi.weight`, `mlp.Wo.weight`,
  `mlp_norm.weight`) and 5 at layer 0, which has no `attn_norm`. QK-norm is a base-vs-large
  distinction, and it is easy to remember backwards.

### 3.4 Reading HF's per-stage output (a trap worth writing down)

`model(**inputs, output_hidden_states=True)` on `ModernBertModel` returns `num_layers + 1`
states, but **not** the obvious "embeddings, then one per layer". Measured on the 28-layer
stock model:

- `hidden_states[0]` = post-embedding-norm state
- `hidden_states[1 .. 27]` = the output of layers 0 .. 26
- `hidden_states[28]` = the **final-norm** state, i.e. `last_hidden_state`
  (verified `torch.equal(hidden_states[28], last_hidden_state[0])`)

The last layer's **raw** output is never exposed, and that matters: ModernBERT's residual stream
reaches absmax ≈ 2.57e4 there, which `final_norm` rescales to ≈ 27.6. Diffing a raw last-layer
output against `hidden_states[28]` therefore reports a "difference" of ~25708 that is nothing but
the missing LayerNorm. Diff the final norm against `hidden_states[28]` instead.

This is what `compare_diag` consumes, so a parity failure can be localised to a layer in one run.

### 3.5 Tokenizer

`tokenizer/tokenizer.json`: BPE, `pre_tokenizer = ByteLevel{use_regex: true}`,
`normalizer = NFC`, `decoder = ByteLevel`, inline `model.vocab` (50280 entries) + `model.merges`,
`added_tokens` = 116 entries whose ids are `0, 1, 50254-50276` and `50277-50367` (i.e. the 23
whitespace runs occupy 50254-50276 and everything from 50277 up is contiguous).
`tokenizer_config.json` declares
`model_input_names: ["input_ids", "attention_mask"]` — **no `token_type_ids`**.

**Specials — 7 of them, not 4** (all declared `special: true`, `normalized: false`):
`1` = `<|padding|>`, `50279` = `<|endoftext|>`, `50280` = `[UNK]`, `50281` = `[CLS]`,
`50282` = `[SEP]`, `50283` = `[PAD]`, `50284` = `[MASK]`. Note `<|padding|>` sits at id **1** and
`[UNK]` at **50280** — both *below* the 50280-entry base vocab's specials, and `1` collides with a
base-vocab id (see the collision note below).
This is the plain GPT-2-style byte-level BPE, i.e. the `Gpt2BpeTokenizer` path (no `Split`
pretokenizer, unlike Qwen). The repo ships **no `vocab.json` / `merges.txt`**, only
`tokenizer.json` — so the loader must read vocab/merges inline from the JSON.

⚠️ **Run the pre-tokenizer pattern over the RAW text, then byte-map each chunk** — not the other
way round. Byte level maps a space (0x20) to `Ġ` (U+0120), which `\p{L}` classifies as a letter,
so matching on the mapped string gives `"Ġ"` + `"2026"` where HF gives `"Ġ20"` + `"26"`
(ids 209, 938 vs 1384). Letter runs hide the bug, because `"Ġ"` plus letters is still one
all-letter chunk. This is fixed in `Gpt2BpeTokenizer` but it is the single easiest thing to get
wrong when porting any byte-level BPE. On SmolLM the visible form is punctuation: `"a - b"`
byte-maps to `"aĠ-Ġb"`, which the mapped-string order chunks as `["aĠ", "-Ġ", "b"]` and so can
never emit the real `" -"` token that HF produces (id 731).

**`added_tokens` are matched over the RAW text, leftmost-longest — which is what HF does.**
Measured against `AutoTokenizer`, not assumed:

| input | HF ids | pieces |
| --- | --- | --- |
| `"a  b"` | `66, 50276, 67` | `a`, `␣␣` (the 2-space added token), `b` |
| `"a" + 24×" " + "b"` | `66, 50254, 67` | `a`, `␣×24` (longest added token), `b` |
| `"a" + 25×" " + "b"` | `66, 50254, 270` | 24 spaces, then `␣b` — the *leftover* space starts a new chunk |
| `"mail \|\|\|EMAIL_ADDRESS\|\|\| here"` | `5719, 209, 50277, 1060` | added token found mid-string |
| `"[unused1]"` | `50286` | whole string is an added token |

The 25-space row is the decisive one: it can only be produced by leftmost-longest raw-text
matching, because the GPT-2 pattern never emits a whitespace-only piece that ends mid-run, so
per-piece extraction could not find a 24-space token there. ModernBERT declares 116 added tokens
(`tokenizer.json`), which partition exactly as follows — measured, not estimated:

| Group | n | ids | contents |
| --- | --- | --- | --- |
| `\|\|\|…\|\|\|` address markers | 3 | `0`, `50277`, `50278` | `\|\|\|IP_ADDRESS\|\|\|`, `\|\|\|EMAIL_ADDRESS\|\|\|`, `\|\|\|PHONE_NUMBER\|\|\|` |
| whitespace runs | 23 | `50254`-`50276` | runs of 24 spaces down to 2 |
| specials | 7 | `1`, `50279`-`50284` | see the specials list above |
| `[unusedN]` | 83 | `50285`-`50367` | `[unused0]` at **50285** upward |

Three things here are easy to get wrong and were wrong in an earlier draft of this file: the
whitespace range ends at **50276**, not 50263; the address-marker family is **3 tokens**, not
`0` plus `50277-50285` (that range also contains 6 specials); and `[unusedN]` starts at **50285**,
not 50286. The marker family is named `PHONE_NUMBER`, not `PHONE_ADDRESS`.

⚠️ **`AutoTokenizer.vocab_size` is 50280 here, but the real vocabulary is 50368.** HF reports
only the base BPE `model.vocab` and excludes the 116 `added_tokens`, while the embedding matrix
is `[50368, 1024]` and the ids in use run to 50367. Sizing an embedding (or an id→piece table, or
an output projection) from `tokenizer.vocab_size` silently loses the last 88 ids, which is
precisely the range holding `[MASK]` (50284) and the whole `[unusedN]` family (50285-50367). Take
50368 from `config.vocab_size` / the checkpoint's embedding shape, or from
`Tokenizer.get_vocab_size()` on the raw backend (50368), never from `AutoTokenizer`.

**The tokenizer lives in a subdirectory, so the load path matters.**
`AutoTokenizer.from_pretrained("samples/data/laya")` **fails** — there is no top-level
`tokenizer.json`, so it falls through to slow-tokenizer conversion and raises. Point it at
`samples/data/laya/tokenizer/`, where it loads as `TokenizersBackend` with
`mask_token_id=50284`, `cls_token_id=50281`, `sep_token_id=50282`, and produces ids identical to
the raw `tokenizers.Tokenizer` backend (verified on `"hello world"` → `[25521, 1533]` and
`" [MASK] option"` → `[50284, 4500]`). Note the two backends *display* that second one
differently (`' [MASK]'` vs `'[MASK]'`) because of added-token whitespace normalization, but the
ids agree — so compare ids, never token strings. `build_sequence` only ever needs
`cls_token_id`, `sep_token_id`, `mask_token`, `mask_token_id` off the tokenizer.

**Added-token ids that collide with the base vocab.** Ids `0` and `1` are declared added tokens
*and* already exist in the 50280-entry base vocab, so the 116 added tokens span only **50368
distinct ids** — which is `config.json`'s `vocab_size`, and HuggingFace's `len(tokenizer)`. Do not
compute it as `50280 + 116`. The collision resolves in favour of the added token on HF's side too
(`convert_ids_to_tokens(0)` is `|||IP_ADDRESS|||`), so overwriting the base entry is correct
behaviour, not a bug.

**NFC scope — one real divergence, and one non-divergence that is easy to mistake for one.**
`Gpt2BpeTokenizer` never reads the `normalized` field of an added token. It normalizes the *entire
input text* once, up front, whenever `normalizeNfc` is set — so every added token, including the 7
declared `normalized: false`, is matched against **normalized** text. HF instead matches a
`normalized: false` added token against **pre-normalization** text. That is a genuine difference in
principle, but it is unobservable here: the only `normalized: false` entries are the 7 ASCII
specials (`<|padding|>`, `<|endoftext|>`, `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`), and NFC
leaves ASCII unchanged. Worth stating precisely because the earlier draft of this file asserted the
opposite — that Nivara honoured the per-token flag and matched the specials un-normalized.

The real remaining divergence is different: NFC is applied only through `LoadFromTokenizerJson`, not
on the shared legacy byte-level BPE path, so it is a per-call-site decision. That is #451.

---

## 4. Laya's decision head (from `laya` 0.3.20, the authoritative spec)

The checkpoint alone does not define the head — the reference code does. **There are two
published copies of that code and they are not equivalent**; see §4.1. `laya/common.py`
from PyPI `laya` 0.3.20 is the maintained one and is what this section describes. Ported
semantics:

```python
QTYPES = {"choice": 0, "score": 1, "noul": 2}

def build_sequence(tok, state, q, max_len, head_max_len, ...):
    """[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]"""
    opts = render_options(q)                       # label-index order
    ins  = str(q["ins"]).replace("[MASK]", " ")
    head_ids = tok(f"{q['t']} question: {ins}", add_special_tokens=False)
    opt_ids  = [[MASK] + tok(" " + opt, add_special_tokens=False)[:48] for opt in opts]
    if head_max_len - sum(len(o) for o in opt_ids) < 16:   # too many/long options
        per = max(4, (head_max_len - 16) // len(opt_ids))  # shrink every option evenly
        opt_ids = [o[:per] for o in opt_ids]
    head_ids = head_ids[:max(8, head_max_len - sum(len(o) for o in opt_ids))]
    ids = [CLS] + head_ids + [SEP] + concat(opt_ids) + [SEP] + state_ids[:room] + [SEP]
    return ids[:max_len], marker_positions
```

- **Option rendering** (`render_options`): `choice` → `"key: description"` (bare `key` when the
  description is `None` or `""`); `score` → `"level {i}: {criteria[i]}"`; `noul` → two options
  so `p[1]` *is* the noul probability, defaulting to `["false: …", "true: …"]` but with both
  label strings overridable per question via `q["labels"]` (`_resolve_noul_labels`). Passing
  `labels` on a non-noul question raises. Descriptions pass through `render_criterion`.
  **A criterion of `0` or `False` is a real value, not an absent one** — only `None`/`""` mean
  "no description". The stale HF copy gets this wrong (§4.1).
- **State truncation** (`truncate_left`, used for multi-turn episode prefixes): take the *last*
  `room` tokens of the state, and take **none** when `room == 0`. The slice must be written
  `state[max(0, len(state) - room):]`, because Python's `state[-0:]` is `state[0:]` — the whole
  state. Getting this wrong silently injects the entire state and displaces the trailing `[SEP]`
  (§4.1). `build_sequence` also accepts a precomputed `state_ids` so a caller can tokenize a
  shared state once and reuse it across every question.
- **Every option is scored at its own `[MASK]` token.** The answer space is defined at request
  time, so new schemas need no retraining. Markers are gathered from the encoder output after
  the head.
- **Head** (`DecisionModel.forward`). Read the real signature, because it contradicts what the
  prose around it used to imply:
  `forward(input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False)`.
  **It takes `input_ids`, not `h`** — line 1 is
  `h = self.encoder(input_ids=…, attention_mask=…).last_hidden_state`, because
  `DecisionModel.__init__(encoder, …)` *holds* the encoder as `self.encoder` (that is why
  `encoder.*` and `head.*` share one `model.safetensors`). A port is free to keep the seam
  explicit — run `ModernBertEncoder<T>`, then the head — but the two must agree that the
  encoder is fed `input_ids` + an `attention_mask`, and `detach_encoder` (training-only)
  is the flag that decides whether gradients reach it.
  1. `h = h + type_emb(qtype)[:, None, :]` — question-type embedding broadcast over the sequence.
  2. 2 × `nn.TransformerEncoderLayer(d=1024, nhead=16, dim_feedforward=4096, dropout=0.1,
     norm_first=True)` with `src_key_padding_mask`. So **pre-norm**, attention + ReLU FFN,
     `nn.LayerNorm` **with bias** (`eps` default 1e-5), ReLU (not GELU) in the FFN, all
     projections biased. In eval, dropout is identity. Verified detail: `nhead` is *derived*,
     `nhead = max(1, d // 64)` → 16 for d=1024, not hardcoded — so a different `d` changes the
     head shape and the checkpoint would not load. `nn.TransformerEncoder` is built with
     `enable_nested_tensor=False` and **no `norm=`**, so there is no final LayerNorm, and no
     positional encoding either (correctly: the encoder's RoPE has already been applied to `h`).
     `pad = ~attention_mask.bool()` is a *boolean* mask whose `True` means **ignore**, the
     opposite polarity to the encoder's additive float mask — do not reuse one for the other.
  3. `logits = scorer(gather(h, marker_pos)).squeeze(-1)`, `masked_fill(~marker_mask, -1e4)`.
  4. `act_head(cat([h[:, 0], feats]))` where `feats = [top1, top1-top2, normalized_entropy, k/255]`
     computed from the **detached** softmax. `k = marker_mask.sum(-1).clamp(min=2)`, so entropy
     is normalized by `log(k)` and `k/255` saturates at 255 options. **A one-option question has
     exactly one marker**, so `topk(2, …)` has no second element to select and raises; the
     answer is still well defined (softmax over one logit is 1.0 whatever its value), so pad the
     missing slot with `0.0` — that yields `top1 - top2 == 1.0`, the same "fully decided" signal
     any unambiguous top-1-vs-rest gap produces.
- **Calibration** (`rl_agent_config.json`): `temperature` is per-qtype
  (`choice 1.637, score 1.251, noul 1.983`), and `temperature_by_options` refines it per
  (type, option-count) bucket via `temp_bucket(qtype, k)` → `choice:2`, `choice:3-5`,
  `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2`. **The bucket table wins** when present.
  The model card is explicit that the raw checkpoint is over-confident and mean ECE only drops
  0.466 → 0.081 after refitting temperatures on your own data.
  - **But the shipped `choice:11+` bucket is out of range and must be clamped, not applied.**
    Its value is `0.10058280825614929`, and the reference defines `TEMP_MIN = 0.5`,
    `TEMP_MAX = 5.0` with `clamp_temperature(t)`; `laya/agent.py` clamps every bucket on load
    and **warns about the rejected entries** rather than using them. A temperature below 1
    *sharpens* logits, and the reference is explicit about why that is wrong here: at 0.1006
    the logits are multiplied ~10×, "so a 0.24 top probability is published as 0.99, and a
    caller gating on confidence is told a coin flip is a certainty." **This is the opposite of
    calibration and it lands on exactly the high-cardinality questions that already degrade**
    (see the 77-option limit below). Any port must reproduce the clamp, and must decide
    explicitly whether to clamp or to honour the raw value.
- **Budgets** (English): `max_len 512`, `head_max_len 192` → the state gets ~320 tokens.
  `max_prefixes 6` for multi-turn.

### 4.1 Two copies of the reference exist, and they are not equivalent

Verified 2026-09-27 by diffing the two published artifacts. Both are Apache-2.0 and both come
from Convai Innovations, so this is a versioning problem, not a fork.

| Source | Where | State |
|---|---|---|
| `laya/common.py` | PyPI `laya` **0.3.20** (uploaded 2026-09-24) | **maintained — use this** |
| `rl_common.py` | HF repo `convaiinnovations/laya` | **stale — do not port from this** |

`build_sequence` is 51 lines in the wheel vs 31 in the HF file; `render_options` 19 vs 10;
`DecisionModel` 53 vs 44. Most of the delta is reformatting and added type hints, but **three
of the differences are bug fixes to real defects in the HF copy**:

1. **`truncate_left` returns the entire state when no room is left.** HF: `st = st[-room:]`.
   Python's `st[-0:]` is `st[0:]` — the whole list — so at `room == 0` the state is included in
   full and the trailing `[SEP]` is pushed out. The wheel uses
   `state_ids[max(0, len(state_ids) - room):]` and comments on exactly this trap.
2. **Falsy criteria are rendered as "no description".** HF: `[k if not v else "%s: %s" % (k, v)
   …]`, so a criterion of `0` or `False` collapses to a bare `key` and the value is lost. The
   wheel tests `v is None or v == ""` and routes values through `render_criterion`, noting that
   "only None/`""` mean 'no description'; 0 and False are legitimate criterion values". This
   bites `score` questions hardest, whose `crit` is a list of criteria.
3. **A one-option question raises** in `p.topk(2, -1)` (see the head bullet above).

The wheel also adds `TEMP_MIN`/`TEMP_MAX`/`clamp_temperature` and the warning about the shipped
`choice:11+` bucket, and caps option tokenization with `truncation=True, max_length=48` rather
than tokenizing the whole string and slicing `[:48]` (same output, less work).

**Consequences for the port:**

- Port the **wheel**. Porting `rl_common.py` verbatim would import all three defects above, and
  two of them fail *silently* — a wrong prompt that still "runs", which is the specific failure
  mode the byte-exact prompt gate exists to catch.
- **The wheel is not a superset.** `encode_record` and `predict_items` are absent from
  `laya/common.py` (they live elsewhere in the package), so keep `rl_common.py` as the reference
  for those and for anything the wheel reorganized away.
- `DecisionModel`'s `head_checkpointing` branch is training-only and irrelevant to inference —
  its one difference (keyword vs positional `src_key_padding_mask` in a non-reentrant
  `checkpoint` call) does not affect Phase 2.
- **The wheel can be used as a live reference without installing it.** `pip download
  laya==0.3.20 --no-deps` (118 KB wheel), extract, and prepend to `sys.path`: `import laya` and
  `from laya import common` both succeed, exposing the entire Phase 2 surface
  (`build_sequence`, `render_options`, `render_criterion`, `_resolve_noul_labels`,
  `clamp_temperature`, `temp_bucket`, `serialize_state`, `DecisionModel`, `build_model`,
  `confidence_from_probs`, `TEMP_MIN`, `TEMP_MAX`, `QTYPES`). `common.py`'s only third-party
  imports are `numpy`, `torch` and `transformers`, all already present — the prompt builder
  itself needs **no** ML dependency at all. This is strictly stronger than diffing text: the
  fixture generator can call upstream's real `build_sequence` and upstream's real
  `DecisionModel` as the reference, so a shared misreading of the *specification* cannot make a
  broken port pass. A textual diff only catches a misreading of the *code*.

  Verified behaviour worth pinning: `clamp_temperature(0.10058280825614929)` returns `0.5`
  (raised to `TEMP_MIN`), while `1.9063563346862793` and `1.2514300346374512` pass through
  unchanged. And `DecisionModel.__init__` takes the **encoder module**, not a width —
  `DecisionModel(1024, 2)` raises `AttributeError: 'int' object has no attribute 'config'`.

### What the model card says is *not* worth building first

- `action.act_probability` has no usable signal (AUROC 0.30 on 396 labelled decisions, reads
  1.0 almost always) — the card itself says gate on answer confidence instead (AUROC 0.77 on
  the same items). So `act_head` is **optional** for a first port. Already reported upstream at
  [NandhaKishorM/laya#185](https://github.com/NandhaKishorM/laya/issues/185), so this is not a
  Nivara bug to file; see §9.
- High-cardinality choice questions degrade (77 options in a 192-token head budget ≈ 3–4
  tokens per label) — a known limit, not a porting bug.
- The English root checkpoint is near chance on `typed-decisions` zero-shot; the
  `typed-decisions` subfolder is the fine-tuned one.

---

## 5. Reuse map — what Nivara already has

| ModernBERT / Laya need | Nivara | Verdict |
|---|---|---|
| Bias-free LayerNorm, eps 1e-5 | `LayerNorm<T>(n, eps, affine)` | `affine:true` always allocates a Beta too; a zero Beta is exactly `bias=False`, so **inference needs no core change** — but a fine-tuning Beta receives a gradient and is *not* a no-op, so it is #446 (see §7) |
| RoPE, `rotate_half`, full head-dim, per-instance theta | `RotaryEmbedding<T>(headDim, maxPos, theta)` (`src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs`) | ✅ direct reuse — same HF convention, same half-split |
| Gated MLP `act(input) · gate` | house style = two `Linear<T>` + `Activation.Silu` + `ReverseGradOperations.Multiply` (`LlamaDecoderBlock.cs:105-107`) | ✅ **same shape with `GeluExact` instead of `Silu`** — no new op |
| Exact-erf GELU | `ReverseGradOperations.GeluExact` / `GradKernels.GeluExact` | ✅ |
| Bidirectional masked attention, arbitrary additive mask | `ReverseGradOperations.MultiHeadAttention` (`[qLen,kvLen]` mask) and `BatchedMultiHeadAttention` (`[B,qLen,kvLen]`) | ✅ **sliding window = a band mask, no new op** |
| Fused QKV / fused gate-up weight split | `StateDictLoader.LoadLinear` binds one prefix; splitting = contiguous row-block copies at load | ✅ sample-side loader helper |
| Exact-integer token ids (BF16/Half-safe) | `Embedding<T>.Forward(int[])` (the DistilBERT fix, `docs/BFLOAT16.md`) | ✅ |
| F16-on-disk safetensors → F32 | `SafeTensorsLoader.Read<float>` `ConvertF16` | ✅ Laya's checkpoint is F16 |
| Memory-mapped load, no full-file `byte[]` | `SafeTensorsLoader.Read<T>(path)` | ✅ (842.6 MB < the 2 GB limit) |
| Byte-level BPE from `tokenizer.json` | `Gpt2BpeTokenizer` (built for SmolLM/Qwen) | ⚠️ needs a `tokenizer.json`-only entry point (no `vocab.json`/`merges.txt` shipped) + NFC |

**Net: the encoder is close to entirely sample-side.** Both of the two original gaps are now closed
by Phase 1: (a) a `Gpt2BpeTokenizer` entry point that reads `tokenizer.json` alone, and (b) standing
in for `bias=False` with a zero Beta. Gap (b) remains only a *fine-tuning* concern — see #446.

## 6. Gaps / cost

1. **Banded attention is not free.** Expressing the 129-wide window as a dense `[512, 512]`
   additive mask makes sliding layers cost the same as full layers. At 512 tokens the mask is
   1 MB (fine); at `max_position_embeddings = 8192` a dense mask is 67M elements — 268 MB in F32 —
   **must not be materialised**. So: dense mask ≤ `ModernBertMasks.MaxDenseLength` (2048, 16 MB),
   which throws above that rather than silently allocating; banded/sparse kernel beyond.
2. **No MLM head work yet.** `decoder.bias` / `head.dense` / `head.norm` are in the backbone
   checkpoint but are only needed for masked-token tasks, not for Laya. Out of Phase 1.
3. **Per-layer theta + per-layer band** means the encoder cannot use one shared fused
   decoder-block kernel the way `LlamaDecoderBlock` does; a ModernBERT-specific fused forward is
   a follow-up optimisation, not a Phase-1 requirement.
4. **GPU path** (`--gpu`, `BertEncoderGpuRunner`) is a post-LN BERT kernel built on the
   HuggingFace split-`query`/`key`/`value` naming; extending it is Phase 3 and needs four new
   pieces (RoPE kernel, GeGLU, pre-norm restructure, banded attention mask) rather than a port.
   See issue #449 for the itemised plan, and §9 for the phase sequencing.
5. **Laya head needs `nn.MultiheadAttention` semantics with a fused biased in-proj** — expressible
   with the existing `MultiHeadAttention` op + `ReverseGradOperations.AddBias`, no new op.

## 7. Deliberate shortcuts (documented, not accidental)

- **LayerNorm `affine: true` with an unset (zero) Beta** stands in for HF's `bias=False`. Numerically
  identical in both directions, so inference is exact. The cost is an extra parameter that a
  fine-tuning loop would give a gradient to — so this is a real gap the moment anything is trained,
  which is why it is #446 rather than a footnote.
- **Dense `[L, L]` band mask** instead of a sparse/banded kernel, gated on sequence length
  (see §6.1).

## 8. Data layout

Weights live under `samples/data/`, gitignored, like every other model:

| Path | Contents | Source |
|---|---|---|
| `samples/data/modernbert/` | `model.safetensors` (1510.2 MB F32, single file), `config.json`, `tokenizer.json`, `tokenizer_config.json` | `hf download answerdotai/ModernBERT-large` |
| `samples/data/laya/` | `model.safetensors` (842.6 MB F16, single file), `encoder/config.json`, `rl_agent_config.json`, `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`, plus `rl_common.py` and `README.md` (reference only — see §4.1) | `hf download convaiinnovations/laya` |

```
hf download convaiinnovations/laya \
  model.safetensors encoder/config.json rl_agent_config.json \
  tokenizer/tokenizer.json tokenizer/tokenizer_config.json \
  rl_common.py README.md --local-dir samples/data/laya
```

`rl_common.py` is fetched even though §4.1 rules it out as the porting source: it is still the
only published copy of `encode_record` / `predict_items`, and it is what §4's tensor map and
tensor-count claims were first checked against. The normative source is the wheel
(`pip download laya==0.3.20 --no-deps`), which is not vendored — it is re-fetchable on demand
and is only needed when the prompt builder changes.

Both directories are already in `.gitignore` (`.gitignore:366-367`, following the existing
one-line-per-model pattern, e.g. `samples/data/distilbert/` at `.gitignore:360`). Reference
fixtures (`last_hidden_state_py.bin`, Laya logits/probs) go in the same directories, so they are
covered by the same entries.

The ModernBERT-large download was needed only for the *stock-backbone* parity gate, which Phase 1
has now passed. The Laya checkpoint contains its own (fine-tuned) copy of the same encoder, so
Phase 2 does not depend on it.

## 9. Phases

- **Phase 1 — ModernBERT encoder.** `ModernBertConfig` + `ModernBertEncoder<T>` in
  `samples/Nivara.Samples`, a `modernbert` mode in `samples/NivaraInference`
  (default / `benchmark` / `compare`), a `Python/modernbert_compare.py` fixture generator, and a
  `last_hidden_state` parity gate vs HuggingFace. Acceptance: max abs/rel diff at the
  DistilBERT-class level (~1e-5 rel) plus tokenizer-id agreement.
- **Phase 2 — Laya head.** `LayaDecisionHead<T>` (2 pre-norm transformer layers, type embedding,
  marker scorer, temperature calibration), `LayaPromptBuilder` (a C# port of `build_sequence` +
  `render_options`), a `laya` mode with a PyTorch parity gate on the same fixture methodology.
  `act_head` optional; multilingual/typed-decisions subfolders optional.
  - **The `act_head` AUROC-0.30 deferral is resolved: no Nivara issue, and none should be filed.**
    The checkpoint is now downloaded and the claim is corroborated by the model card (README
    line 371): `action.act_probability` reads 1.0 for almost every input and its raw logits run
    *against* correctness at AUROC 0.30 on 396 labelled decisions, while gating on `confidence`
    reaches AUROC 0.77 on the same items. The decisive point is that this is **already tracked
    upstream** at [NandhaKishorM/laya#185](https://github.com/NandhaKishorM/laya/issues/185) and
    is documented by the model card itself. It is an upstream *model* limitation, not a Nivara
    defect, so a Nivara issue would be a duplicate of a bug we cannot fix and do not own. The
    port should still reproduce `act_head` faithfully (it is 4 tensors / 263,938 params and
    `forward` computes it unconditionally), and the cheap partial check is whether
    `act_probability` reads ≈1.0 for arbitrary synthetic inputs — that needs no eval data and no
    396 labels, so it can confirm the port is faithful without reproducing the AUROC itself.
  - **Resolved — reference source: port from the PyPI wheel, not the HF `rl_common.py`.** The
    cross-check §4.1 records was run and it changed the answer, so this is no longer a choice
    between two options: `laya/common.py` from `laya` 0.3.20 is the maintained reference and the
    HF copy has three defects, two of which fail silently. Port `build_sequence` / `render_options`
    / `render_criterion` / `_resolve_noul_labels` / `clamp_temperature` / `temp_bucket` from the
    wheel into `Python/laya_compare.py`. That surface is pure Python plus the tokenizer, so it
    stays self-contained with no new dependency and `laya compare` remains runnable on a clean
    checkout — and `pip download laya==0.3.20 --no-deps` is enough to re-verify the source later
    without installing anything. Byte-exact prompt parity is the gate, because the prompt is the
    API contract and a silently different render still "runs"; the stale HF copy is precisely the
    case that would pass a loose check while producing the wrong prompt.
  - Resolved earlier, recorded so they are not re-asked: `samples/data/modernbert/` and
    `samples/data/laya/encoder/` are **separate** directories, so Phase 1 stands alone; the Laya
    checkpoint is 842.6 MB on top of the 1510 MB ModernBERT download and was confirmed wanted.
- **Phase 3 — GPU path.** An ILGPU `ModernBertGpuRunner`: a RoPE elementwise kernel (absent
  entirely today), GeGLU, a pre-norm restructure, and a band parameter in the fused attention
  kernel. Gated GPU-vs-CPU rather than GPU-vs-PyTorch, since Phase 1 already pins the CPU path to
  HuggingFace.
