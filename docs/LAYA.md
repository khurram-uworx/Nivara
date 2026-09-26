# Laya (and ModernBERT) in Nivara — research reference

Durable reference for running [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
and its ModernBERT backbone in Nivara. Everything here was verified against primary sources
(the HF checkpoint's own safetensors header, the `answerdotai/ModernBERT` reference implementation,
and the installed `transformers` 5.14.1 source) rather than assumed.

The executable plan for the branch lives in `docs/TODO.md`. This file is the "why" that must not
be re-derived: architecture semantics, tensor maps, prompt format, reuse map, and the traps.

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
model.safetensors            # single file, F16 on disk, ~808 MB, 206 tensors (encoder + head)
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

29 + 1 + 28×5 = **170 encoder tensors**. 28 layers, hidden 1024, 16 heads (headDim 64),
intermediate 2624.

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
  overlay is `abs(q_idx - kv_idx) <= sliding_window` (`masking_utils.py:141-151`). So the band is
  **129 wide (64 each side + self)**. The `answerdotai` reference agrees via flash-attn
  `window_size = (config.sliding_window // 2, config.sliding_window // 2)` = `(64, 64)`,
  inclusive (`src/bert_layers/attention.py:308-315`; the `+1` in HF's
  `self.sliding_window = config.sliding_window + 1` is flash-attn's inclusive-boundary
  adjustment, not a different band).
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
  loader hands the **upper** row block to the activated projection.
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
  thing to check when a *bidirectional* encoder produces `NaN`. Verified in Phase 1: with the
  clamp, HF and Nivara agree on those rows too (both zero); without it, the `NaN` compounds
  across layers because the mask is an *add* and `NaN + (-inf) = NaN`.
- **QK-norm: ModernBERT-large has none.** The stock checkpoint holds exactly 6 tensors per layer
  (`attn.Wqkv.weight`, `attn.Wo.weight`, `attn_norm.weight`, `mlp.Wi.weight`, `mlp.Wo.weight`,
  `mlp_norm.weight`) and 5 at layer 0, which has no `attn_norm`. QK-norm is a base-vs-large
  distinction, and it is easy to remember backwards.

### 3.5 Reading HF's per-stage output (a trap worth writing down)

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

### 3.4 Tokenizer

`tokenizer/tokenizer.json`: BPE, `pre_tokenizer = ByteLevel{use_regex: true}`,
`normalizer = NFC`, `decoder = ByteLevel`, inline `model.vocab` (50280 entries) + `model.merges`,
`added_tokens` ids `0, 1, 50254…50263`. `tokenizer_config.json` declares
`model_input_names: ["input_ids", "attention_mask"]` — **no `token_type_ids`**.
Specials: `[CLS]`=50281, `[SEP]`=50282, `[PAD]`=50283, `[MASK]`=50284, `[UNK]`.
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
(`tokenizer.json`): the `|||IP_ADDRESS|||` / `|||EMAIL_ADDRESS|||` / `|||PHONE_ADDRESS|||` family at
ids 0 and 50277–50285, `[unusedN]` from 50286, 23 whitespace-run tokens (ids 50254–50276, runs of
24 down to 2 spaces), and the BERT specials at 50281–50284.

One real remaining divergence: NFC is applied only through `LoadFromTokenizerJson`, not on the
shared legacy byte-level BPE path. It also only applies to added tokens declared
`"normalized": true`; the few declared `false` (the `special: true` entries) are matched on the
un-normalized text, which is moot for ASCII specials like `[CLS]` / `[SEP]`.

---

## 4. Laya's decision head (from `rl_common.py`, the authoritative spec)

The checkpoint alone does not define the head — `rl_common.py` does. Ported semantics:

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
  description is empty); `score` → `"level {i}: {criteria[i]}"`; `noul` → always exactly
  `["false: …", "true: …"]` so `p[1]` *is* the noul probability.
- **Every option is scored at its own `[MASK]` token.** The answer space is defined at request
  time, so new schemas need no retraining. Markers are gathered from the encoder output after
  the head.
- **Head** (`DecisionModel.forward`), applied to the *encoder* output:
  1. `h = h + type_emb(qtype)[:, None, :]` — question-type embedding broadcast over the sequence.
  2. 2 × `nn.TransformerEncoderLayer(d=1024, nhead=16, dim_feedforward=4096, dropout=0.1,
     norm_first=True)` with `src_key_padding_mask`. So **pre-norm**, attention + ReLU FFN,
     `nn.LayerNorm` **with bias** (`eps` default 1e-5), ReLU (not GELU) in the FFN, all
     projections biased. In eval, dropout is identity.
  3. `logits = scorer(gather(h, marker_pos)).squeeze(-1)`, `masked_fill(~marker_mask, -1e4)`.
  4. `act_head(cat([h[:, 0], feats]))` where `feats = [top1, top1-top2, normalized_entropy, k/255]`
     computed from the **detached** softmax.
- **Calibration** (`rl_agent_config.json`): `temperature` is per-qtype
  (`choice 1.637, score 1.251, noul 1.983`), and `temperature_by_options` refines it per
  (type, option-count) bucket via `temp_bucket(qtype, k)` → `choice:2`, `choice:3-5`,
  `choice:6-10`, `choice:11+`, `score:3-5`, `noul:2`. **The bucket table wins** when present.
  The model card is explicit that the raw checkpoint is over-confident and mean ECE only drops
  0.466 → 0.081 after refitting temperatures on your own data.
- **Budgets** (English): `max_len 512`, `head_max_len 192` → the state gets ~320 tokens.
  `max_prefixes 6` for multi-turn.

### What the model card says is *not* worth building first

- `action.act_probability` has no usable signal (AUROC 0.30, reads 1.0 almost always) —
  the card itself says gate on answer confidence instead (AUROC 0.77). So `act_head` is
  **optional** for a first port.
- High-cardinality choice questions degrade (77 options in a 192-token head budget ≈ 3–4
  tokens per label) — a known limit, not a porting bug.
- The English root checkpoint is near chance on `typed-decisions` zero-shot; the
  `typed-decisions` subfolder is the fine-tuned one.

---

## 5. Reuse map — what Nivara already has

| ModernBERT / Laya need | Nivara | Verdict |
|---|---|---|
| Bias-free LayerNorm, eps 1e-5 | `LayerNorm<T>(n, eps, affine)` | `affine:true` always allocates a Beta too; a zero Beta is exactly `bias=False`, so **no core change needed** (see §7) |
| RoPE, `rotate_half`, full head-dim, per-instance theta | `RotaryEmbedding<T>(headDim, maxPos, theta)` (`src/Nivara/AutoDiff/Nn/RotaryEmbedding.cs`) | ✅ direct reuse — same HF convention, same half-split |
| Gated MLP `act(input) · gate` | house style = two `Linear<T>` + `Activation.Silu` + `ReverseGradOperations.Multiply` (`LlamaDecoderBlock.cs:105-107`) | ✅ **same shape with `GeluExact` instead of `Silu`** — no new op |
| Exact-erf GELU | `ReverseGradOperations.GeluExact` / `GradKernels.GeluExact` | ✅ |
| Bidirectional masked attention, arbitrary additive mask | `ReverseGradOperations.MultiHeadAttention` (`[qLen,kvLen]` mask) and `BatchedMultiHeadAttention` (`[B,qLen,kvLen]`) | ✅ **sliding window = a band mask, no new op** |
| Fused QKV / fused gate-up weight split | `StateDictLoader.LoadLinear` binds one prefix; splitting = contiguous row-block copies at load | ✅ sample-side loader helper |
| Exact-integer token ids (BF16/Half-safe) | `Embedding<T>.Forward(int[])` (the DistilBERT fix, `docs/BFLOAT16.md`) | ✅ |
| F16-on-disk safetensors → F32 | `SafeTensorsLoader.Read<float>` `ConvertF16` | ✅ Laya's checkpoint is F16 |
| Memory-mapped load, no full-file `byte[]` | `SafeTensorsLoader.Read<T>(path)` | ✅ (808 MB < the 2 GB limit) |
| Byte-level BPE from `tokenizer.json` | `Gpt2BpeTokenizer` (built for SmolLM/Qwen) | ⚠️ needs a `tokenizer.json`-only entry point (no `vocab.json`/`merges.txt` shipped) + NFC |

**Net: the encoder is close to entirely sample-side.** The two real gaps are (a) a
`Gpt2BpeTokenizer` entry point that reads `tokenizer.json` alone, and (b) a bias-free LayerNorm
mode — and (b) is only cosmetic (a zero Beta is numerically identical).

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
   See `docs/TODO.md` for the itemised plan.
5. **Laya head needs `nn.MultiheadAttention` semantics with a fused biased in-proj** — expressible
   with the existing `MultiHeadAttention` op + `ReverseGradOperations.AddBias`, no new op.

## 7. Deliberate shortcuts (documented, not accidental)

- **LayerNorm `affine: true` with an unset (zero) Beta** stands in for HF's `bias=False`. Numerically
  identical. The only cost is an unused parameter and a `+0`. If we ever fine-tune ModernBERT,
  this should become a real `LayerNorm(…, bias: false)` flag — tracked in the TODO's issues log.
- **Dense `[L, L]` band mask** instead of a sparse/banded kernel, gated on sequence length
  (see §6.1).

## 8. Data layout

Weights live under `samples/data/`, gitignored, like every other model:

| Path | Contents | Source |
|---|---|---|
| `samples/data/modernbert/` | `model.safetensors` (1.6 GB F32, single file), `config.json`, `tokenizer.json`, `tokenizer_config.json` | `hf download answerdotai/ModernBERT-large` |
| `samples/data/laya/` | `model.safetensors` (808 MB F16, single file), `encoder/config.json`, `rl_agent_config.json`, `tokenizer/tokenizer.json` | `hf download convaiinnovations/laya` |

`samples/data/modernbert/` and `samples/data/laya/` need `.gitignore` entries
(the existing pattern is one line per model dir, e.g. `samples/data/distilbert/` at
`.gitignore:360`). Reference fixtures (`last_hidden_state_py.bin`, Laya logits/probs) go in the
same directories, which are already ignored.

The ModernBERT-large download is only needed for the *stock-backbone* parity gate; the Laya
checkpoint contains its own (fine-tuned) copy of the same encoder, so Phase 2 does not depend on
it.

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
- **Phase 3 — GPU path.** An ILGPU `ModernBertGpuRunner`: a RoPE elementwise kernel (absent
  entirely today), GeGLU, a pre-norm restructure, and a band parameter in the fused attention
  kernel. Gated GPU-vs-CPU rather than GPU-vs-PyTorch, since Phase 1 already pins the CPU path to
  HuggingFace.
