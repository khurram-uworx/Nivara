# Laya decision head — implementation reflection

Status: **Implemented and gated** (2026-09-27, issue #460, branch `khurram/laya`).
Phase 2 is complete. The GPU head — what remained of Phase 3 — is unblocked
and not wired; that follow-up is #462.
This is a reflection of what we built and what we learned while porting Laya's
typed decision head onto the ModernBERT encoder that #449 already runs. It is
*not* a usage guide (that lives in
[`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md))
and *not* a roadmap. The pre-implementation research that used to live here was
rewritten away; the durable facts and the places a first reading of the
reference was wrong are what remain.

Related: the encoder's GPU reflection in [docs/BERT-GPU.md](BERT-GPU.md) §1b,
the backend probe in the inference sample's README (the 2026-09-27 GEMM
numbers below are that probe, not a new session).

The end-to-end wall-clock below was taken on AC power. A battery reading of
this model is not a number worth keeping (see BERT-GPU.md lesson 8). The
correctness gate does not depend on power state; the timing does.

## 1. Result — gated

`laya compare` against the **`laya==0.3.20` wheel itself**, not a transcription
of it. The generator downloads the wheel, loads `laya/common.py` by path, and
runs that file's `DecisionModel`. Four fixture questions, one per question type
plus the high-cardinality choice case, on the 822 MB F16 checkpoint widened to
F32 at load. Bound `|csharp − wheel| ≤ 1e-3·(1 + |wheel|)`, the same bar the
ModernBERT gate uses:

| check | result |
|---|---|
| Prompt ids and marker positions | **byte-exact** on all four (40 / 119 / 48 / 39 tokens) |
| Marker-scorer logits | **max \|diff\| 1.3e-5** (`choice2`), **2e-6** on the other three |
| Typed decision | match, including the option **key** (not the rendered description) |
| Temperature bucket | match, including `choice:11+` **0.1006 → 0.5** |
| `act_head` class-0 probability | 1.0 on both sides |

`laya benchmark`, AC power, Release, F32, .NET 11. Padded to
`max_len` 512 so the four questions are the same forward; the valid-token
counts are how long the prompt actually is, not how long the forward was.
Protocol is the mode's own: one untimed encoder pass, then three timed
encoder-plus-head passes, median reported. The head's first call falls inside
the timed loop, which is the likely source of the high sample on `noul`.

| question | valid | markers | median | min | max |
|---|---|---|---|---|---|
| choice2 | 40 | 2 | **4187 ms** | 4001 | 4715 |
| choice13 | 119 | 13 | **4207 ms** | 3770 | 4772 |
| score4 | 48 | 4 | **3910 ms** | 3870 | 4492 |
| noul | 39 | 2 | **4243 ms** | 3770 | 5697 |

Weight bind was 6.6 s on top of a 1.1 s safetensors parse, so a cold run is
load plus about four seconds of forward. This is a reference timing, not a
deployment claim. The 2026-09-27 probe projected the CPU GEMM alone at
**2414 ms**. The measured forward is about **4.1 s**. The gap is real and it
is not a contradiction: GEMM is ~99.9% of the *arithmetic* and a much smaller
share of the *time*, because norms, the dense `[512, 512]` mask, attention,
and per-op dispatch are latency. #440 still targets the arithmetic. It will
not turn 4.1 s into 0.9 s by itself.

The `act_head` row is a weak signal and the gate says so. The shipped head
saturates near 1.0 regardless of input (upstream
[NandhaKishorM/laya#185](https://github.com/NandhaKishorM/laya/issues/185),
documented on the model card: AUROC 0.30 on 396 labelled decisions, against
0.77 for answer confidence). It is reproduced because `forward` computes it
unconditionally. It is not a Nivara defect, and no Nivara issue should be filed
for it.

Prompt parity runs **first** and returns before any forward if an id or a
marker disagrees. A different prompt still produces logits, and a numeric diff
would blame the wrong thing.

`src/Nivara` is untouched. Every new type is sample-side.

## 2. How it runs — the library map

An engineer extending this should start here. The sample is the scratchpad;
this section maps the code.

### The load path

```
samples/data/laya/
  model.safetensors          206 tensors, F16 except temperature [3] which is F32
  encoder/config.json        ModernBERT-large, layer_types + rope_parameters
  tokenizer/tokenizer.json   not at the model root — loading the root fails
  rl_agent_config.json       max_len, head_max_len, head_layers, temperature tables
    → SafeTensorsLoader.Read<float>          F16 widened to F32 at load
    → ModernBertEncoder<T>.LoadWeights(..., prefix: "encoder")
    → LayaDecisionHead<T>.LoadWeights        head.*, scorer.*, act_head.*, type_emb
    → Gpt2BpeTokenizer.LoadFromTokenizerJson (NFC on)
    → LayaCalibration                        bucket table, then per-type fallback, then clamp
```

The encoder config is **not** at the model root. Laya nests a copy of
ModernBERT-large under `encoder/`, and `ModernBertConfig.FromJson` already
reads the newer `layer_types` + `rope_parameters` layout. Stock ModernBERT
uses the `model.` prefix; Laya uses `encoder.`. One loader, the prefix selects.

206 tensors: `encoder.*` 170, `head.*` 24, `scorer.*` 6, `act_head.*` 4,
`type_emb.weight` `[3, 1024]`, `temperature` `[3]`. No biases under
`encoder.*`. The head's norms and projections are biased. That split is the
whole of why the two halves load differently.

### What was added where

| Piece | Location | Notes |
|---|---|---|
| `LayaPromptBuilder`, `LayaQuestion`, `LayaSequence` | `samples/Nivara.Samples/LayaPromptBuilder.cs` | port of `build_sequence` / `render_options` / `render_criterion` |
| `LayaCalibration`, `LayaDecision`, `LayaTemperature` | `samples/Nivara.Samples/LayaCalibration.cs` | bucket, clamp, decode. Temperature is applied here, not in the head |
| `LayaHeadLayer<T>`, `LayaDecisionHead<T>` | `samples/Nivara.Samples/LayaHeadModel.cs` | the head. Attention is the reused `BertSelfAttention<T>`, not a new module |
| `StateDictLoader.LoadFusedLinearSlice` | `samples/Nivara.Samples/StateDictLoader.cs` | additive. Reads `in_proj_weight` / `in_proj_bias` by key, not by prefix |
| `Laya` mode | `samples/NivaraInference/Laya.cs` | default, `benchmark`, `compare`. `--gpu` rejected |
| `Python/laya_compare.py` | `samples/NivaraInference/Python/` | wheel download + `importlib` load of `common.py`. Nothing is installed |

`Gpt2BpeTokenizer.ReadInlineMerges` was fixed in the same branch, before the
head existed. Laya writes `model.merges` as arrays (`["Ġ", "Ġ"]`); ModernBERT
and SmolLM write strings (`"Ġ Ġ"`). The reader skipped every non-string entry,
so all 50,009 Laya merges were discarded and the tokenizer silently degraded
to character splitting (`noul` came out as four ids instead of two). Both
forms are accepted now. The symptom is a prompt that still "runs".

### The forward, in the wheel's order

```
h = encoder(ids)                              # ModernBertEncoder, prefix "encoder"
h = h + type_emb[qtype]                       # one row, broadcast over positions
h = head.layers[i](h, src_key_padding_mask)   # 2 layers, no final norm
logits = scorer(h[marker_pos]).squeeze(-1)    # [k], untempered
p = softmax(logits)                           # feeds the features, not the answer
feats = [top1, top1 − top2, ent, max(2, k) / 255]
act = act_head(cat([h[0], feats]))            # reproduced, not gated on
answer = softmax(logits / temperature)        # LayaCalibration, after the head
```

`nhead = max(1, d / 64)`. At d=1024 that is 16 heads of 64, which is also
`GpuBuffers.MaxHeadDim`. The head's feed-forward is `4 * d` and its activation
is **ReLU**, because that is `nn.TransformerEncoderLayer`'s default. The scorer
and the act head use exact-erf GELU, which is `nn.GELU()`'s default. The two
defaults disagree, and the checkpoint shapes do not disambiguate.

## 3. What shipped — decisions worth keeping

- **The reference is the wheel, loaded by path.** `pip download laya==0.3.20
  --no-deps`, unzip, `importlib` on `laya/common.py`, bypassing
  `laya/__init__.py`. A transcription would agree with this port for the wrong
  reason. The HF repo's `rl_common.py` is stale and is not the source; it is
  still downloaded with the checkpoint because it is the only published copy of
  a few helpers, and it is labelled reference-only.
- **String states only.** The wheel JSON-dumps dict and list states with
  `ensure_ascii=False` and `separators=(", ", ": ")`. Matching that
  byte-for-byte is its own port. A structured state throws rather than being
  serialized differently. The gate's state is a string.
- **The CPU mode is a reference, not a deployment claim.** The 2026-09-27 probe
  settled the backend: at S=512, GEMM is ~99.9% of the arithmetic (attention
  0.13%), the GPU GEMM projects **~920 ms** against a CPU GEMM of **~2414 ms**
  (2.6×), and #440's tile-32 / 2×2 is still unclaimed on the GPU side. Those
  numbers are that probe, reproduced in the sample README. They were not
  re-measured here. The AC-power wall-clock in §1 is a different quantity:
  about 4.1 s end to end against that 2414 ms GEMM projection.
- **One question at a time.** Each question has its own length. `k` is the
  marker count, so the wheel's `masked_fill(~marker_mask, -1e4)` has no padding
  slot here. It is documented as a no-op, not reimplemented. The
  `clamp(min=2)` on the option-count feature is kept, because a one-marker
  question has no second `topk` entry and the wheel pads it with 0.0.
- **No new attention module.** `BertSelfAttention<T>` already has biased q/k/v,
  a biased output projection, the dense `[L, L]` additive key-padding mask, and
  the `1/sqrt(headDim)` scale. The checkpoint's fused `in_proj_weight` is a
  load-time slice. Mask polarity needs no conversion: `src_key_padding_mask`
  True means ignore, and a value below 0.5 means ignore.
- **Temperature is not applied in the head.** The checkpoint carries a
  `temperature [3]` buffer and the wheel registers it, but `forward` never
  reads it. The caller divides after the logits come back. The buffer is loaded
  and exposed as `CheckpointTemperature` so the weights are accounted for, and
  not applied — reading it would be a second, silently divergent source for
  numbers `rl_agent_config.json` already owns. The bucket table wins when it
  has an entry. `choice:11+` is 0.1006, which is below `TEMP_MIN` 0.5, so it
  clamps, and the raw value stays visible.
- **The reported choice is the option key.** `crit.keys()[argmax]`, not the
  rendered `"key: description"`. They coincide only when the option has no
  description. `choice13`'s answer is `opt1`, not `opt1: topic number 1`.

### Head tensors

| Key | Shape | What it is |
|---|---|---|
| `head.layers.{0,1}.self_attn.in_proj_weight` / `in_proj_bias` | `[3072, 1024]` / `[3072]` | fused Q‖K‖V, **with bias**, underscore not dot |
| `head.layers.{0,1}.self_attn.out_proj.weight` / `.bias` | `[1024, 1024]` / `[1024]` | |
| `head.layers.{0,1}.linear1` / `linear2` | `[4096, 1024]` / `[1024, 4096]` | plain FFN, ReLU, not gated |
| `head.layers.{0,1}.norm1` / `norm2` | `[1024]` weight and bias | biased LayerNorm, eps 1e-5 |
| `scorer.0` / `scorer.1` / `scorer.3` | LayerNorm, Linear, Linear | index 2 is GELU and has no parameters |
| `act_head.0` / `act_head.2` | `[256, 1028]` / `[2, 256]` | index 1 is GELU. Input is `d + 4` |
| `type_emb.weight` | `[3, 1024]` | choice / score / noul |
| `temperature` | `[3]` F32 | buffer. Not read by `forward` |

There is no `head.norm` and no positional encoding on the head stack. The last
thing to touch the hidden state is the second layer's FFN residual. `pooled`
is that state at position 0, the `[CLS]` row **after** the head, not the
encoder's.

### Calibration, as shipped

Per-type fallback: choice 1.6369, score 1.2514, noul 1.9834. Bucket table:
`choice:2` 1.9064, `choice:3-5` 1.7602, `choice:6-10` 1.0000, `choice:11+`
**0.1006 → clamped to 0.5**, `score:3-5` 1.2514, `noul:2` 1.9834. Range is
`[0.5, 5.0]`. A non-number falls back to 1.0. The features the act head reads
are the **untempered** softmax. Temperature is applied only in the decode
step, to the first `k` logits.

`answer_confidence` is `max(p)`, rounded to 4 decimal places, half-to-even
(Python's `round`). `confidence` is normalized entropy for choice and score,
and `max(p, 1−p)` for noul. Only the first is calibrated. The sample prints
both, and says which is which.

## 4. What we learned

1. **A plausible reading of the reference is not a verified one.** The first
   pass over `docs/LAYA.md` §4 produced nine candidate corrections against the
   wheel. Six were real. Three were wrong, and two of those would have shipped
   a false claim into this document while looking rigorous. Both were
   falsified by a two-line probe:
   - `truncation=True, max_length=48` is **byte-identical** to `[:48]`. The
     wheel's own comment says so. The earlier claim that a token straddling
     the boundary would differ was simply wrong. The difference is that the
     wheel avoids tokenizing a long tail, which is throughput, not ids.
   - HuggingFace does **not** truncate at `model_max_length` 8192. A
     20,001-token state comes back whole, with a warning. Capping the state
     at 8192 would have been the divergence it was meant to prevent. The
     state is bounded only by `room`, i.e. by `max_len`.
   - The third false correction claimed §4 used a stale `opt_budget` after the
     option shrink. The line is evaluated after the shrink and already uses
     the recomputed sum.
2. **The six that stood, and that the port now does.** State truncation
   defaults to the **first** `room` tokens, not the last. `truncate_left` is
   the exception, and the wheel sets it only for list states, which this port
   rejects. `room = max(0, max_len − len(ids) − 1)` — the `− 1` pays for the
   closing `[SEP]`, and `state[-room:]` is wrong when `room` is 0 because
   Python's `−0` is `0`. Markers at or past `max_len` are dropped, so
   `len(markers) != len(options)` is reachable and the head rejects an empty
   marker list rather than scoring zero options. Temperature is applied in
   decode, not in the head. A `noul` question with no criterion renders the
   fallback strings `"no, the statement does not hold"` /
   `"yes, the statement holds"`, and the labels are stripped and must be two
   distinct non-empty strings. Only `null` and `""` mean "no description";
   `"0"` and `"false"` are real criteria. The stale HF copy got that one wrong.
3. **The fused-QKV name is not a prefix.** ModernBERT writes
   `attn.Wqkv.weight`, which a prefix-plus-`.weight` helper covers, and it has
   no bias. `nn.MultiheadAttention` writes `self_attn.in_proj_weight` and
   `in_proj_bias` — an underscore before the role, not a dot, and both exist.
   A prefix helper looks up `in_proj.weight`, finds nothing, and throws a
   message naming a key the checkpoint does not contain.
   `LoadFusedLinearSlice` takes the two keys.
4. **The key-padding mask is one entry per position, not per valid token.**
   A shorter mask is not a weaker mask. The consumer iterates positions, so a
   short mask leaves the tail unconstrained and the padding is attended to
   normally. The test that caught it pads the same real tokens and requires
   the marker logits to be unchanged.
5. **A constant input cannot test the FFN activation.** LayerNorm subtracts
   the mean, so a constant row is zero and both ReLU and GELU produce nothing.
   The probe that pins ReLU uses a non-constant row and a non-uniform weight,
   and element 0 is cross-checked against `torch.nn.LayerNorm` plus `relu`
   (196.504028). GELU on the same weights gives 171.740021.
6. **Compare ids, never token strings.** Laya's `tokenizer.json` is
   structurally the same shape as ModernBERT's and a different file.
   `AutoTokenizer.vocab_size` reports 50280; the real vocab is 50368.
   `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]` 50284, `[UNK]` 50280.
   Load `tokenizer/`, not the model root.
7. **The reported answer and the prompt text are different strings.** Getting
   this backwards still passes every bare-key test. The test that pins it uses
   a described option.

## 5. Where reality diverged from the pre-implementation estimate

- **Phase 3 was already done for the encoder.** #449 shipped
  `ModernBertGpuRunner` before this work started. The estimate that Phase 3
  was "a new runner, not a config change" was right, and it had already
  landed. What remained of Phase 3 is the head, and the head needs no new
  kernel: pre-norm, biased LayerNorm, ReLU, fused biased QKV, all present.
  It is not wired onto the GPU runner. `--gpu` is rejected, with the
  supported modes named. Wiring it is a follow-up, and it should be gated
  against this CPU head rather than against PyTorch directly — that was the
  reason to do the CPU head first, and it is now available.
- **The planned `LayaHeadAttention<T>` was duplication.** Dropped before it
  was written. `BertSelfAttention<T>` already had the four properties the new
  type was going to grow.
- **The planned shared JSON fixture was not how the inputs stayed aligned.**
  The questions live in two places, `Laya.cs` and `laya_compare.py`, and the
  prompt-parity gate is what catches a drift. A shared file would also have
  worked. The gate is the stronger check, because it compares the rendered
  ids, not the question text.
- **`act_head` was not optional.** The earlier note said it could be skipped
  because it carries no signal. `forward` computes it unconditionally, it is
  four tensors, and omitting it would make `load_state_dict(strict=True)` on
  the wheel's own model a different architecture. It is reproduced and
  labelled.
- **The highest-value remaining item was never the head.** GEMM is ~99.9% of
  the arithmetic. #440 (tile-32 / 2×2) is worth more than wiring the head onto
  the GPU, and more than any further dispatch reduction. The head is a few
  percent of a forward on the probe's shapes (`head ff1` 204 GMAC/s,
  `head ff2` 155, `act 1` 188, `scorer 1` 59 — the small one is a 8-row GEMM,
  which is why it is slow per MAC and still cheap in absolute time). Those
  rates are the 2026-09-27 probe, not a new measurement.

## 6. What's next

1. **#440 — tile-32 / 2×2 GEMM.** The leading item. A 2–3× kernel win on the
   term that is ~99.9% of the work. The GPU number in the probe is the
   conservative one: that harness reads the iGPU at roughly half the
   idle-machine rate, and #440 is still unclaimed.
2. **#462 — wire the head onto `ModernBertGpuRunner`.** Unblocked by this
   work, not started. Zero new kernels. Gate it GPU-vs-CPU against this head,
   the same way `modernbert --gpu compare` gates the encoder. `--gpu` on
   `laya` says so and exits 1. Lower value than #440.
3. **#448 — mask-as-select.** The fully-masked-row `NaN` hazard. The GPU
   encoder's `max == -inf → zeros` clamp is the prerequisite #449 landed, not
   the structural fix. The head's CPU path uses the same additive `-inf` mask
   as the encoder, so it inherits the same hazard at sequence lengths where a
   query can see no key. The fixture questions are short enough that this does
   not fire; it is not a reason to call the gate complete at long context.
4. **#447 — banded attention on the CPU.** The dense `[L, L]` mask is capped
   at `ModernBertMasks.MaxDenseLength` (2048) and throws past it. The GPU
   encoder already carries the band inside the kernel, which is why
   `modernbert --gpu benchmark` runs at 4096. The head builds the same dense
   mask and has the same cap.
5. **Structured states, multilingual, `typed-decisions`.** Out of scope, on
   purpose. A structured state throws `NotSupportedException` naming the
   decision. The English root checkpoint is near chance on `typed-decisions`
   zero-shot; that subfolder is a different model.

## 7. References

- Wheel: `pip download laya==0.3.20 --no-deps`. `laya/common.py` is the
  maintained reference. The HF repo's `rl_common.py` is stale.
- Checkpoint: `hf download convaiinnovations/laya` into `samples/data/laya/`
  (gitignored). Encoder config in `encoder/`, tokenizer in `tokenizer/`.
- Gate: `python samples/NivaraInference/Python/laya_compare.py` then
  `dotnet run --project samples/NivaraInference -c Release -- laya compare`.
- Upstream limitation, not ours:
  [NandhaKishorM/laya#185](https://github.com/NandhaKishorM/laya/issues/185).
- Encoder GPU path and the GEMM argument: [docs/BERT-GPU.md](BERT-GPU.md) §1b,
  issues #449, #440, #448, #447.
