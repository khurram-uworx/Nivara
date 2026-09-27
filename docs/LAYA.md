# Laya (convaiinnovations/laya) — decision head on ModernBERT-large

Status: **Implemented and gated** (2026-09-27, issue #460, branch `khurram/laya`). Phase 2 is complete. The GPU head — what remained of Phase 3 — is unblocked and not wired; that follow-up is #462.

This is a reflection of what we built and what we learned while porting Laya's typed decision head onto the ModernBERT encoder that #449 already runs. It is *not* a usage guide (that lives in [`samples/NivaraInference/README.md`](../samples/NivaraInference/README.md)) and *not* a roadmap.

Related: the encoder's GPU reflection in [docs/GPU-ACCL.md](GPU-ACCL.md) §1b, the backend probe in the inference sample's README.

## Model overview

A ModernBERT-large encoder — the same architecture as [docs/MODERNBERT.md](MODERNBERT.md), nested under the `encoder.` prefix — plus a typed decision head. The checkpoint is **842.6 MB of F16** (206 tensors, 421,293,830 parameters), widened to F32 at load, so the resident set is about 1.7 GB.

- **Encoder**: ModernBERT-large (28 layers, d=1024, fused input|gate `Wi`=5248, ffn=2624), loaded under the `encoder.` prefix via `ModernBertEncoder<T>.LoadWeights`
- **Head**: 2 pre-norm layers (`nn.TransformerEncoderLayer` defaults: biased LayerNorm eps 1e-5, biased fused QKV, **ReLU** feed-forward at `4 * d`, no final norm, no positional encoding). The scorer and the act head use exact-erf GELU. The two defaults disagree, and the shapes do not say which is which.
- **Attention**: the reused `BertSelfAttention<T>`. No second attention module. The checkpoint's `in_proj_weight` / `in_proj_bias` (underscore, not dot) are sliced at load by `StateDictLoader.LoadFusedLinearSlice`.
- **Type embedding** added to every position, then the head, then a gather at the `[MASK]` marker of each option. `nhead = max(1, d / 64)` — 16 heads of 64 at the shipped width.
- **Temperature is not in the head.** The checkpoint's `temperature [3]` buffer is loaded and not applied. `LayaCalibration` looks up the bucket table, falls back to the per-type value, and clamps to `[0.5, 5.0]`. `choice:11+` ships at 0.1006 and clamps to 0.5; the sample prints the raw value.
- **`act_head` is reproduced and is not a signal.** It reads ~1.0 on the shipped checkpoint regardless of input (upstream NandhaKishorM/laya#185). The sample prints that caveat next to the number. Gate on `answer_confidence`, not on it, and not on the uncalibrated entropy `confidence`.
- **The reported choice is the option key**, not the rendered `"key: description"`.
- **Verification**: `laya compare` against the wheel, not a transcription. Prompt ids and markers byte-exact on the four fixture questions; marker logits max |diff| 1.3e-5 (`choice2`) and 2e-6 on the other three; typed decision and temperature bucket match, including the clamp. Bound `|diff| <= 1e-3·(1 + |ref|)`. Prompt parity fails the gate on its own, before any forward.
- **Tokenizer**: `tokenizer/tokenizer.json`, not the model root. Same loader as ModernBERT (`LoadFromTokenizerJson`, NFC on). Laya writes BPE merges as JSON arrays; the reader accepts both that form and the string form ModernBERT and SmolLM use.

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

The encoder config is **not** at the model root. Laya nests a copy of ModernBERT-large under `encoder/`, and `ModernBertConfig.FromJson` already reads the newer `layer_types` + `rope_parameters` layout. Stock ModernBERT uses the `model.` prefix; Laya uses `encoder.`. One loader, the prefix selects.

206 tensors: `encoder.*` 170, `head.*` 24, `scorer.*` 6, `act_head.*` 4, `type_emb.weight` `[3, 1024]`, `temperature` `[3]`. No biases under `encoder.*`. The head's norms and projections are biased. That split is the whole of why the two halves load differently.

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

There is no `head.norm` and no positional encoding on the head stack. The last thing to touch the hidden state is the second layer's FFN residual. `pooled` is that state at position 0, the `[CLS]` row **after** the head, not the encoder's.

### Calibration, as shipped

Per-type fallback: choice 1.6369, score 1.2514, noul 1.9834. Bucket table: `choice:2` 1.9064, `choice:3-5` 1.7602, `choice:6-10` 1.0000, `choice:11+` **0.1006 → clamped to 0.5**, `score:3-5` 1.2514, `noul:2` 1.9834. Range is `[0.5, 5.0]`. A non-number falls back to 1.0. The features the act head reads are the **untempered** softmax. Temperature is applied only in the decode step, to the first `k` logits.

`answer_confidence` is `max(p)`, rounded to 4 decimal places, half-to-even (Python's `round`). `confidence` is normalized entropy for choice and score, and `max(p, 1−p)` for noul. Only the first is calibrated. The sample prints both, and says which is which.

## How this model improved the library

All new types are sample-side (`src/Nivara` is untouched):

| Piece | Location | Notes |
|---|---|---|
| `LayaPromptBuilder`, `LayaQuestion`, `LayaSequence` | `samples/Nivara.Samples/LayaPromptBuilder.cs` | port of `build_sequence` / `render_options` / `render_criterion` |
| `LayaCalibration`, `LayaDecision`, `LayaTemperature` | `samples/Nivara.Samples/LayaCalibration.cs` | bucket, clamp, decode. Temperature is applied here, not in the head |
| `LayaHeadLayer<T>`, `LayaDecisionHead<T>` | `samples/Nivara.Samples/LayaHeadModel.cs` | the head. Attention is the reused `BertSelfAttention<T>`, not a new module |
| `StateDictLoader.LoadFusedLinearSlice` | `samples/Nivara.Samples/StateDictLoader.cs` | additive. Reads `in_proj_weight` / `in_proj_bias` by key, not by prefix |
| `Laya` mode | `samples/NivaraInference/Laya.cs` | default, `benchmark`, `compare`. `--gpu` rejected |
| `Python/laya_compare.py` | `samples/NivaraInference/Python/` | wheel download + `importlib` load of `common.py`. Nothing is installed |

`Gpt2BpeTokenizer.ReadInlineMerges` was fixed in the same branch, before the head existed. Laya writes `model.merges` as arrays (`["Ġ", "Ġ"]`); ModernBERT and SmolLM write strings (`"Ġ Ġ"`). The reader skipped every non-string entry, so all 50,009 Laya merges were discarded and the tokenizer silently degraded to character splitting (`noul` came out as four ids instead of two). Both forms are accepted now. The symptom is a prompt that still "runs".

## Library features used

| Capability | Where exercised |
|---|---|
| `ModernBertEncoder<T>.LoadWeights` prefix `encoder` | Laya's nested copy of ModernBERT-large, same loader as the stock backbone |
| `LayaPromptBuilder.BuildSequence` | `[CLS] <type> question: <ins> [SEP] [MASK] opt … [SEP] state [SEP]`; string states only |
| `LayaDecisionHead<T>` / `LayaHeadLayer<T>` | type embedding, 2 pre-norm ReLU layers, marker gather, scorer, act head |
| `BertSelfAttention<T>.ForwardWithMask` | reused; no second attention module. Key-padding mask is one entry per position |
| `StateDictLoader.LoadFusedLinearSlice` | `in_proj_weight` / `in_proj_bias` sliced into q, k, v |
| `LayaCalibration` | bucket table, per-type fallback, clamp to `[0.5, 5.0]`; `choice:11+` 0.1006 → 0.5 |
| `laya compare` | wheel-loaded reference; prompt ids byte-exact, then logits, then the typed decision |

## Verification

`laya compare` against the **`laya==0.3.20` wheel itself**, not a transcription of it. The generator downloads the wheel, loads `laya/common.py` by path, and runs that file's `DecisionModel`. Four fixture questions, one per question type plus the high-cardinality choice case, on the 822 MB F16 checkpoint widened to F32 at load. Bound `|csharp − wheel| ≤ 1e-3·(1 + |wheel|)`, the same bar the ModernBERT gate uses:

| check | result |
|---|---|
| Prompt ids and marker positions | **byte-exact** on all four (40 / 119 / 48 / 39 tokens) |
| Marker-scorer logits | **max \|diff\| 1.3e-5** (`choice2`), **2e-6** on the other three |
| Typed decision | match, including the option **key** (not the rendered description) |
| Temperature bucket | match, including `choice:11+` **0.1006 → 0.5** |
| `act_head` class-0 probability | 1.0 on both sides |

Prompt parity runs **first** and returns before any forward if an id or a marker disagrees. A different prompt still produces logits, and a numeric diff would blame the wrong thing.

## Performance

`laya benchmark`, AC power, Release, F32, .NET 11. Padded to `max_len` 512 so the four questions are the same forward; the valid-token counts are how long the prompt actually is, not how long the forward was. Protocol is the mode's own: one untimed encoder pass, then three timed encoder-plus-head passes, median reported. The head's first call falls inside the timed loop, which is the likely source of the high sample on `noul`.

| question | valid | markers | median | min | max |
|---|---|---|---|---|---|
| choice2 | 40 | 2 | **4187 ms** | 4001 | 4715 |
| choice13 | 119 | 13 | **4207 ms** | 3770 | 4772 |
| score4 | 48 | 4 | **3910 ms** | 3870 | 4492 |
| noul | 39 | 2 | **4243 ms** | 3770 | 5697 |

Weight bind was 6.6 s on top of a 1.1 s safetensors parse, so a cold run is load plus about four seconds of forward. This is a reference timing, not a deployment claim. The 2026-09-27 probe projected the CPU GEMM alone at **2414 ms**. The measured forward is about **4.1 s**. The gap is real and it is not a contradiction: GEMM is ~99.9% of the *arithmetic* and a much smaller share of the *time*, because norms, the dense `[512, 512]` mask, attention, and per-op dispatch are latency. #440 still targets the arithmetic. It will not turn 4.1 s into 0.9 s by itself.

The `act_head` row is a weak signal and the gate says so. The shipped head saturates near 1.0 regardless of input (upstream [NandhaKishorM/laya#185](https://github.com/NandhaKishorM/laya/issues/185), documented on the model card: AUROC 0.30 on 396 labelled decisions, against 0.77 for answer confidence). It is reproduced because `forward` computes it unconditionally. It is not a Nivara defect, and no Nivara issue should be filed for it.

## GPU path

Laya is a ModernBERT-large encoder (28 layers, d=1024, fused input|gate `Wi`=5248, ffn=2624) plus a 2-layer decision head, run at S=512. That is **175.7 G MACs** for the encoder, ~32× DistilBERT, so it is the first model in this repo where the backend choice is not a foregone conclusion. Three measurements settled it. All are reproducible:

```
dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gpu-alloc
dotnet run --project tests/Nivara.PerformanceTests -c Release -- --cpu-gemm
dotnet run --project tests/Nivara.PerformanceTests -c Release -- --gemm
```

**1. Memory: the binary gate passes.** Laya's checkpoint is 421,293,830 F16 parameters and `--gpu` is F32-only, so F32 needs 1.685 GB (1.569 GiB) of device buffers. Every GPU row in this document is 66-110M parameters (0.26-0.44 GB), so this was a 4-6× jump. The probe allocates in 64 MiB chunks (an embedding table plus ~200 per-layer tensors is the realistic shape), fills every chunk from the host and reads a sample back:

| target | buffers | alloc ms | fill GB/s | result |
|---|---|---|---|---|
| 1.569 GiB (Laya F32) | 26 | 109.1 | 8.31 | ok |
| 3.000 GB | 48 | 788.5 | 7.58 | ok |

The full working set allocates and verifies on a 7.559 GiB Arc iGPU. F16 device buffers are therefore a later optimisation, not a prerequisite. Note allocation cost is superlinear (109 ms at 1.569 GiB, 789 ms at 3 GB) — driver paging, not Nivara.

Treat the milliseconds as indicative only. Four consecutive runs gave Laya's row 109.1 / 147.7 / 133.1 / 100.6 ms at 8.31 / 7.32 / 7.18 / 7.77 GB/s, every one of them returning the same verdict and exit 0. The gate is binary; the fill rate is a rough estimate.

**2. Throughput: GPU wins on the dominant term.** Both backends were measured at identical shapes, in the same session, each cell gated against the same host double-precision truth (`maxAbs <= 1e-3`), so this is a like-for-like comparison rather than a cross-harness one.

| shape | GPU GMAC/s (best kernel) | CPU GMAC/s (best leg) |
|---|---|---|
| laya qkv [512x1024x3072] | 202 (Row4Qkv) | 83.2 (Blocked) |
| laya attn out [512x1024x1024] | 202 (Row4) | 111.3 (AutoDiff) |
| laya fc1 (Wi) [512x1024x5248] | 205 (Row4) | 69.7 (AutoDiff) |
| laya fc2 (Wo) [512x2624x1024] | 197 (Row4Bias) | 87.0 (Blocked) |
| laya head ff1 [512x1024x4096] | 204 (Row4Bias) | 78.5 (Blocked) |
| laya head ff2 [512x4096x1024] | 155 (Row4Relu) | 70.5 (Blocked) |
| laya act 1 [512x1028x256] | 188 (Row4Bias) | 100.0 (Blocked) |
| laya scorer 1 [8x1024x1024] | 59 (Row4Bias) | 30.3 (Blocked) |
| laya qkv@128 [128x1024x3072] | 193 (Row4Qkv) | 63.5 (Blocked) |

Rolled up into a projected Laya forward, GEMM only (28 layers of qkv + attn out + Wi + Wo, plus the 2-layer head and the act/scorer tail:

| leg | total ms | GMAC/s |
|---|---|---|
| GPU, Row4 as committed | **~920** | ~200 |
| CPU, best in-tree leg (`MatMulTransposedB`) | 2414 | 76 |
| CPU, per-call B transpose (`MatMul`) | 2791 | 66 |
| CPU, register-blocked reference (probe-local) | 2529 | 73 |

**GPU is 2.6× faster on GEMM**, and the GPU figure is the conservative one: this harness reads the iGPU at roughly half the idle-machine rate (see the `--gemm` note in `tests/Nivara.PerformanceTests/README.md`), so the real margin is likely wider. The GPU side also has #440's tile-32/2×2 still unclaimed, which is a 2-3× *speedup* and can only widen the gap.

Two findings worth keeping, both negative:

- **A register-blocked CPU GEMM does not help.** The probe wrote one (`Parallel.For` over disjoint output rows, four output columns in `Vector<float>` registers across the whole K loop, A and B read in place with no `RentCopy`) on the hypothesis that `MultiplyRowFloat`'s one-`Dot`-per-output-element structure was the bottleneck. It came in at 2529 ms against the in-tree kernel's 2414 ms. BCL's `TensorPrimitives.Dot` is already well tuned, so **a blocked rewrite of the GEMM is not the lever** — which is the specific thing #456-era thinking pointed at.
- **Therefore the CPU's known ~6.1-6.5× deficit is not a GEMM problem** — at Laya's shapes. The CPU GEMM does 66-80 GMAC/s against a 26 GMAC/s end-to-end reading, so ~3× of the deficit is in non-GEMM work. That is the open lead, tracked as **#458**.

**3. Attention was deliberately left out of the measurement.** At S=512 it is 8.4 M MACs per layer against 6.3 G for the four GEMMs, about **0.13%** of the work, so it cannot move the decision. Its one real hazard — `AttentionKernels` computes `Exp(s - max)` with `s = max = -inf` on a fully-masked row and has no finite-check, unlike the CPU `GradKernels.SoftmaxSingle` — is filed on **#448** on the static reading instead.

**Decision: GPU**, and it is the safe direction of the two. The comparison is unusually favourable because the CPU side was given its best available showing — a purpose-built register-blocked kernel — and still lost, while the GPU side was measured exactly as committed with a known 2-3× improvement (#440) still in hand. The CPU `laya` mode is a reference for the parity gate, not a deployment path.

Wiring the head onto `ModernBertGpuRunner` needs no new kernel: pre-norm, *biased* LayerNorm, ReLU, fused *biased* QKV, all present. It is not wired. `--gpu` is rejected, with the supported modes named. Gate it GPU-vs-CPU against this CPU head, the same way `modernbert --gpu compare` gates the encoder.

## What we learned

1. **A plausible reading of the reference is not a verified one.** The first pass over this document produced nine candidate corrections against the wheel. Six were real. Three were wrong, and two of those would have shipped a false claim into this document while looking rigorous. Both were falsified by a two-line probe:
   - `truncation=True, max_length=48` is **byte-identical** to `[:48]`. The wheel's own comment says so. The earlier claim that a token straddling the boundary would differ was simply wrong. The difference is that the wheel avoids tokenizing a long tail, which is throughput, not ids.
   - HuggingFace does **not** truncate at `model_max_length` 8192. A 20,001-token state comes back whole, with a warning. Capping the state at 8192 would have been the divergence it was meant to prevent. The state is bounded only by `room`, i.e. by `max_len`.
2. **The six that stood, and that the port now does.** State truncation defaults to the **first** `room` tokens, not the last. `truncate_left` is the exception, and the wheel sets it only for list states, which this port rejects. `room = max(0, max_len − len(ids) − 1)` — the `− 1` pays for the closing `[SEP]`, and `state[-room:]` is wrong when `room` is 0 because Python's `−0` is `0`. Markers at or past `max_len` are dropped, so `len(markers) != len(options)` is reachable and the head rejects an empty marker list rather than scoring zero options. Temperature is applied in decode, not in the head. A `noul` question with no criterion renders the fallback strings `"no, the statement does not hold"` / `"yes, the statement holds"`, and the labels are stripped and must be two distinct non-empty strings. Only `null` and `""` mean "no description"; `"0"` and `"false"` are real criteria. The stale HF copy got that one wrong.
3. **The fused-QKV name is not a prefix.** ModernBERT writes `attn.Wqkv.weight`, which a prefix-plus-`.weight` helper covers, and it has no bias. `nn.MultiheadAttention` writes `self_attn.in_proj_weight` and `in_proj_bias` — an underscore before the role, not a dot, and both exist. A prefix helper looks up `in_proj.weight`, finds nothing, and throws a message naming a key the checkpoint does not contain. `LoadFusedLinearSlice` takes the two keys.
4. **The key-padding mask is one entry per position, not per valid token.** A shorter mask is not a weaker mask. The consumer iterates positions, so a short mask leaves the tail unconstrained and the padding is attended to normally. The test that caught it pads the same real tokens and requires the marker logits to be unchanged.
5. **A constant input cannot test the FFN activation.** LayerNorm subtracts the mean, so a constant row is zero and both ReLU and GELU produce nothing. The probe that pins ReLU uses a non-constant row and a non-uniform weight, and element 0 is cross-checked against `torch.nn.LayerNorm` plus `relu` (196.504028). GELU on the same weights gives 171.740021.
6. **Compare ids, never token strings.** Laya's `tokenizer.json` is structurally the same shape as ModernBERT's and a different file. `AutoTokenizer.vocab_size` reports 50280; the real vocab is 50368. `[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]` 50284, `[UNK]` 50280. Load `tokenizer/`, not the model root.
7. **The reported answer and the prompt text are different strings.** Getting this backwards still passes every bare-key test. The test that pins it uses a described option.

## What's next

1. **#440 — tile-32 / 2×2 GEMM.** The leading item. A 2–3× kernel win on the term that is ~99.9% of the work.
2. **#462 — wire the head onto `ModernBertGpuRunner`.** Unblocked by this work, not started. Zero new kernels. Gate it GPU-vs-CPU against this head. Lower value than #440.
3. **#448 — mask-as-select.** The fully-masked-row `NaN` hazard. The GPU encoder's `max == -inf → zeros` clamp is the prerequisite #449 landed, not the structural fix. The head's CPU path uses the same additive `-inf` mask as the encoder, so it inherits the same hazard at sequence lengths where a query can see no key.
4. **#447 — banded attention on the CPU.** The dense `[L, L]` mask is capped at `ModernBertMasks.MaxDenseLength` (2048) and throws past it. The GPU encoder already carries the band inside the kernel, which is why `modernbert --gpu benchmark` runs at 4096. The head builds the same dense mask and has the same cap.
5. **Structured states, multilingual, `typed-decisions`.** Out of scope, on purpose. A structured state throws `NotSupportedException` naming the decision. The English root checkpoint is near chance on `typed-decisions` zero-shot; that subfolder is a different model.

## References

- Wheel: `pip download laya==0.3.20 --no-deps`. `laya/common.py` is the maintained reference. The HF repo's `rl_common.py` is stale.
- Checkpoint: `hf download convaiinnovations/laya` into `samples/data/laya/` (gitignored). Encoder config in `encoder/`, tokenizer in `tokenizer/`.
- Gate: `python samples/NivaraInference/Python/laya_compare.py` then `dotnet run --project samples/NivaraInference -c Release -- laya compare`.
- Upstream limitation, not ours: [NandhaKishorM/laya#185](https://github.com/NandhaKishorM/laya/issues/185).
- Encoder GPU path and the GEMM argument: [docs/GPU-ACCL.md](GPU-ACCL.md) §1b, issues #449, #440, #448, #447.
