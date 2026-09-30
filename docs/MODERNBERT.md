# ModernBERT-large (answerdotai/ModernBERT-large)

## Model overview

The **first encoder that is not BERT-shaped**: 28 layers, 1024 hidden, 16 heads (head_dim 64), 2624 intermediate, 8192 max positions, 394.8M parameters (1510 MB F32). Ten layers attend globally and eighteen use a **bidirectional sliding window** — 10 full + 18 sliding for `global_attn_every_n_layers = 3` (layer `i` is global when `i % 3 == 0`).

> The checkpoint file holds **173** tensors but the encoder binds only **170**. The other three — `head.dense.weight` (1,048,576), `head.norm.weight` (1,024) and `decoder.bias` (50,368) — are an MLM head that `ModernBertModel` never instantiates. Summing the whole file gives 395,881,664, which is 1,099,968 more than the encoder; the encoder's own 394,781,696 is the number that matches `sum(p.numel() for p in model.parameters())` exactly. (Likewise `decoder.weight` is tied to the token embedding and counted once.)

## Architecture

- **Pre-norm, not Post-LN**: every sub-layer is `x = x + f(LayerNorm(x))`. Layer 0's `attn_norm` is an `Identity` in HF and has **no checkpoint entry**, so the loader leaves it null.
- **No QK-norm and no layer scale.** The checkpoint has 6 tensors per layer (5 at layer 0) and 170 encoder tensors total — no per-head learnable scale, unlike Gemma-style models. `decoder.weight` is tied to the token embedding.
- **Bias-free LayerNorm and MLP/attention projections** (`norm_bias`, `mlp_bias`, `attention_bias` all false). The norms are constructed with `LayerNorm<T>(..., bias: false)`, so no beta is registered and an optimizer cannot drift one off zero. The projections use `Linear<T>(..., bias: false)`.
- **Fused QKV**: `Wqkv` is one `[3072, 1024]` matrix split into three **contiguous** row blocks (q, k, v) by `StateDictLoader.LoadLinearSlice`.
- **GeGLU with the halves the *other way round* from the names.** `modeling_modernbert.py:89-91` is `input, gate = Wi(x).chunk(2, dim=-1)` then `act(input) * gate`. The **first** row block gets the GELU (`mlp.inputProj`) and the second stays linear (`mlp.gateProj`) — the opposite of what HF's variable naming suggests. Building it the other way is the single hardest bug in this port: it produces a plausible forward pass at **cosine 0.82** at layer 0.
- **Per-layer-type RoPE**: `global_rope_theta = 160000` on full layers, `local_rope_theta = 10000` on sliding ones, applied as HF's `rotate_half` (interleaved-half split, not the GPT-NeoX permute-and-halve variant). Newer checkpoints that ship `rope_parameters` per layer type are also read.
- **Bidirectional band mask fused with right padding** into a dense `[L, L]` additive tensor of 0 / `-inf`: `abs(q - kv) <= local_attention / 2` (the `+1` in `ModernBertAttention.__init__` is flash-attention window semantics and must *not* be applied) **and** `kv < validLength`. `ModernBertMasks.MaxDenseLength` caps this at 2048 and throws beyond it. **Two** such masks are built per forward — one global, one sliding (`ModernBertModel.cs:435-436`), outside the layer loop, and each layer is then handed the one its type calls for — so a dense mask at the model's 8192-token context is 67M elements *each*, 268 MB in F32 apiece or **537 MB for the pair**, not 268 MB total. This cap is the CPU path's only; the GPU path carries the band inside the attention kernel and materialises no `[L, L]` mask at all, which is why `modernbert --gpu benchmark` runs at 4096 while `modernbert benchmark` stops at 2048. Note that "score masking only" would *not* lift the cap, since the cost is the allocation; only a kernel that never builds the mask does. The GPU band is now actually used rather than merely plumbed (**#447**), and the CPU-side equivalent is filed as **#473** — a feasibility fix, not a speed one, since attention is 2.1%/4.1% of CPU MACs at L=128/256.
- **Tokenizer**: the checkpoint ships **only** `tokenizer.json` (inline `model.vocab` + `model.merges`), loaded through `Gpt2BpeTokenizer.LoadFromTokenizerJson` with **opt-in NFC** (default on for that factory only, so the SmolLM and Qwen paths stay byte-identical).
- **Footprint**: 1510.2 MB of F32 weights → ~1.5 GB resident after load, 2.5–3 GB peak managed heap. The safetensors parse is ~2.7 s and binding the weights into modules another ~10.7 s, so a cold `benchmark` run is dominated by load — more than twice a seq-128 forward (2.59 s). This is the largest model in the sample by a wide margin and the only one where load time is a meaningful share of the wall clock.

Nivara modules used: `ModernBertEncoder<T>` / `ModernBertLayer<T>` / `ModernBertAttention<T>` / `ModernBertMlp<T>` (all new, in `Nivara.Samples`), `Embedding<T>`, `LayerNorm<T>`, `Linear<T>`, `ReverseGradOperations.Gelu`, `ReverseGradOperations.Softmax`, `ReverseGradOperations.MatMul`.

## How this model improved the library

Two pre-existing core bugs were found and fixed, both now regression-tested:

1. **`Gpt2BpeTokenizer` ran the GPT-2 pre-tokenizer pattern over the byte-mapped string instead of the raw text.** Byte level maps a space to `Ġ` (U+0120), which `\p{L}` classifies as a letter, so the space joined the following word: `" 2026"` came out as `Ġ` + `2026` instead of HF's `Ġ20` + `26`. The four pre-existing tokenizer tests all used letter-only text, which cannot distinguish the two orders.

2. **`GradKernels` returned `NaN` for a fully-masked softmax row.** ModernBERT's sliding window leaves rows at or past `validLength + window` with no visible key, so the row max is `-inf` and `x - max` is `NaN` — and because the mask is applied as an **add**, `NaN + (-inf) = NaN` meant the poisoned row escaped suppression in the next layer and took every other query row with it. The kernels now clamp a row whose max is exactly `-inf` to zeros, which is the one case PyTorch's `_safe_softmax` (what `scaled_dot_product_attention` uses) clamps too. The test is deliberately `max == -inf` and not `!max.IsFinite`: a `NaN` or `+inf` row max means the model already diverged, PyTorch propagates both to `NaN`, and zeroing them would delete the most useful diagnostic a diverging run produces. The residual gap — a `NaN` already present in q/k/v was still not suppressed, because `NaN + (-inf) = NaN` — was filed as #448 and is now **closed**: `AttentionKernels<T>.ApplyMask` *assigns* `-inf` to a suppressed cell rather than summing into it, which discards the bad value instead of combining with it, so neither a `NaN` nor a `+inf` score can escape the mask. Only the `-inf` cell is a suppression signal — a `NaN` or `+inf` mask cell stays additive (PyTorch parity for a non-finite *mask* is unchanged) and a finite `finfo.min` fill keeps its magnitude (so a fully-masked row still collapses to one constant).

## Library features used

| Capability | Where exercised |
|---|---|
| `ModernBertEncoder<T>` / `ModernBertLayer<T>` | 28 pre-norm layers, embed → blocks → final norm |
| `ModernBertMasks.Build` (banded bidirectional + padding) | 10 full layers (`band = -1`) and 18 sliding layers (`band = 64`), one `[128, 128]` additive 0/`-inf` mask each |
| `ModernBertConfig.LayerTypes` derivation | `global_attn_every_n_layers = 3` → 10 full + 18 sliding; also reads explicit `layer_types` + `rope_parameters` |
| Per-layer-type RoPE (`RopeThetaFull` 160000 / `RopeThetaSliding` 10000) | Q/K rotary, `rotate_half` split |
| `StateDictLoader.LoadLinearSlice` | Fused `Wqkv` → q/k/v row thirds; fused `Wi` → `inputProj`/`gateProj` halves (activated half is the *first*) |
| `ModernBertEncoder<T>.LoadWeights` prefix parameter | `model.` (stock HF) and `encoder.` (Laya's nested copy) off one loader |
| `Gpt2BpeTokenizer.LoadFromTokenizerJson` + opt-in NFC | Checkpoint ships only `tokenizer.json`; 128/128 token ids match the HF fixture |
| `GradKernels` safe-softmax clamp | Fully-masked sliding-window rows return zeros instead of `NaN` (regression-tested) |
| `Embedding<T>` | Token embeddings |
| `LayerNorm<T>` | Pre-norm layers |
| `Linear<T>` | QKV, attention output, MLP projections |
| `ReverseGradOperations.Gelu` | GeGLU activation |
| `ReverseGradOperations.Softmax` | Attention |
| `ReverseGradOperations.MatMul` | All projections |

## Verification

`compare` PASSED against HuggingFace 5.14.1 (`attn_implementation: "sdpa"`), 128 padded positions / 26 valid: **128/128 token ids identical**, `maxAbs 1.62e-5`, `meanAbs 9.22e-7`, `maxRel 6.09e-4`, **cosine 1.0000000000**, 0 non-finite values on either side in both the valid and padding regions (gate bound `|diff| <= 1e-3·(1 + |ref|)`).

Re-run after the encoder norms switched from an unloaded zero Beta to `LayerNorm<T>(..., bias: false)`. The gate reproduced that baseline at the reported precision: printed max abs diff `0.00001621`, mean abs diff `0.0000009215`, max rel diff `0.00060944`, cosine `1.0000000000`. The zero-Beta emulation and the real bias-free norm agree.

`compare_diag` matches at every stage from embeddings through layer 26 (cosine 1.000000, max|diff| ≤ 5.9e-3); the raw layer-27 output is absmax 25730.7 vs HF 25741.3 (0.04%), and the final norm is back to `max|diff| 1.6e-5`.

## Performance

| Input | PyTorch | Nivara | Slowdown |
|-------|---------|--------|----------|
| 128 tokens | 287.3 ms | 1249.8 ms | ~4.3× |
| 256 tokens | 504.0 ms | 1922.4 ms | ~3.8× |

Cold load (safetensors parse 1.85 s + weight load 4.85 s = 6.7 s) dominates short runs — ~5.4× a seq-128 forward. Nivara: 20.8 tok/s at seq 128, 13.5 tok/s at seq 256, ~44.6 and ~68.7 ms/layer. Each row is 1 warmup + 3 timed passes, median reported. `--seq N[,N...]` picks the lengths and `--warmup N` / `--iters N` the pass counts on both benchmark paths; the defaults are the tables above. The PyTorch side of the comparison uses 3 warmup passes, not 1 — `--warmup 3` matches it. See the [NivaraInference README](../samples/NivaraInference/README.md) for full benchmark tables.

## GPU path

`ModernBertGpuRunner` + `modernbert --gpu` (issue #449). The first GPU runner that shares no structure with DistilBERT or MiniLM: rotary instead of learned positions, gated FFN, pre-norm, and per-layer attention geometry.

The gate is GPU-vs-CPU in-process (no PyTorch fixture needed). Fixture seqLen 128, 26 valid positions:

| region | maxAbs | maxRel | cosine | violations |
|---|---|---|---|---|
| valid | **2.861E-005** | 5.577E-004 | 1.0000001 | 0 |
| padding | 6.866E-004 | 4.867E-003 | — | 0 |

Benchmark rows scale with sequence length: 128 → 400 ms, 512 → 2.30 s, 2048 → 19.6 s, 4096 → 63.0 s. `modernbert --gpu benchmark` defaults to exactly those four lengths, and `--seq` times a subset — the 4096 row that the CPU path cannot reach is now one flag away rather than a source edit. See [docs/ACCELERATION.md](ACCELERATION.md) §1b for full details.
