# DistilBERT family — DistilBERT, MiniLM, SST-2

## Model overview

Three BERT-family encoder models sharing the same Post-LN architecture pattern:

| Model | Layers | Hidden | Heads | Params | Output |
|-------|--------|--------|-------|--------|--------|
| DistilBERT (base-uncased) | 6 | 768 | 12 | 67.0M | `[seqLen, 768]` hidden states |
| MiniLM (L6-v2) | 6 | 384 | 6 | 22.7M | 384-dim sentence embedding |
| DistilBERT SST-2 | 6 | 768 | 12 | 66.9M | 2-class sentiment logits |

**DistilBERT** — the 6-layer, 768-dim pre-trained encoder (the baby-step before the fine-tuned SST-2 showcase):

- **Embedding stack**: word + position embeddings (no token-type embeddings) summed, then LayerNorm
- **6× Post-LN DistilBERT layers**: self-attention → residual → `sa_layer_norm` → FFN (`lin1` → GELU → `lin2`) → residual → `output_layer_norm`
- **GELU activation** in the FFN intermediate (exact erf)
- **Weight mapping** from `distilbert.*` SafeTensors keys via `DistilBertLoader.LoadEncoderWeights`
- **Verification**: `last_hidden_state` matches HuggingFace to `max abs diff 5e-6` (cosine 0.99999988)

**MiniLM** — a 6-layer Post-LN BERT encoder producing 384-dimensional sentence embeddings:

- **Embedding stack**: token + position + segment embeddings summed, then LayerNorm
- **6× Post-LN BERT layers**: LayerNorm → Self-Attention → residual → LayerNorm → FFN → residual
- **GELU activation** in the FFN intermediate (exact erf)
- **Bidirectional self-attention** with optional padding mask (via `MultiheadAttention<T>`)
- **[CLS] token pooling** — extracts the first token's embedding from the output sequence
- **L2 normalization** — output embedding normalized to unit length for cosine similarity
- **Tokenization** via `Microsoft.ML.Tokenizers.BertTokenizer` (sample-only dependency)

**DistilBERT SST-2** — the fine-tuned sequence-classification showcase: the base encoder plus a classification head that outputs binary sentiment logits.

- **Encoder**: identical to the base `distilbert` mode (word + position embeddings, 6 Post-LN layers, exact erf GELU)
- **No token-type embeddings** (`includeTokenTypeEmbedding: false`) — DistilBERT never feeds segment ids
- **Head**: `pre_classifier` (768→768) → **ReLU** → `classifier` (768→2). The HF architecture applies `nn.ReLU()` after `pre_classifier`; a naive port using `GeluExact` on the head produced logits off by ~0.05, so the head uses `ReverseGradOperations.Relu`
- **Softmax + argmax** for the sentiment label and confidence
- **Inference-default path**: `PredictLogits` runs outside any `Grad()` scope, producing leaf logits with no computation-graph overhead
- **Padded-input contract**: `BertEncoder.ForwardBatched` requires attention-mask tensors of length `batchSize * seqLen`; token IDs are passed as exact `int[]` (see the BFloat16 note) so they survive narrow-precision dtypes, and `PredictLogits` passes the padded `[maxLen]` token ids
- **Verification**: `compare` matches HuggingFace to `max abs logit diff 9.5e-7`, `argmax agreement 8/8`; the `bf16` mode matches the same reference at `8/8` argmax with a `max abs logit diff ~0.33` (genuine BFloat16 precision)

## How this model improved the library

The BERT-family models were the first text models in the sample. They surfaced:

- **`BertSelfAttention<T>`** — fused `ReverseGradOperations.MultiHeadAttention` kernel (#86): heads packed once per forward, QK^T/softmax/PV run as a single per-head pass over `TensorPrimitives` row kernels with no per-head `Slice`/`Transpose` graph nodes.
- **`DistilBertLoader.LoadEncoderWeights`** — explicit SafeTensors key mapping from `distilbert.*` and `distilbert.transformer.layer.{0-5}.*` to Nivara module parameters.
- **`DistilBertForSequenceClassification<T>`** — shared classifier model with `pre_classifier` → ReLU → `classifier` head, inference-default path (`PredictLogits` outside `Grad()` scope).
- **GELU exact vs tanh distinction** — BERT-family models use exact erf GELU (`GeluExact`); GPT-style models use the tanh approximation. Both paths retained.

> **GELU note:** BERT-family models (MiniLM, DistilBERT) use the exact erf GELU (`GeluExact`). The tanh approximation (`ReverseGradOperations.Gelu`) matches HF `gelu_new`/GPT-2 and is retained for GPT-style `TransformerBlock`.

## Library features used

| Capability | Where exercised |
|---|---|
| `Embedding<T>` Gather-based lookup | Token/position/segment embeddings (MiniLM) |
| `Embedding<T>` without token-type embeddings | `includeTokenTypeEmbedding: false` (DistilBERT) |
| `LayerNorm<T>` with affine parameters | After embedding, after each attention and FFN |
| `BertSelfAttention<T>` padding-mask path | 6 attention layers (768-dim, 12 heads) |
| `MultiheadAttention<T>` bidirectional mode, padding mask | MiniLM 6 attention layers |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `ReverseGradOperations.Relu` | SST-2 classification-head activation (matches HF `nn.ReLU`) |
| `ReverseGradOperations.Add` (residual) | Every residual connection |
| `ReverseGradOperations.Softmax` | SST-2 sentiment probability |
| `ReverseGradOperations.MatMul` | SST-2 classifier |
| `DistilBertLoader.LoadEncoderWeights` | `distilbert.*` SafeTensors weight mapping |
| `DistilBertForSequenceClassification<T>` | SST-2 classifier model (`pre_classifier` → ReLU → `classifier`) |
| `GradientUtils.Constant` | Padded token-id / attention-mask input tensors |
| `GradientUtils.Grad()`-free inference | Leaf logits, no computation graph overhead |
| `Module<T>.Eval()` | Inference mode (disables dropout) |
| `Module<T>` tree with `LoadStateDict` | Full model construction |
| `Microsoft.ML.Tokenizers` integration | BERT WordPiece tokenizer (MiniLM) |
| Softmax + argmax via tensor span | SST-2 sentiment label + confidence |

## Verification

| Model | Gate | Result |
|-------|------|--------|
| DistilBERT | `last_hidden_state` vs HuggingFace | max abs diff 5e-6, cosine 0.99999988 |
| MiniLM | `compare` vs CPU reference | hidden maxRel 1.0e-5, pooled maxRel 1.5e-7, 0 violations |
| SST-2 | `compare` vs HuggingFace | max abs logit diff 9.5e-7, argmax 8/8 |
| SST-2 bf16 | `compare` vs HuggingFace | argmax 8/8, max abs logit diff ~0.33 |

## Narrow-precision inference (BFloat16 / Half)

Every transformer model in this sample is generic over the compute dtype `T : IFloatingPointIeee754<T>`, so it runs in F32, BFloat16, or Half (fp16). See [docs/BFLOAT16.md](BFLOAT16.md) for engine-level details.

**What a narrow-precision mode does**
- Loads the on-disk **F32** weights as `BFloat16` (`SafeTensorsLoader.Read<BFloat16>`) or `Half` (`SafeTensorsLoader.Read<Half>`) — the loader truncates each `float` to the narrow dtype at load time.
- Builds the `<BFloat16>` / `<Half>` model via the generic `LoadWeights<...>` and runs the full forward pass in that dtype.
- Diffs the output against the same PyTorch reference fixtures used by `compare`.

**Token-ID correctness (the subtle bit)** — BFloat16 represents integers *exactly* only up to 256 and Half only up to 2048, but transformer vocabularies reach ~30k. Converting token IDs to a narrow-precision tensor before the embedding lookup corrupts them (e.g. `30522 → 30512` in BF16), sending the lookup to the wrong row and producing garbage (we measured a ~7.4 logit diff vs the F32 reference before the fix). The fix keeps token IDs as **exact `int`**: `Embedding<T>`, `BertEncoder<T>`, `MiniLMDistilled<T>` and `DistilBertForSequenceClassification<T>` all expose `Forward(int[] tokenIds, ...)` overloads that look up embeddings by exact integer index, independent of the compute dtype. Only the attention mask stays a narrow tensor (its `0`/`1` values round-trip exactly).

**Results** (against the F32 HuggingFace reference, CPU):

| Model | Metric | F32 vs Ref | BFloat16 vs Ref | Half (fp16) vs Ref |
|---|---|---|---|---|
| `distilbert_sst` | argmax agreement | 8/8 | **8/8** | **8/8** |
| `distilbert_sst` | max abs logit diff | ~1e-6 | **~0.33** | **~0.22** |

Half uses a 10-bit mantissa (vs BF16's 7), which is why its logits land closer to the F32 reference; both preserve every SST-2 prediction.

**Memory** — narrow precision stores each weight in 2 bytes (FP16/BF16) vs 4 for F32, so weight memory **exactly halves**:

| Model | F32 weights | FP16 / BF16 weights |
|---|---|---|
| MiniLM | ~91 MB | ~45.5 MB |
| DistilBERT (base) | ~255.5 MB | ~127.8 MB |
| DistilBERT SST-2 | ~255.4 MB | ~127.7 MB |

**Speed** — on CPU, the narrow matmul runs through non-SIMD fallbacks and is dramatically *slower* per pass than F32 (fp16 was ~26x slower, issue [#363](https://github.com/khurram-uworx/Nivara/issues/363)). Don't read narrow benchmarks as a CPU speed win — treat them as a memory trade that preserves every prediction.

## Performance

See [docs/SILK.md](SILK.md) §1 for GPU benchmark tables. CPU benchmarks live in the [NivaraInference README](../samples/NivaraInference/README.md).

## GPU path

All three models run on the promoted Silk OpenCL backend (`--silk`, F32-only); CPU remains the
reference. Comparative notes and implementation details for the sample-only ILGPU backend live in
[ILGPU.md](../docs/ILGPU.md).

### `--silk` (Silk.NET, hand-authored OpenCL C)

The same three models also run through the second backend: hand-authored OpenCL C
(`Silk.NET.OpenCL` 2.23.0) compiled in-process by the driver, in
`src/Nivara.Extensions/Silk/SilkBertEncoderRunner.cs`.

- **Encoder + head, eight kernels.** `embed_sum`, `gather_rows`, `layer_norm_residual`, the four Row4
  GEMM epilogues (`gemm_row4_bias`, `gemm_row4_relu`, `gemm_row4_gelu`, `gemm_row4_qkv`) and
  `batched_attention`. The SST-2 head runs entirely on the device — `gather_rows` (CLS row) →
  `gemm_row4_relu` → `gemm_row4_bias` — not a host round-trip.
- **Parity.** `distilbert --silk compare` — GPU vs same-process CPU, **0/98304 violations** at
  `|gpu−cpu| ≤ 1e-3·(1+|cpu|)` (the `last_hidden_state_py.bin` PyTorch fixture is absent, so this is
  GPU-vs-CPU only). `distilbert_sst --silk compare` — logits vs CPU and vs PyTorch, **0/16
  violations, argmax 8/8 both**.
- **Gate.** `--silk-kernels` holds `gemm_row4_bias`, `gemm_row4_relu`, `gemm_row4_qkv` and
  `gather_rows` to **byte-identity** against ILGPU over both MiniLM and DistilBERT shapes; DistilBERT
  drives `headDim` to 64, the attention kernel's `MaxHeadDim` local-memory bound.
- **Whole-forward A/B (AC, valid; both backends run the same benchmark body).** Median of 3 runs at
  128 tokens (3 warmup + 10 timed before each): DistilBERT **52.3 ms (Silk) vs 55.8 ms (ILGPU),
  ≈0.94×**; DistilBERT SST-2 **52.1 ms vs 57.0 ms, ≈0.91×**. Silk is at least at parity and modestly
  ahead, with a tighter run-to-run spread (Silk ≈1–3 ms, ILGPU up to ≈8 ms). One host (Arc 140T), one
  session (2026-10-13) — read it as parity-to-slightly-ahead, not a win. The absolute ILGPU values sit
  ≈10% below the post-M2 rows in [SILK.md](SILK.md) §1 — a different session's clock
  state, not a protocol change — so compare the ratio, not the absolute milliseconds. Per kernel, the
  `--silk-kernels` gate holds all 32 outputs bit-identical on AC, with Silk ahead on
  `batched_attention` (0.80–0.85).

Detail: [docs/SILK.md](SILK.md) Part 2.
