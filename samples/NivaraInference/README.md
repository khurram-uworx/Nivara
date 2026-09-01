# Nivara HuggingFace Inference Sample

Load pre-trained HuggingFace models (MobileNetV2, ResNet-18, MiniLM, DistilBERT, DistilBERT SST-2, SmolLM-135M-Instruct) into Nivara's zero-dependency tensor engine and run forward inference in pure managed .NET — no Python runtime, no CUDA, no third-party ML framework.

The same architecture is also implemented in PyTorch (`samples/NivaraInference/Python/`) for direct CPU performance comparison.

## Quick start

```bash
# Download model weights via HuggingFace CLI
hf download google/mobilenet_v2_1.0_224 --local-dir samples/data/mobilenet_v2
hf download microsoft/resnet-18 model.safetensors config.json --local-dir samples/data/resnet18
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir samples/data/minilm
# (distilbert-base-uncased already present under samples/data/distilbert)
hf download distilbert/distilbert-base-uncased-finetuned-sst-2-english config.json model.safetensors vocab.txt tokenizer_config.json --local-dir samples/data/distilbert_sst
# SmolLM-135M-Instruct — BF16-native Llama-family causal LM (GQA, SiLU, RoPE; see "SmolLM" below)
hf download HuggingFaceTB/SmolLM-135M-Instruct config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt generation_config.json special_tokens_map.json --local-dir samples/data/smollm-135m

# Run inference
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2
dotnet run --project samples/NivaraInference -c Release -- resnet18
dotnet run --project samples/NivaraInference -c Release -- minilm
dotnet run --project samples/NivaraInference -c Release -- distilbert
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst
dotnet run --project samples/NivaraInference -c Release -- smollm                 # greedy causal-LM generation (F32)
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16  # native BF16 (256.6 MB)

# Benchmark (10 passes each)
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark

# Narrow-precision inference (half weight memory; see "Narrow-precision inference" below)
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst bf16
dotnet run --project samples/NivaraInference -c Release -- distilbert bf16
dotnet run --project samples/NivaraInference -c Release -- minilm bf16
# fp16 / Half variant (--precision fp16, or a bare fp16/half positional)
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --precision fp16
dotnet run --project samples/NivaraInference -c Release -- distilbert --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm --precision fp16
# Benchmark also honors --precision (times F32 / fp16 / bf16; see "Speed" below)
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision bf16
```

## Supported models

| Model | Type | Weight size | Tensors | Parameters | Output |
|-------|------|-------------|---------|------------|--------|
| MobileNetV2 | Vision (classification) | 13.5 MB | 262 | 3.4M | 1001 classes |
| ResNet-18 | Vision (classification) | 44.6 MB | 102 | 11.7M | 1000 classes |
| MiniLM (L6-v2) | Text (embedding) | 91 MB | 104 | 22.7M | 384-dim embedding |
| DistilBERT (base-uncased) | Text (encoder) | 255.5 MB | 105 | 67.0M | `[seqLen, 768]` hidden states |
| DistilBERT SST-2 (fine-tuned) | Text (classification) | 255.4 MB | 104 | 66.9M | 2-class sentiment (`NEGATIVE`/`POSITIVE`) |
| SmolLM-135M-Instruct | Text (causal LM) | 269 MB | 272 | 134.5M | token ids (generation) |

## Usage

### C# (Nivara)

**Vision models:**
```bash
# Random-data inference
dotnet run --project samples/NivaraInference -- mobilenet_v2
dotnet run --project samples/NivaraInference -- resnet18

# Benchmark (10 synthetic + real-image passes)
dotnet run --project samples/NivaraInference -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -- resnet18 benchmark

# Compare output with PyTorch reference
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare
dotnet run --project samples/NivaraInference -- resnet18 compare

# Step-by-step layer diagnostics
dotnet run --project samples/NivaraInference -- mobilenet_v2 compare_diag
dotnet run --project samples/NivaraInference -- resnet18 compare_diag

# Single image inference
dotnet run --project samples/NivaraInference -- mobilenet_v2 path/to/image.jpg
dotnet run --project samples/NivaraInference -- resnet18 path/to/image.jpg
```

**MiniLM:**
```bash
# Tokenize and embed a sentence
dotnet run --project samples/NivaraInference -- minilm

# Benchmark (10 passes)
dotnet run --project samples/NivaraInference -- minilm benchmark

# Pairwise cosine similarity demo
dotnet run --project samples/NivaraInference -- minilm similarity
```

**DistilBERT:**
```bash
# Forward a sentence through the base encoder (output: [128, 768] hidden states)
dotnet run --project samples/NivaraInference -- distilbert

# Benchmark (3 warmup + 10 timed passes)
dotnet run --project samples/NivaraInference -- distilbert benchmark

# Compare hidden states with a PyTorch reference (run the Python script first)
python samples/NivaraInference/Python/distilbert_compare.py
dotnet run --project samples/NivaraInference -- distilbert compare
```

**DistilBERT SST-2 (sequence classification):**
```bash
# Interactive sentiment REPL over the fine-tuned SST-2 classifier
dotnet run --project samples/NivaraInference -- distilbert_sst

# Benchmark (3 warmup + 10 timed passes)
dotnet run --project samples/NivaraInference -- distilbert_sst benchmark

# Compare logits + softmax probs with a PyTorch reference (run the Python script first)
python samples/NivaraInference/Python/distilbert_sst_compare.py
dotnet run --project samples/NivaraInference -- distilbert_sst compare
```

**SmolLM-135M-Instruct (causal LM / generation):**
```bash
# Greedy generation from the fixed prompt (F32; add --precision bf16 for the native BF16 path,
# or fp16 for Half). Run the Python reference generator first to enable the PyTorch diff.
python samples/NivaraInference/Python/smollm_generate_reference.py
dotnet run --project samples/NivaraInference -- smollm
dotnet run --project samples/NivaraInference -- smollm --precision bf16
```

### Python (PyTorch reference)

```bash
cd samples/NivaraInference/Python
pip install -r requirements.txt

python mobilenet.py           # Basic inference
python resnet18.py

python mobilenet_compare.py   # Forward pass on shared input for C# comparison
python resnet18_compare.py

python mobilenet_diag.py      # Step-by-step layer diagnostics
python resnet18_diag.py

python minilm_benchmark.py     # MiniLM CPU timing (same methodology as C#)
python distilbert_benchmark.py # DistilBERT CPU timing (same methodology as C#)
python minilm_compare.py       # MiniLM reference embeddings for C# comparison
python distilbert_compare.py   # DistilBERT reference hidden states for C# comparison
python distilbert_sst_compare.py # DistilBERT SST-2 reference logits for C# comparison

python generate_input.py      # Regenerate shared comparison fixture
```

## Model architectures

### MobileNetV2

A lightweight classification network built from inverted residual blocks:

- **Stem**: 3×3 conv → BatchNorm → ReLU6
- **16 inverted residual blocks** with expansion/depthwise/project phases
- **Depthwise separable convolutions** (groups = input channels) for 3×3 layers
- **ReLU6** activation via `Clip(Relu(x), 0, 6)`
- **Residual shortcuts** only when `stride == 1 && inChannels == outChannels`
- **Head**: 1×1 conv → global avg pool → 1001-class linear classifier

Nivara modules used: `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `ReLU6` via `Clip` + `Relu`, depthwise grouped convolutions.

### ResNet-18

A standard 18-layer residual network:

- **Stem**: 7×7 conv → BatchNorm → ReLU → 3×3 MaxPool
- **4 stages** with channel progression: 64 → 128 → 256 → 512
- **BasicBlock**: two 3×3 convs with BatchNorm + ReLU, identity shortcut (or 1×1 conv when dimensions change)
- **Head**: global average pooling → 1000-class linear classifier
- **Downsampling** at stage boundaries via strided convolution in the shortcut path

Nivara modules used: `Conv2d<T>`, `BatchNorm2d<T>`, `Linear<T>`, `MaxPool2d<T>`, `AdaptiveAvgPool2d<T>`, residual addition via `ReverseGradOperations.Add`.

### MiniLM (sentence-transformers/all-MiniLM-L6-v2)

A 6-layer Post-LN BERT encoder producing 384-dimensional sentence embeddings:

- **Embedding stack**: token + position + segment embeddings summed, then LayerNorm
- **6× Post-LN BERT layers**: LayerNorm → Self-Attention → residual → LayerNorm → FFN → residual
- **GELU activation** in the FFN intermediate (exact erf)
- **Bidirectional self-attention** with optional padding mask (via `MultiheadAttention<T>`)
- **[CLS] token pooling** — extracts the first token's embedding from the output sequence
- **L2 normalization** — output embedding normalized to unit length for cosine similarity
- **Tokenization** via `Microsoft.ML.Tokenizers.BertTokenizer` (sample-only dependency)

Nivara modules used: `Embedding<T>` (Gather path), `LayerNorm<T>`, `Linear<T>`, `MultiheadAttention<T>`, `ReverseGradOperations.GeluExact`, `ReverseGradOperations.Add`.

### DistilBERT (distilbert-base-uncased)

The 6-layer, 768-dim pre-trained encoder (the baby-step before the fine-tuned SST-2 showcase):

- **Embedding stack**: word + position embeddings (no token-type embeddings) summed, then LayerNorm
- **6× Post-LN DistilBERT layers**: self-attention → residual → `sa_layer_norm` → FFN (`lin1` → GELU → `lin2`) → residual → `output_layer_norm`
- **GELU activation** in the FFN intermediate (exact erf)
- **Weight mapping** from `distilbert.*` SafeTensors keys via `DistilBertLoader.LoadEncoderWeights`
- **Verification**: `last_hidden_state` matches HuggingFace to `max abs diff 5e-6` (cosine 0.99999988)

Nivara modules used: `Embedding<T>`, `LayerNorm<T>`, `Linear<T>`, `BertSelfAttention<T>` (fused `ReverseGradOperations.MultiHeadAttention`), `ReverseGradOperations.GeluExact`, `ReverseGradOperations.Add`.

> **GELU note:** BERT-family models (MiniLM, DistilBERT) use the exact erf GELU (`GeluExact`). The tanh approximation (`ReverseGradOperations.Gelu`) matches HF `gelu_new`/GPT-2 and is retained for GPT-style `TransformerBlock`.

### DistilBERT SST-2 (distilbert-base-uncased-finetuned-sst-2-english)

The fine-tuned sequence-classification showcase: the base encoder plus a classification head that outputs binary sentiment logits.

- **Encoder**: identical to the base `distilbert` mode (word + position embeddings, 6 Post-LN layers, exact erf GELU)
- **No token-type embeddings** (`includeTokenTypeEmbedding: false`) — DistilBERT never feeds segment ids
- **Head**: `pre_classifier` (768→768) → **ReLU** → `classifier` (768→2). The HF architecture applies `nn.ReLU()` after `pre_classifier`; a naive port using `GeluExact` on the head produced logits off by ~0.05, so the head uses `ReverseGradOperations.Relu`
- **Softmax + argmax** for the sentiment label and confidence
- **Inference-default path**: `PredictLogits` runs outside any `Grad()` scope, producing leaf logits with no computation-graph overhead
- **Padded-input contract**: `BertEncoder.ForwardBatched` requires attention-mask tensors of length `batchSize * seqLen`; token IDs are passed as exact `int[]` (see the BFloat16 note) so they survive narrow-precision dtypes, and `PredictLogits` passes the padded `[maxLen]` token ids
- **Verification**: `compare` matches HuggingFace to `max abs logit diff 9.5e-7`, `argmax agreement 8/8`; the `bf16` mode matches the same reference at `8/8` argmax with a `max abs logit diff ~0.33` (genuine BFloat16 precision)

Nivara modules used: `DistilBertForSequenceClassification<T>` (shared from `Nivara.Samples`), `Embedding<T>`, `LayerNorm<T>`, `Linear<T>`, `BertSelfAttention<T>`, `ReverseGradOperations.GeluExact` (encoder FFN), `ReverseGradOperations.Relu` (head), `ReverseGradOperations.Softmax`, `ReverseGradOperations.MatMul`.

### SmolLM-135M-Instruct (HuggingFaceTB/SmolLM-135M-Instruct)

The **5th HuggingFace model** (and first causal LM / generative model in the sample)
and the primary driver for the BF16 widening work
(`docs/BFLOAT16-TRANSFORMER.md`). It is a **BF16-native** Llama-family causal LM —
all 272 on-disk tensors are `BF16` (269 MB), exercising the native
`SafeTensorsLoader.Read<BFloat16>` zero-hop path (unlike the other 4 models, which
are F32 on disk). The Nivara side runs the full stack in Nivara's AutoDiff engine
over the model ops below and greedily decodes a response.

- **Config**: `hidden_size=576`, `intermediate_size=1536`, 30 layers,
  `num_attention_heads=9`, **`num_key_value_heads=3` (GQA)**, `hidden_act=silu`
  (gated FFN), RMSNorm (`eps=1e-5`), RoPE (`theta=10000`),
  `max_position_embeddings=2048`, `vocab_size=49152`, `tie_word_embeddings=true`
- **Tokenizer**: **GPT-2 byte-level BPE** (not SentencePiece — see the note below),
  chat variant (`<|im_start|>`/`<|im_end|>` template; bos `<|im_start|>`,
  eos/pad `<|im_end|>`), 49152-token vocab built from `vocab.json` + `merges.txt`
- **Nivara modules added for this model**: `RMSNorm<T>` (affine gamma), `Activation.Silu`,
  `RotaryEmbedding<T>` (RoPE, Llama `rotate_half` half-split layout),
  `LlamaCausalAttention<T>` (GQA 9↔3 KV heads via KV-repeat), `LlamaDecoderBlock<T>`
  (pre-norm attention + residual + pre-norm gated SiLU FFN + residual),
  `LlamaForCausalLM<T>` (embed → 30 blocks → final RMSNorm → tied-embedding LM head),
  and `Gpt2BpeTokenizer` (sample-local byte-level BPE reader)
- **Tied LM head**: the input embedding weight is reused as the output projection —
  the checkpoint has no separate LM-head tensors
- **Reference fixture**: `Python/smollm_generate_reference.py` saves the token-id
  stream and final-position logits for diffing:

  ```bash
  python samples/NivaraInference/Python/smollm_generate_reference.py
  # -> samples/data/compare_smollm_py.bin, samples/data/compare_smollm_logits_py.bin
  ```

  The C# `smollm generate` mode diffs against these when present (see "Causal-LM
  generation" below).

**Usage:**

```bash
# Greedy causal-LM generation (F32, BF16 native on disk, or fp16)
dotnet run --project samples/NivaraInference -c Release -- smollm
dotnet run --project samples/NivaraInference -c Release -- smollm --precision bf16
dotnet run --project samples/NivaraInference -c Release -- smollm --precision fp16
```

Each run loads SmolLM-135M, tokenizes the fixed prompt *"The capital of France is"*,
greedily decodes up to 32 new tokens (inference-only: no `GradientUtils.Grad()` scope,
so no graph nodes are built), prints the token ids + decoded text, and — when the
PyTorch reference fixtures exist — diffs the token-id stream and final-position
logits.

**BF16 SIMD widening**: with scalar BFloat16 math, a 32-token generation is
impractical (~100× slower). The `smollm` mode therefore enables
`NivaraPrimitives.UseWidenSimd` for the narrow (BFloat16/Half) runs so the Phase-1
widen-compute-narrow SIMD kernels drive the matmuls (and restores the prior global
value afterwards, so other model modes are unaffected).

**Numerical caveats** (documented tolerance, not bit-exact): greedy argmax agreement
with the PyTorch reference is high but not perfect — F32 matches ~30/32 generated
tokens (decoded text byte-identical through the first ~30; the tail diverges because a
small numeric difference at a near-tie flips argmax and the error compounds), and BF16
matches ~22/32 with a final-position-logits cosine similarity of ~0.94 vs the
reference. This is the expected "numeric precision diff" behavior for a single forward
step, not a structural mismatch.

> **Tokenizer correction (historical)**: this README previously listed SmolLM's
> tokenizer as SentencePiece. It is actually a **GPT-2 byte-level BPE** tokenizer
> (`tokenizer_class: GPT2Tokenizer`, `add_prefix_space: false`). The
> `Microsoft.ML.Tokenizers` BPE path cannot reproduce SmolLM's byte-level token IDs
> (every pre-tokenizer variant diverges at space-prefixed tokens), so a sample-local
> `Gpt2BpeTokenizer` (HF `bytes_to_unicode` map + GPT-2 regex + ranked greedy merges)
> implements the reader.

#### Gaps found & fixed (5th model)

Adding the causal-LM path surfaced ops Nivara did not yet have. Each was implemented,
unit-tested, and verified end-to-end against the PyTorch reference:

- **`RMSNorm<T>`** with affine gamma (`Llama` RMSNorm uses per-channel `weight`, unlike
  the plain mean/var normalization already present).
- **`Activation.Silu`** (`x·sigmoid(x)`, forward + VJP + JVP) — Llama uses SiLU gating,
  not GELU.
- **`RotaryEmbedding<T>`** (RoPE) + `GradKernels.RotaryForward/Backward` — precomputed
  cos/sin from `rope_theta`. Caught and fixed a subtle layout bug during end-to-end
  verification: the first implementation used the GPT-NeoX **interleaved-pairwise**
  rotation, but the Llama family uses HF **`rotate_half` (half-split)**, which
  anti-correlated the logits (cosine −0.92); after the fix the F32 greedy agreement went
  4/32 → 30/32 with byte-identical text.
- **`LlamaCausalAttention<T>`** — **GQA** (9 Q / 3 KV heads) via KV-repeat
  (`ReverseGradOperations.GqaRepeatKV` + `GradKernels.HeadRepeat`), fused causal masked
  attention loop.
- **`LlamaDecoderBlock<T>`** — pre-norm attention + residual, pre-norm **gated SiLU
  FFN** (`down(silu(gate)⊙up)`) + residual.
- **`LlamaForCausalLM<T>`** — embed → 30 blocks → final RMSNorm → **tied-embedding LM
  head** (reuses the input embedding weight).
- **`Gpt2BpeTokenizer`** — sample-local **GPT-2 byte-level BPE** reader (see the
  tokenizer-correction note above).
- **BF16 SIMD widening** enabled on the `smollm` narrow runs so generation is practical
  (see the BF16 section above).

### Weight loading

Each model defines a static `LoadWeights()` factory that maps HuggingFace tensor names to Nivara module parameters. No reflection or generic deserialization — explicit, type-safe loading with full compile-time checking.

- **MobileNetV2**: 262 tensors mapped to 262 module parameters (Conv2d weight/bias, BatchNorm running mean/var/weight/bias, Linear weight/bias)
- **ResNet-18**: 102 tensors mapped to 102 module parameters
- **MiniLM**: 96 tensors mapped from HuggingFace keys like `encoder.layers.N.attention.self.query.weight` to Nivara `Linear<T>` weight/bias fields
- **DistilBERT**: 105 tensors mapped via `DistilBertLoader.LoadEncoderWeights` from `distilbert.embeddings.*` and `distilbert.transformer.layer.{0-5}.*` keys
- **DistilBERT SST-2**: 104 tensors — 102 encoder tensors via `DistilBertLoader.LoadEncoderWeights` + `pre_classifier.{weight,bias}` and `classifier.{weight,bias}` loaded via `DistilBertForSequenceClassification<T>.LoadWeights`
- **SmolLM-135M**: 272 tensors (all BF16 on disk) via `LlamaLoader.Load<TModel,TWeight>` + `LlamaConfig.FromJson` — maps `model.embed_tokens.weight` (reused for the tied LM head), `model.layers.N.*` (input_layernorm, self_attn.{q,k,v,o}_proj, post_attention_layernorm, mlp.{gate,up,down}_proj), and `model.norm.weight`, with RMSNorm/attention/MLP weights bound via `StateDictLoader.LoadRMSNorm`/`LoadLinear`

## Narrow-precision inference (BFloat16 / Half)

Every transformer model in this sample is generic over the compute dtype
`T : IFloatingPointIeee754<T>`, so it runs in F32, BFloat16, or Half (fp16).
A `--precision` argument selects the compute dtype (text models only; the
vision samples stay F32):

```bash
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst --precision bf16
dotnet run --project samples/NivaraInference -c Release -- distilbert --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm --precision fp16
```

`--precision` accepts `f32` (default), `bf16`, or `fp16`. A bare
`bf16` / `fp16` / `half` positional is also accepted (`-- distilbert_sst bf16`), so
the pre-#341 `bf16` invocations keep working unchanged.

**What a narrow-precision mode does**
- Loads the on-disk **F32** weights as `BFloat16` (`SafeTensorsLoader.Read<BFloat16>`) or
  `Half` (`SafeTensorsLoader.Read<Half>`) — the loader truncates each `float` to the
  narrow dtype at load time (analogous to PyTorch loading an F32 checkpoint into a
  `torch.bfloat16`/`torch.float16` model; the file on disk stays F32).
- Builds the `<BFloat16>` / `<Half>` model via the generic `LoadWeights<...>` and runs the
  full forward pass in that dtype.
- Diffs the output against the same PyTorch reference fixtures used by `compare` (logits for
  SST-2; normalized embeddings / L2 norms for MiniLM and DistilBERT).

**Token-ID correctness (the subtle bit)** — BFloat16 represents integers *exactly* only up to
256 and Half only up to 2048, but transformer vocabularies reach ~30k. Converting token IDs to a
narrow-precision tensor before the embedding lookup corrupts them (e.g. `30522 → 30512` in BF16),
sending the lookup to the wrong row and producing garbage (we measured a ~7.4 logit diff vs the
F32 reference before the fix). The fix keeps token IDs as **exact `int`**: `Embedding<T>`,
`BertEncoder<T>`, `MiniLMDistilled<T>` and `DistilBertForSequenceClassification<T>` all expose
`Forward(int[] tokenIds, ...)` overloads that look up embeddings by exact integer index,
independent of the compute dtype. Only the attention mask stays a narrow tensor (its `0`/`1`
values round-trip exactly). See `docs/BFLOAT16.md` for the engine-level details.

**Results** (against the F32 HuggingFace reference, CPU):

| Model | Metric | F32 vs Ref | BFloat16 vs Ref | Half (fp16) vs Ref |
|---|---|---|---|---|
| `distilbert_sst` | argmax agreement | 8/8 | **8/8** | **8/8** |
| `distilbert_sst` | max abs logit diff | ~1e-6 | **~0.33** | **~0.22** |

Half uses a 10-bit mantissa (vs BF16's 7), which is why its logits land closer to the F32
reference; both preserve every SST-2 prediction.

**Memory** — narrow precision stores each weight in 2 bytes (FP16/BF16) vs 4 for
F32, so weight memory **exactly halves** (same parameter count, half the bytes):

| Model | F32 weights | FP16 / BF16 weights |
|---|---|---|
| MiniLM | ~91 MB | ~45.5 MB |
| DistilBERT (base) | ~255.5 MB | ~127.8 MB |
| DistilBERT SST-2 | ~255.4 MB | ~127.7 MB |
| SmolLM-135M | ~513 MB (widened) | **~256.6 MB (native on disk)** |

**Speed** — `benchmark` now accepts `--precision` (all three dtypes), so you can time F32,
fp16, and bf16 inference for the same model in one generic code path (3 warmup + 10 timed
passes, avg/min/max ms, with params + weight MB reported):

```bash
# F32, fp16, bf16 MiniLM
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision fp16
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark --precision bf16

# Same for distilbert / distilbert_sst (e.g. --precision fp16)
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark --precision fp16
```

Measured on CPU, MiniLM (seqLen 128), single thread:

| Precision | Avg ms/pass | Weight MB |
|-----------|-------------|-----------|
| F32       | ~142 | 86.6 |
| Half      | ~3658 | 43.3 |
| BFloat16  | similar to Half (see issue #363) | 43.3 |

The **halved weight memory** is the narrow-precision win; on CPU the narrow matmul runs
through non-SIMD fallbacks and is dramatically *slower* per pass than F32 (fp16 was ~26x
slower in the measurement above, issue [#363](https://github.com/khurram-uworx/Nivara/issues/363)).
Don't read narrow benchmarks as a CPU speed win — treat them as a memory trade that preserves
every prediction.

The base `distilbert` and `minilm` narrow-precision modes run correctly (unit-length
embeddings, sensible cosine similarities — e.g. 0.90 between "I love programming" and "I love
coding"). The column/tensor engine's BFloat16 path is documented in `docs/BFLOAT16.md`.

SmolLM differs from the other models: it is **BF16-native on disk**, so the `smollm
--precision bf16` mode reads the weights directly (no F32→BF16 truncation at load).
Because a scalar-BF16 32-token generation is impractical, the `smollm` BF16/Half runs
enable `NivaraPrimitives.UseWidenSimd` so matmuls flow through the Phase-1
widen-compute-narrow SIMD kernels (see `docs/BFLOAT16-TRANSFORMER.md`).

**Reference fixtures for `compare` / narrow-precision diffs** — the quantitative cosine (or
logit) diff against the HuggingFace reference is shown only when the F32 reference `.bin` files
exist. They are **not checked into the repo** (they live in / beside the gitignored model-weight
directories, which hold the multi-hundred-MB checkpoints), but each has a local Python generator.
Run them once on-demand to enable the diffs:

```bash
# Base DistilBERT hidden states -> samples/data/distilbert/last_hidden_state_py.bin
python samples/NivaraInference/Python/distilbert_compare.py
# MiniLM embeddings -> samples/data/compare_minilm_embeddings_py.bin
python samples/NivaraInference/Python/minilm_compare.py
# DistilBERT SST-2 logits -> samples/data/compare_distilbert_sst_py.bin
python samples/NivaraInference/Python/distilbert_sst_compare.py
```

Without a fixture the relevant mode prints "reference not found; skipping diff" and otherwise
runs normally (e.g. the SST-2 mode still prints each predicted label).

## SafeTensors loader

The sample includes a custom zero-dependency `SafeTensorsLoader` that parses the HuggingFace SafeTensors binary format directly:

- **Memory-mapped header parsing** via `System.Text.Json` — reads the JSON header from the first 8 bytes + offset table
- **Zero-copy tensor extraction** using `MemoryMarshal.Cast<byte, float>` — the weight data is reinterpret-cast directly from the memory-mapped file buffer
- **Dtype support** — loads **F32** (native `float`), **F16** (`System.Numerics.Half`), and **BF16** (`System.Numerics.BFloat16`) tensors, converting each to the requested result type `T` via `T.CreateChecked`. Narrow on-disk dtypes are widened when `T` is wider (e.g. a BF16 checkpoint read as `float[]` widens losslessly), and a wider on-disk dtype is narrowed when `T` is `BFloat16` (e.g. the `bf16` run mode reads the on-disk F32 weights as `BFloat16`, truncating to genuine 7-bit mantissa). Any other dtype raises `NotSupportedException` with guidance.

## Performance benchmarks

Measured on the same machine (CPU-only, no GPU): Intel Core Ultra 7 255H
(16 logical processors), Nivara in Release mode, PyTorch with MKL-optimized kernels.
Both use batch size 1 with 3-pass warmup + 10 timed passes. Both columns were
recorded in the same session. Numbers vary with machine load — only the same-row
PyTorch-vs-Nivara ratio is meaningful.

| Model | Input | PyTorch (CPU) | Nivara (.NET 10) | Slowdown |
|-------|-------|---------------|-------------------|----------|
| **MobileNetV2** | 1×3×224×224 | 22 ms | 665 ms | **~30×** |
| **ResNet-18** | 1×3×224×224 | 14 ms | 251 ms | **~18×** |
| **MiniLM-L6** | 128 tokens | 11 ms | 64 ms | **~6×** |
| **DistilBERT** | 128 tokens | 35 ms | 185 ms | **~5×** |
| **DistilBERT SST-2** | 128 tokens | 35 ms | 184 ms | **~5×** |

*Recorded 2026-08-21 — Intel Core Ultra 7 255H, 16 logical processors, .NET 10.0.11, PyTorch 2.13.0+cpu, Polars 1.43.2.*

The SST-2 row reuses the DistilBERT PyTorch timing (same architecture, only the
weights differ; `Python/distilbert_sst_compare.py` is accuracy-only, no timing).
PyTorch vision is multi-threaded MKL; Nivara's conv kernels are single-threaded
naive loops, which widens the vision gap on this low-power 4-core CPU — the
transformer gap (~6×) is the more representative figure on this machine.

AutoDiff graph nodes are only created inside `GradientUtils.Grad()` scopes (used by `TrainingLoop` and manual training code). Inference passes outside `Grad()` produce leaf tensors with no computation graph overhead. The AutoDiff refactor closed most of the gap: on the 2026-08-04 machine it cut vision inference ~4× (MobileNetV2 ~2,254 ms → ~563 ms, ResNet-18 ~641 ms → ~263 ms) and transformers ~1.5× (MiniLM ~110 → ~73 ms, DistilBERT ~186 → ~164 ms, SST-2 ~232 → ~187 ms). The vision gap is dominated by convolution kernels (especially depthwise convolutions in MobileNetV2), which use naive nested loops — ResNet-18 benefits from fewer depthwise layers. Transformer inference runs on a transpose-free path: `Linear` passes the raw weight `[out, in]` directly to the kernel's transposed-B matmul (no per-forward weight transpose), bias is applied via a row-broadcast `AddBias` op, op results are wrapped without a copy, and LayerNorm/Gelu/GeluExact skip saved-state allocations when gradients are not tracked. Attention runs through the fused `ReverseGradOperations.MultiHeadAttention` kernel (#86): heads are packed once per forward and QK^T/softmax/PV run as a single per-head pass over `TensorPrimitives` row kernels with no per-head `Slice`/`Transpose` graph nodes, keeping DistilBERT encoder inference at ~508 ms on this laptop.

## Sample data

| File | Purpose |
|------|---------|
| `samples/data/mobilenet_v2/model.safetensors` | MobileNetV2 weights (~13.5 MB) |
| `samples/data/resnet18/model.safetensors` | ResNet-18 weights (~44.6 MB) |
| `samples/data/minilm/model.safetensors` | MiniLM weights (~87 MB) |
| `samples/data/minilm/config.json` | MiniLM BERT config |
| `samples/data/minilm/vocab.txt` | MiniLM wordpiece vocabulary |
| `samples/data/distilbert/model.safetensors` | DistilBERT weights (~255.5 MB, 105 tensors) |
| `samples/data/distilbert/config.json` | DistilBERT config |
| `samples/data/distilbert/vocab.txt` | DistilBERT wordpiece vocabulary |
| `samples/data/distilbert/last_hidden_state_py.bin` | PyTorch reference hidden states (generated by `Python/distilbert_compare.py`) |
| `samples/data/distilbert_sst/model.safetensors` | Fine-tuned DistilBERT SST-2 weights (~255.4 MB, 104 tensors) |
| `samples/data/distilbert_sst/config.json` | DistilBERT SST-2 config (`dim=768`, `n_layers=6`, `n_heads=12`, 2 labels) |
| `samples/data/distilbert_sst/vocab.txt` | DistilBERT wordpiece vocabulary |
| `samples/data/compare_distilbert_sst_py.bin` | PyTorch reference logits + softmax probs (generated by `Python/distilbert_sst_compare.py`) |
| `samples/data/smollm-135m/model.safetensors` | SmolLM-135M-Instruct weights (~269 MB, 272 tensors, all BF16) |
| `samples/data/smollm-135m/config.json` | SmolLM-135M config (`hidden=576`, `n_layers=30`, GQA 9/3, SiLU, RoPE) |
| `samples/data/smollm-135m/tokenizer.json` | SmolLM tokenizer (GPT-2 byte-level BPE; `<|im_start|>`/`<|im_end|>` chat template) |
| `samples/data/compare_smollm_py.bin` | PyTorch reference token-id stream (generated by `Python/smollm_generate_reference.py`) |
| `samples/data/compare_smollm_logits_py.bin` | PyTorch reference final-position logits (generated by `Python/smollm_generate_reference.py`) |
| `samples/data/compare_input.bin` | Shared `[1,3,224,224]` input for compare modes (generated by `Python/generate_input.py`) |
| `samples/data/images/` | Synthetic test images at various resolutions (created by `Python/create_images.py`) |

## Nivara capabilities exercised

### Vision models

| Capability | Where exercised |
|---|---|
| `Conv2d<T>` with asymmetric padding, grouped convs, 1×1 fast path | All conv layers in both models |
| `BatchNorm2d<T>` with running statistics | Every conv → BN block |
| `MaxPool2d<T>` with argmax | ResNet-18 stem |
| `AdaptiveAvgPool2d<T>` with gradient broadcast | Both model heads |
| `Linear<T>` with MatMul + bias | Classifier heads |
| `Module<T>` tree with `LoadStateDict` | Full model construction |
| Depthwise separable convolutions (groups = channels) | MobileNetV2 3×3 blocks |

### MiniLM (text)

| Capability | Where exercised |
|---|---|
| `Embedding<T>` Gather-based lookup | Token/position/segment embeddings |
| `LayerNorm<T>` with affine parameters | After embedding, after each attention and FFN |
| `MultiheadAttention<T>` bidirectional mode, padding mask | 6 attention layers |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `ReverseGradOperations.Add` (residual) | Every residual connection |
| `Module<T>.Eval()` | Inference mode (disables dropout) |
| `Microsoft.ML.Tokenizers` integration | BERT WordPiece tokenizer |

### DistilBERT (text)

| Capability | Where exercised |
|---|---|
| `Embedding<T>` without token-type embeddings | `includeTokenTypeEmbedding: false` |
| `BertSelfAttention<T>` padding-mask path | 6 attention layers (768-dim, 12 heads) |
| `ReverseGradOperations.GeluExact` | FFN intermediate activation (exact erf) |
| `DistilBertLoader.LoadEncoderWeights` | `distilbert.*` SafeTensors weight mapping |

### DistilBERT SST-2 (text classification)

| Capability | Where exercised |
|---|---|
| `DistilBertForSequenceClassification<T>` | Shared classifier model (`pre_classifier` → ReLU → `classifier`) |
| `ReverseGradOperations.Relu` | Classification-head activation (matches HF `nn.ReLU`) |
| `GradientUtils.Constant` | Padded token-id / attention-mask input tensors |
| `GradientUtils.Grad()`-free inference | Leaf logits, no computation graph overhead |
| `MiniLMTokenizer.Encode` + `Microsoft.ML.Tokenizers.BertTokenizer` | WordPiece tokenization with `[CLS]`/`[SEP]` |
| Softmax + argmax via tensor span | Sentiment label + confidence |

### SmolLM-135M-Instruct (causal LM / generation)

| Capability | Where exercised |
|---|---|
| `RMSNorm<T>` affine gamma | Pre-norm in every decoder block + final norm |
| `Activation.Silu` (forward/VJP/JVP) | Gated SiLU FFN gate path |
| `RotaryEmbedding<T>` (RoPE, `rotate_half`) | Q/K rotary position embeddings |
| `LlamaCausalAttention<T>` + `GqaRepeatKV` | GQA self-attention (9 Q / 3 KV) |
| `LlamaDecoderBlock<T>` | Pre-norm attention + gated SiLU FFN + residuals |
| `LlamaForCausalLM<T>` + tied LM head | Embed → blocks → final norm → `hidden @ embed^T` |
| `Gpt2BpeTokenizer` | Sample-local GPT-2 byte-level BPE tokenization |
| `NivaraPrimitives.UseWidenSimd` | SIMD widen-compute-narrow BF16 matmul (native path) |
| Greedy generation (inference-default) | 32-token decode, no `GradientUtils.Grad()` scope |

## Release Benchmark

Run this during release prep (step 5 of `RELEASING.md`). Requires Python, PyTorch,
and HuggingFace model weights (see Quick start for `hf download` commands).

Run both sides in the same session for fair comparison:

```powershell
# Nivara (C#) — one pass per model
dotnet run --project samples/NivaraInference -c Release -- mobilenet_v2 benchmark
dotnet run --project samples/NivaraInference -c Release -- resnet18 benchmark
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert benchmark
dotnet run --project samples/NivaraInference -c Release -- distilbert_sst benchmark

# PyTorch (Python) — run immediately after on the same machine
cd samples/NivaraInference/Python
python benchmark.py
```

**Update the Performance benchmarks table:**
1. Shift existing timing columns to **Prev (PyTorch)** / **Prev (Nivara)**.
2. Place fresh measurements in **Current (PyTorch)** / **Current (Nivara)**.
3. Add **Prev Slowdown** (old ratio) and **Current Slowdown** (new ratio).
   Alternatively, keep single columns and add a **Δ%** column for Nivara only.
4. Update the machine line, recording date, and prose referencing ratios.
