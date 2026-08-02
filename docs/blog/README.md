# "The Signal Chain: A Software Engineer's Field Guide to a BERT"

Source material for a 4-post blog series. The posts are **not** "introducing Nivara" —
they are "what I learned implementing a transformer from scratch in C#", using the
`all-MiniLM-L6-v2` sentence-embedding model as the running example because almost every
software engineer today has *used* an embedding model, RAG pipeline, or chatbot, but few
have looked inside one.

Everything in these files is grounded in the actual implementation. Every number, formula,
and snippet can be traced back to a `file:line` reference at the end of each post.

---

## Audience & voice

- **Audience:** software engineers, 0–10 years of experience, who keep a polite distance
  from the maths of ML.
- **Voice:** how circuit-analysis / logic-design courses were taught — "the AND gate does
  *this* to the signals." Every layer is a block with a defined I/O contract (a datasheet).
- **The one-sentence thesis of the whole series:**
  > A transformer is built from the same parts engineers already know — a lookup table,
  > a multiply–accumulate, a gain-control stage, a soft switch, a bypass wire, and a
  > rectifier. If you can trace shapes and read a datasheet, you can read a transformer.

## Running example (used in every post)

Two sentences through MiniLM → two 384-dimensional vectors → cosine similarity:

```
"This is a cat."            →  embedding_A   (384 floats)
"This is a dog."            →  embedding_B   (384 floats)
"I love programming."       →  embedding_C
"The weather is nice today."→  embedding_D
"I love coding."            →  embedding_E

cosine(A, B) should be high, cosine(A, C) low, cosine(C, E) high.
```

Live demo: `dotnet run --project samples/NivaraInference -c Release -- minilm similarity`
(the exact sentence list lives in `samples/NivaraInference/Program.cs:1185-1192`).

## The signal chain (the map everything hangs off)

```
"the cat sat"
   │  WordPiece tokenizer (the lexer)
   ▼
[128] token ids          [101, 2361, 2892, 2893, 102, 0, 0, ...]   (padded to 128)
   │  token lookup + position lookup (+ segment lookup) → summed → LayerNorm
   ▼
[128, 384]  embedding matrix
   │  ×6 identical BERT layers, each:
   │      Attn(x) → +x (residual) → LayerNorm → Linear(384→1536) → GELU → Linear(1536→384) → +prev → LayerNorm
   ▼
[128, 384]
   │  take row 0 (the [CLS] token) → "pool"
   ▼
[384]
   │  L2 normalize (scale to unit length)
   ▼
[384]  unit vector  ←  now cosine similarity == dot product
```

## Model facts (shared across all posts)

`sentence-transformers/all-MiniLM-L6-v2` (a MiniLM distilled BERT encoder, 12 heads).

| Config key | Value | Notes |
|---|---|---|
| `hidden_size` | 384 | every token's vector width |
| `num_attention_heads` | 12 | head dim = 384/12 = **32** |
| `num_hidden_layers` | 6 | identical stacked BERT layers |
| `intermediate_size` | 1536 | FFN widens 384 → 1536 |
| `max_position_embeddings` | 512 | position table length |
| `vocab_size` | 30522 | token lookup table row count |
| `layer_norm_eps` | 1e-12 | **not** the PyTorch default 1e-5 |

Defaults mirror `BertConfig` (`samples/Nivara.Samples/BertModel.cs:8-16`).

**Parameter budget (~22.6M, computed from config):**

| Component | Params | % of model |
|---|---|---|
| Token embedding table (30522 × 384) | 11,720,448 | ~52% |
| Position embedding (512 × 384) | 196,608 | |
| Segment embedding (2 × 384) | 768 | |
| Embedding LayerNorm (2 × 384) | 768 | |
| Each BERT layer | 1,774,464 | ×6 = 10,646,784 |
| — of which attention (4 × Linear(384,384)+bias) | 591,360 | |
| — of which FFN (fc1 + fc2) | 1,181,568 | |
| — of which LayerNorms (2 × 768) | 1,536 | |
| **Total** | **22,565,376** | (~22.7M as listed in the sample README) |

**Key shapes:** input `[128]` token IDs → embeddings `[128, 384]` → attention scores
`[128, 128]` → output `[128, 384]` → pooled `[384]`.

## Performance numbers to cite

Measured same machine, CPU only, batch size 1, 3-pass warmup + 10 timed passes.
Source: `samples/NivaraInference/README.md:218-230`.

| Model | Input | PyTorch (CPU) | Nivara (.NET 10) | Slowdown |
|---|---|---|---|---|
| MobileNetV2 | 1×3×224×224 | 115 ms | 2,254 ms | ~20× |
| ResNet-18 | 1×3×224×224 | 68 ms | 641 ms | ~9× |
| **MiniLM-L6** | **128 tokens** | **11 ms** | **110 ms** | **~10×** |
| DistilBERT | 128 tokens | 31 ms | 186 ms | ~6× |
| DistilBERT SST-2 | 128 tokens | 31 ms | 232 ms | ~8× |

The MiniLM gap is ~10×. The honest engineering takeaway (used in Post 3/4): the gap is in
the matmul kernels (naive loops vs MKL), not in the architecture or the concepts.

**Fused-attention win (Post 3):** replacing per-head Slice/Transpose graph ops with one
fused packed-heads kernel cut DistilBERT encoder inference **~236 ms → ~186 ms (~21%)**
(`README.md:230`).

## The "what I learned" anecdotes (seed material)

1. **Shapes are the type system.** Most bugs were shape mismatches. Tracing
   `[L,D] × [D,D]` like a compiler caught almost everything.
2. **The transpose-free inference trick.** Linear weights are stored `[out, in]`.
   Inference feeds them straight into a transposed-B matmul — zero per-forward transposes.
   The transposed copy is only built (and cached) when gradients are on.
3. **The ε gotcha.** BERT's `layer_norm_eps` is `1e-12`; PyTorch's LayerNorm default is
   `1e-5`. Copy-pasting the wrong one is *silent* signal degradation — no error thrown.
4. **The −∞ mask trick.** Padding positions are suppressed by *adding* `-inf` to their
   attention scores before softmax → `e^-inf = 0`. No branching in the hot loop; the mask
   is just another input.
5. **GELU is a spec, not a choice.** exact erf GELU (BERT family: MiniLM, DistilBERT),
   tanh-approx GELU (GPT family, HF `gelu_new`), and ReLU all look alike and differ
   numerically. Using exact GELU where HuggingFace uses ReLU shifted SST-2 logits by ~0.05
   and flipped borderline argmaxes.
6. **Softmax must subtract the row max** before `exp`, or scores overflow.
7. **Constants are part of the model spec.** Attention scale `1/√d_head` is baked into the
   module constructor; `d`, `eps`, `num_heads` are read from `config.json` at load time.
8. **Inference and training are different programs.** Outside a `Grad()` scope no
   backprop nodes are allocated at all — inference is pure forward spans.
9. **Half the model is memory, not math.** The token lookup table is ~52% of all
   parameters.

## How to run the sample (for the "try it yourself" sections)

```bash
# one-time: download MiniLM weights
hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir samples/data/minilm

# embed a sentence
dotnet run --project samples/NivaraInference -c Release -- minilm
# benchmark (10 passes)
dotnet run --project samples/NivaraInference -c Release -- minilm benchmark
# pairwise cosine-similarity demo
dotnet run --project samples/NivaraInference -c Release -- minilm similarity
# compare embedding against a PyTorch reference (run python first)
python samples/NivaraInference/Python/minilm_compare.py
dotnet run --project samples/NivaraInference -c Release -- minilm compare
```

## Series map

| File | Post | Covers |
|---|---|---|
| `1-the-workhorses-from-words-to-numbers.md` | 1. The Workhorses: From Words to Numbers | Tokenizer, embeddings (lookup tables), LayerNorm, Linear |
| `2-self-attention-the-soft-crossbar-switch.md` | 2. Self-Attention, the Soft Crossbar Switch (concept) | Q, K, V, dot-product relevance, softmax soft-mux, scaling, multi-head |
| `3-self-attention-implemented-the-dirty-details.md` | 3. Self-Attention, Implemented: The Dirty Details | Packing heads, transpose-free QKᵀ, softmax stability, the −∞ padding mask, fused kernel perf |
| `4-the-bypass-wire-the-rectifier-and-what-i-learned.md` | 4. The Bypass Wire, the Rectifier, and Everything I Learned | FFN, GELU vs ReLU bug, residual connections, [CLS]+L2 pooling, retrospective |

## Source files referenced throughout

- Model: `samples/Nivara.Samples/BertModel.cs` (BertConfig, BertSelfAttention, BertLayer, BertEncoder, MiniLMDistilled, MiniLMTokenizer)
- Modules: `src/Nivara/AutoDiff/Nn/{Embedding,Linear,LayerNorm,LayerNormKernel,MultiheadAttention,Activation}.cs`
- Ops: `src/Nivara/AutoDiff/Operations/{ReverseGradOperations,AttentionKernels}.cs`
- Tensors: `src/Nivara/AutoDiff/ReverseGradTensor.cs`, `src/Nivara/Tensors/NivaraTensorExtensions.cs` (GELU kernels)
- Sample entry: `samples/NivaraInference/Program.cs`, `samples/NivaraInference/README.md`
