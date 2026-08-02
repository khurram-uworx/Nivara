# Post 1 — The Workhorses: From Words to Numbers

> Series: *The Signal Chain: A Software Engineer's Field Guide to a BERT*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"The Workhorses: From Words to Numbers"**
>
> What you'll learn: the four blocks that do 90% of a transformer's work and hold almost
> all of its parameters — the **lexer** (WordPiece tokenizer), the **lookup table**
> (embeddings), the **automatic gain control** (LayerNorm), and the **multiply–accumulate
> machine** (Linear). By the end the reader can trace the front half of MiniLM's data flow
> with shapes in hand.

## Hook (use as-is or rewrite)

> Everyone wants to hear about attention — it has the catchy diagrams. But if you open up
> a real sentence-embedding model, the first ~90% of the runtime and more than half of the
> parameters are boring-looking blocks: a *lexer*, a *lookup table*, an *amplifier*, and a
> *gain-control stage*. These are the AND gates and op-amps of the whole machine. And
> here's the thing — they're the blocks you actually end up debugging.

## 1. The whole chain on one page

Introduce the block diagram (this becomes the map for the entire series):

```
"the cat sat"
   │  WordPiece tokenizer (the lexer)
   ▼
[128] token ids          [101, 2361, 2892, 2893, 102, 0, 0, ...]
   │  token + position (+ segment) embeddings, summed → LayerNorm
   ▼
[128, 384]
   │  ×6 BERT layers (posts 2–4)
   ▼
[128, 384]
   │  take row 0 ([CLS]) → L2 normalize
   ▼
[384] unit vector  ←  cosine similarity lives here
```

Talking points:

- **Everything is a signal.** Every layer's input and output is a 1-D array of `float`s
  (a row-major matrix). A layer is a pure function with a defined contract: shapes in,
  shapes out.
- **Introduce the datasheet convention** used in every post. A layer datasheet looks like:

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| text | `[128]` ints | 0 | split into subword IDs | the lexer |

- **The running example** (see series README): "This is a cat." vs "This is a dog." →
  two 384-d vectors → high cosine similarity; vs "I love programming." → low similarity.
- Set expectations: this post covers the front half (lexer → embeddings → LayerNorm →
  Linear). Posts 2–3 cover attention, Post 4 the FFN + residuals + the retrospective.

## 2. The lexer: WordPiece tokenization

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| text | `[128]` token ids | 0 | split into subword tokens | the lexer (compiler front-end) |

Facts & evidence:

- The model uses a WordPiece tokenizer. In the sample it's
  `Microsoft.ML.Tokenizers.BertTokenizer.Create(vocabPath)` loading `vocab.txt`
  (`samples/Nivara.Samples/BertModel.cs:437-463`, `MiniLMTokenizer.Load/Encode`).
- `[CLS]` and `[SEP]` are added around the sentence (`addSpecialTokens: true`,
  `BertModel.cs:449`). Standard BERT WordPiece IDs: `[PAD]=0`, `[UNK]=100`, `[CLS]=101`,
  `[SEP]=102`, `[MASK]=103`. Verify against the sample's own printed output before quoting
  exact ids — the program prints the first 10 input token ids
  (`samples/NivaraInference/Program.cs:653-657`).
- Sequences are padded to a fixed `maxLen = 128` with the padding token
  (`BertModel.cs:459-461`), and an attention mask `[0,1]` marks real vs padded positions.
- **Why subwords (the engineering payoff):** a finite vocabulary can't cover all of
  English, so unknown words are *split* into known pieces ("tokenization" → "token" +
  "ization"). Analogy: a lexer that never hits "unrecognized token" — unknown input
  decomposes into known lexemes instead of failing. This is why OOV words don't crash
  the model.

> **Compiler analogy to lean on:** the tokenizer is the front-end of a compiler — it turns
> a character stream into a token stream against a fixed vocabulary ("the language spec").
> `[CLS]` is BOF, `[SEP]` is EOF, `[PAD]` is padding whitespace.

## 3. The lookup table: token embeddings

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128]` token ids | `[128, 384]` | **11,720,448 (~52%)** | `out[i] = table[token_id[i]]` | ROM / `dict` / database index |

Facts & evidence:

- The embedding is a matrix `[vocab_size=30522, hidden=384]` stored as a parameter
  (`Embedding<T>` constructor, `src/Nivara/AutoDiff/Nn/Embedding.cs:18-33`; config in
  `BertConfig`, `BertModel.cs:8-16`).
- The forward pass is a **read, not a computation**: it gathers rows by index.
  `Embedding.Forward` validates ids then calls `ReverseGradOperations.Gather(weight, ids)`
  (`Embedding.cs:35-64`):

```csharp
// src/Nivara/AutoDiff/Nn/Embedding.cs:42-63
var tokenIds = new int[totalTokens];
for (int i = 0; i < totalTokens; i++)
    tokenIds[i] = int.CreateChecked(input.Data[i]);

for (int i = 0; i < totalTokens; i++)
{
    if (tokenIds[i] < 0 || tokenIds[i] >= numEmbeddings)
        throw new ArgumentOutOfRangeException(...);
}

var result = ReverseGradOperations.Gather(weight.Tensor, tokenIds);
if (originalShape.Length > 1)
    result.Reshape(originalShape.Append(embeddingDim).ToArray());
return result;
```

- `Gather` itself: `ReverseGradOperations.Gather(source, indices, axis)` at
  `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:1834` (row lookup, axis 0).
- **Parameter punchline:** this one table is ~11.7M of the ~22.6M parameters — more than
  half the model. "Most AI parameters aren't math, they're memory."

Analogy options:

- **ROM / content-addressable table:** token id → row of floats, exactly a hardware lookup.
- **`dict[tokenId]`** in code — `Vector3`-ish array indexed by integer.
- The learning that made the rows meaningful happened *during training*; at inference the
  layer is pure read-only memory.

## 4. Why order matters: position embeddings

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128]` (indices 0..127) | `[128, 384]` | 196,608 | `pos_table[position]` | a counter feeding a table |

Facts & evidence:

- Position ids are literally `0, 1, 2, ...` per sequence, generated at forward time
  (`BertEncoder.Forward`, `BertModel.cs:258-272`):

```csharp
var posIds = new T[seqLen];
for (int i = 0; i < seqLen; i++) posIds[i] = T.CreateChecked(i);
var posEmbInput = ReverseGradTensor<T>.FromArray(posIds, requiresGrad: false);
posEmbInput.Reshape(seqLen);

var wordEmb = wordEmbed.Forward(input);
var posEmb  = posEmbed.Forward(posEmbInput);
var hidden  = _includeTokenTypeEmbedding
    ? ReverseGradOperations.Add(wordEmb, ReverseGradOperations.Add(posEmb, TokenTypeEmb(seqLen)))
    : ReverseGradOperations.Add(wordEmb, posEmb);
hidden = embedLn.Forward(hidden);
```

- Position table length `max_position_embeddings = 512` (`BertConfig`, `BertModel.cs:14`).
- **Why order matters:** the lookup table gives identical vectors to identical tokens no
  matter where they appear — "bank" in "river bank" and "bank loan" get the same row.
  The position table is what lets the model distinguish word #1 from word #2. Analogy: a
  state machine needs a clock/counter to know where it is; the position table is the
  counter.
- Segment (token-type) embedding: `[2, 384]`, 768 params, folded into the same sum for
  models that use it (`BertModel.cs:15`); DistilBERT skips it (`includeTokenTypeEmbedding:
  false`).
- The three embeddings are **summed element-wise** (`ReverseGradOperations.Add`) — no
  averaging, no concatenation, just a wire sum. Good spot for a "signals get mixed here"
  diagram: three `[128,384]` signals → one `[128,384]`.

## 5. LayerNorm: the automatic gain control

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128, 384]` | `[128, 384]` | 768 (γ+β) | normalize each row, then rescale | AGC / signal conditioning |

Facts & evidence:

- Normalizes **per row** (per token), over the 384 features: subtract the mean, divide by
  std, then apply affine gain γ and offset β. Formula (the only equation this post needs):

```
μ    = mean over the row
σ²   = variance over the row
y[i] = (x[i] − μ) / sqrt(σ² + ε) · γ[i] + β[i]
```

- Kernel implementation, `src/Nivara/AutoDiff/Nn/LayerNormKernel.cs:45-80` (two passes:
  row means, then diff → dot → inverse std → scale):

```csharp
// LayerNormKernel.cs:45-52 (pass 1 — row means)
for (int r = 0; r < rows; r++)
{
    int offset = r * normalizedShape;
    var row = input.Slice(offset, normalizedShape);
    T sum = T.CreateChecked(double.CreateChecked(TensorPrimitives.Sum(row)));
    mean[r] = sum / T.CreateChecked(normalizedShape);
}

// LayerNormKernel.cs:66-79 (pass 2 — normalize + affine)
T sumSq = T.CreateChecked(double.CreateChecked(TensorPrimitives.Dot(diffSpan, diffSpan)));
T variance = sumSq / T.CreateChecked(normalizedShape);
invStd[r] = T.One / T.CreateChecked(Math.Sqrt(double.CreateChecked(variance + eps)));
// ...
TensorPrimitives.Multiply(diffSpan, inv, diffSpan);   // (x − μ) / std
TensorPrimitives.Multiply(diffSpan, gamma, outputSlice); // · γ
TensorPrimitives.Add(outputSlice, beta, outputSlice);     // + β
```

- Affine parameters: γ starts at **1**, β starts at **0** (`LayerNorm.cs:37-41`) — the
  model "learns" the gain/offset knobs during training. ε is a tiny constant that exists
  **only** to stop division by zero (and is a silent-accuracy trap, see below).
- `LayerNorm.Forward` validates rank ≥ 2 and that the last dim matches
  (`LayerNorm.cs:48-53`).

Engineering framing:

- **AGC analogy:** every audio chain has an automatic gain controller — remove DC offset,
  level the signal, then apply the gain you actually want. LayerNorm does exactly this per
  token.
- **Why it's needed with 6 stacked stages:** cascading amplifiers accumulate offset and
  gain drift; any bias from stage 1 gets amplified through stage 6. Normalizing at each
  stage keeps every block in a stable operating range so the nonlinearities downstream
  (post 2/4) never saturate or dead-zone.
- **ε = "a tiny series resistance"** so you never divide by zero.

**The ε gotcha (what I learned):** the BERT config ships `layer_norm_eps = 1e-12`
(`BertConfig.LayerNormEps`, `BertModel.cs:16`), which is *not* PyTorch's LayerNorm default
of `1e-5`. Copying the wrong ε degrades outputs silently — no error, no warning. The model
spec *includes* its constants.

## 6. Linear: the multiply–accumulate machine

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[L, 384]` | `[L, 384]` | 147,840 (384×384 + 384 bias) | `y = x·Wᵀ + b` | bank of matched filters / FIR taps |

Facts & evidence:

- The operation used *everywhere* else — attention projections (post 2), the FFN (post 4),
  the classifier head. `Linear.Forward`, `src/Nivara/AutoDiff/Nn/Linear.cs:55-68`:

```csharp
public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
{
    var w = weight.Tensor;
    var output = GradientUtils.IsGradEnabled
        ? ReverseGradOperations.MatMul(input, GetTransposedWeight(w))
        : ReverseGradOperations.MatMulTransposedB(input, w);   // ← inference path

    if (useBias && bias != null)
        output = ReverseGradOperations.AddBias(output, bias.Tensor);

    return output;
}
```

- **The math is a dot product per output channel:**

```
for each row r, for each output channel j:
    y[r, j] = (Σ over i)  x[r, i] · W[j, i]   +  b[j]
```

  i.e. `y = x·Wᵀ + b`. Weight is stored `[out=384, in=384]` row-major
  (`Linear.cs:36-39`); each row of W is a "template" and the output is the correlation of
  the input row against all templates.

- Bias is applied as a row-broadcast add (`AddBias`, `ReverseGradOperations.cs:49-102`).
- `MatMul` shape contract: `[aRows,aCols] × [bRows,bCols]` requires `aCols == bRows`
  (`ReverseGradOperations.cs:250-253`).

Engineering framing:

- **Matched-filter analogy:** each output channel answers "how much does the input look
  like template #j?" Dot product == correlation == how aligned two signal vectors are.
- **MAC unit:** each output float is one sum of products — the multiply–accumulate that
  DSPs have had as an instruction for decades.
- **Bias = offset voltage.**

**What I learned (transpose-free inference):** weights are stored `[out, in]` (the way
PyTorch/HuggingFace serialize them), but matmul wants the second operand transposed. Two
options:
- Build a transposed copy per forward — wasteful in inference, where you may run the model
  thousands of times on the same weights.
- Or teach the kernel a transposed-B mode: `a @ bᵀ` with `b` in raw layout, zero copies.
  Nivara does the second: `MatMulTransposedB` (`ReverseGradOperations.cs:304-343`) is
  inference-only and builds **no** gradient graph; the transposed copy is built once and
  cached only when gradients are enabled (`Linear.GetTransposedWeight`, `Linear.cs:70-82`,
  cache invalidated by `weight.Version`).

## 7. What I learned building these (the retrospective hooks)

1. **Shapes are the type system.** The #1 bug class was shape mismatches. Once I started
   tracing `[L,D] × [D,D] → [L,D]` with the same discipline as type-checking, the model
   built itself. LayerNorm throws on rank < 2 (`LayerNorm.cs:51`), MatMul throws on
   `aCols != bRows` (`ReverseGradOperations.cs:250`) — these guards are the compile-time
   errors you wish every ML API had.
2. **Half the model is a lookup table.** 11.7M of 22.6M params are a read-only memory.
   "The AI is mostly a database with amplifiers bolted on."
3. **The ε trap.** `1e-12` vs `1e-5` — silent. Constants are part of the model spec.
4. **Inference and training are different programs.** The Linear forward literally
   branches on `GradientUtils.IsGradEnabled` and uses different kernels. Outside a `Grad()`
   scope no backprop bookkeeping happens at all.
5. **Why shapes/contracts matter for performance too:** the transposed-weight cache is only
   valid while the weight doesn't change (`weight.Version`), which is a subtle
   invalidation problem — an engineer's version of "is this cache stale?"

## Facts & numbers to reuse (checklist)

- Vocab 30,522; hidden 384; heads 12 (head dim 32); layers 6; FFN 1536; max pos 512; eps 1e-12.
- Token embedding 11,720,448 params (~52% of model); position 196,608; segment 768.
- Linear(384,384) + bias = 147,840 params; used 4× in attention, 2× in FFN, plus head.
- WordPiece adds `[CLS]`/`[SEP]`; pads to 128; mask marks real vs padded.
- LayerNorm: per-token, γ starts 1, β starts 0, ε guards divide-by-zero.
- Linear: `y = x·Wᵀ + b`; inference uses a transpose-free kernel, training caches a transpose.
- Run it: `dotnet run --project samples/NivaraInference -c Release -- minilm`
  prints build time, param count, first 10 token ids, output stats, and L2 norm (~1.0).

## Source references

- `samples/Nivara.Samples/BertModel.cs` — BertConfig (8-16), BertEncoder.Forward (258-278),
  MiniLMTokenizer.Encode (444-463)
- `src/Nivara/AutoDiff/Nn/Embedding.cs` — constructor (18-33), Forward/Gather (35-64)
- `src/Nivara/AutoDiff/Nn/Linear.cs` — Forward two-path (55-68), transposed cache (70-82)
- `src/Nivara/AutoDiff/Nn/LayerNorm.cs` — affine init (37-46), Forward validation (48-53)
- `src/Nivara/AutoDiff/Nn/LayerNormKernel.cs` — mean pass (45-52), normalize pass (66-80)
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — Add (14), AddBias (49),
  MatMul (234), MatMulTransposedB (304-343), Gather (1834)
- `src/Nivara/AutoDiff/Utilities/GradientUtils.cs` — IsGradEnabled / Grad (14-30)
- `samples/NivaraInference/Program.cs` — RunMiniLMInference (632-686), similarity demo (1177-1231)

## Visual ideas

- Block diagram of the front half (lexer → 3 lookups → sum → LN).
- Datasheet cards for the 4 layers.
- A tiny worked dot-product: `x=[1,0,-1]`, one weight row `[0.5, 2, -1]` → `y = 1·0.5+0·2+(−1)·(−1) = 1.5`, then +bias.
- GELU-vs-ReLU curve is deferred to Post 4.
