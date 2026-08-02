# Post 1 — The Workhorses: From Words to Numbers

> Series: *The Signal Chain: A Software Engineer's Field Guide to a BERT*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"The Workhorses: From Words to Numbers"**
>
> What you'll learn: the four blocks that do 90% of a transformer's work and hold almost
> all of its parameters — the **lexer** (WordPiece tokenizer), the **lookup table**
> (embeddings), the **gain-staging stage** (LayerNorm), and the **threshold-logic vote**
> (Linear). By the end the reader can trace the front half of MiniLM's data flow
> with shapes in hand.

## Hook (use as-is or rewrite)

> Everyone wants to hear about attention — it has the catchy diagrams. But if you open up
> a real sentence-embedding model, the first ~90% of the runtime and more than half of the
> parameters are boring-looking blocks: a *lexer*, a *lookup table*, a *voting machine*, and
> a *gain-staging stage*. These are the AND gates and comparators of the whole machine. And
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

- **Introduce the two-part pattern** used for every block (series convention, README):
  first **the math, gently** — the equation if it's simpler than code, the code if it's
  simpler, plain words otherwise — then **how MiniLM uses it** — which layer, what it
  contributes, what it costs. By the last post the reader has every piece assembled.

- **The running example** (see series README): "This is a cat." vs "This is a dog." →
  two 384-d vectors → high cosine similarity; vs "I love programming." → low similarity.
- Set expectations: this post covers the front half (lexer → embeddings → LayerNorm →
  Linear). Posts 2–3 cover attention, Post 4 the FFN + residuals + the retrospective.

## 2. The lexer: WordPiece tokenization

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| text | `[128]` token ids | 0 | split into subword tokens | the lexer (compiler front-end) |

**1. The math, gently.** There is no arithmetic in this block — it converts characters to
integers. That makes it the easiest node in the series and a good template for the rule we
use everywhere: *if the code is simpler than the equation, show the code; if a concept is
simpler still, say it in words.* The whole "math" of tokenization is two dictionary
lookups:

```csharp
tokens[i] = vocab[wordpiece(text)];   // e.g. "tokenization" → "token" + "ization"
```

The data is crunched, but the crunching is bookkeeping: one finite vocabulary
(`vocab.txt`), one integer per token.

**2. How MiniLM uses it.** This is node #1 in the chain — text goes in, a `[128]` vector of
token ids comes out. MiniLM's tokenizer is WordPiece (`MiniLMTokenizer.Load/Encode`,
`samples/Nivara.Samples/BertModel.cs:437-463`):

- `[CLS]` and `[SEP]` are added around the sentence (`addSpecialTokens: true`,
  `BertModel.cs:449`). Standard BERT ids: `[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102`,
  `[MASK]=103` — the program prints the first 10 input token ids so you can verify
  (`samples/NivaraInference/Program.cs:653-657`).
- Sequences are padded to `maxLen = 128` with the padding token (`BertModel.cs:459-461`),
  and an attention mask `[0,1]` marks real vs padded positions (attention reads it in
  post 3).
- **Why subwords:** a finite vocabulary can't cover all of English, so unknown words are
  *split* into known pieces ("tokenization" → "token" + "ization"). The model never hits
  "unrecognized token." Compiler front-end framing: `[CLS]` is BOF, `[SEP]` is EOF,
  `[PAD]` is padding whitespace.

The piece this node contributes: the integers every other block indexes. Everything after
this is float math on those integers.

## 3. The lookup table: token embeddings

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128]` token ids | `[128, 384]` | **11,720,448 (~52%)** | `out[i] = table[token_id[i]]` | ROM / `dict` / database index |

**1. The math, gently.** The code is simpler than any equation — this is a *read*, not a
computation. Each token id picks one row out of a big matrix:

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
```

`Gather` is row lookup on axis 0 (`ReverseGradOperations.cs:1834`). Think `dict[tokenId]` —
an array indexed by an integer — or, in hardware terms, a ROM / content-addressable table.
The entire "math" of this node is `out[i] = table[token_id[i]]`.

**2. How MiniLM uses it.** The table is `[vocab_size=30522, hidden=384]`
(`Embedding<T>` constructor, `src/Nivara/AutoDiff/Nn/Embedding.cs:18-33`; config in
`BertConfig`, `BertModel.cs:8-16`) — and it is **11,720,448 of the ~22.6M parameters:
more than half the model.** The rows became meaningful *during training*; at inference
this is pure read-only memory. This is the piece that turns integers into vectors, and it
is where "most AI parameters aren't math, they're memory."

> Same node family, two more times: the position table (next section) and the segment
> table (`[2,384]`, 768 params) — three lookups summed into one `[128,384]` matrix.
- The learning that made the rows meaningful happened *during training*; at inference the
  layer is pure read-only memory.

## 4. Why order matters: position embeddings

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128]` (indices 0..127) | `[128, 384]` | 196,608 | `pos_table[position]` | a counter feeding a table |

**1. The math, gently.** Zero new math — this is the previous node (a table lookup) plus an
element-wise add, the first time the series mixes signals:

```csharp
var posIds = new T[seqLen];
for (int i = 0; i < seqLen; i++) posIds[i] = T.CreateChecked(i);
var posEmbInput = ReverseGradTensor<T>.FromArray(posIds, requiresGrad: false);
posEmbInput.Reshape(seqLen);

var wordEmb = wordEmbed.Forward(input);
var posEmb  = posEmbed.Forward(posEmbInput);
var hidden  = ReverseGradOperations.Add(wordEmb, posEmb);   // ← a wire sum
```

Position ids are literally `0, 1, 2, ...` per sequence, generated at forward time
(`BertEncoder.Forward`, `BertModel.cs:258-272`). "Adding a position vector" is adding 384
numbers, position by position.

**2. How MiniLM uses it.** The lookup table gives identical vectors to identical tokens no
matter where they appear — "bank" in "river bank" and "bank loan" get the same row. The
position table is what lets the model distinguish word #1 from word #2. Analogy: a state
machine needs a clock/counter to know where it is; the position table is the counter.

- Table length `max_position_embeddings = 512` (`BertConfig`, `BertModel.cs:14`);
  `[512, 384]` = 196,608 params.
- The three lookups — token, position, segment — are **summed element-wise**
  (`ReverseGradOperations.Add`): no averaging, no concatenation, just a wire sum. Good
  spot for a "signals get mixed here" diagram: three `[128,384]` signals → one `[128,384]`.
- DistilBERT skips the segment table (`includeTokenTypeEmbedding: false`, `BertModel.cs:15`).

Piece this node contributes: word identity (table) + word order (table) merged into one
vector per token — and the first example of the model "mixing" signals, the reason
LayerNorm sits right behind it.

## 5. LayerNorm: the gain-staging stage

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128, 384]` | `[128, 384]` | 768 (γ+β) | normalize each row, then rescale | recording-console levels / gain staging |

**1. The math, gently.** Three plain-English steps, then the one equation this post needs
(the equation is genuinely simpler than the code here, so we use it):

1. subtract the row's **average** — remove the DC offset / the baseline level,
2. divide by the row's **spread** (the standard deviation) — scale to a healthy loudness,
3. multiply by **γ**, add **β** — the gain and offset knobs the model learned.

```
μ    = mean over the row
σ²   = variance over the row
y[i] = (x[i] − μ) / sqrt(σ² + ε) · γ[i] + β[i]
```

"Variance" is just "how far the numbers spread from the mean" — no more. The kernel is the
same three steps in code, `src/Nivara/AutoDiff/Nn/LayerNormKernel.cs:45-80` (two passes:
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

**2. How MiniLM uses it.** Gain staging, per token: normalize each row over its 384
features, then re-apply the learned knobs. This is the recording-console node.

- **Recording-console analogy:** every track's level meter has a healthy zone — green, not
  peaking into the red. A signal in the red **clips**; a signal in the weeds gets buried in
  noise. Tokens arrive at wildly different volumes, so before routing further you normalize.
  LayerNorm subtracts the **DC offset** (mean), scales by **loudness** (std), then re-applies
  **γ** and **β** — the per-band sliders on an equalizer you leave in a fixed position
  (γ is per-dimension, so per-band gain; β is per-band trim). γ starts at **1**, β at **0**
  (`LayerNorm.cs:37-41`) — the model "learns" the knob positions during training.
- **Why manage each signal's level individually? Because a signal is a mix of frequencies.**
  A token's 384 values are 384 components stacked into one waveform. Each band's LED meter
  can look fine on its own — but when the bands align, the *combined* peak is far hotter
  than any single one. The overall meter is what clips.
  Zoom out: this is a whole *mix*. The model constantly adds signals together — residual
  connections literally sum two `[128,384]` matrices (post 4), attention blends every token
  into every other (post 2), the FFN amplifies (post 4). If every channel arrives hot, the
  sum drives the bus into the red even though no single channel did. That's why you re-level
  *every* signal *at every stage* — so the mix never collectively overloads.
- **Headroom payoff:** downstream stages — the amplifiers and the rectifier — clip on hot
  signals. Leave headroom and nothing distorts; that's gain staging, the same discipline a
  recording engineer applies before every stage.
- **Where it sits:** right after the embedding sum, and then inside every BERT layer after
  each mixer — **13 total** (1 here + 2 in each of the 6 layers). This is also why it sits
  after every mixer: each mix is new collective clipping risk.
- **ε = "don't compute gain on silence"** — the tiny floor so you never divide by zero.
  `LayerNorm.Forward` validates rank ≥ 2 and the last-dim match (`LayerNorm.cs:48-53`).

**The ε gotcha (what I learned):** the BERT config ships `layer_norm_eps = 1e-12`
(`BertConfig.LayerNormEps`, `BertModel.cs:16`), which is *not* PyTorch's LayerNorm default
of `1e-5`. Copying the wrong ε degrades outputs silently — no error, no warning. The model
spec *includes* its constants.

## 6. Linear: the threshold-logic vote

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[L, 384]` | `[L, 384]` | 147,840 (384×384 + 384 bias) | `y = x·Wᵀ + b` | threshold-logic vote / MAC array |

**1. The math, gently.** A dot product — and it's the one node where the code loop and the
equation are equally short, so here's both, in the order engineers read them. Code:

```
for each row r, for each output channel j:
    y[r, j] = (Σ over i)  x[r, i] · W[j, i]   +  b[j]
```

i.e. `y = x·Wᵀ + b`: multiply each input value by a weight, add them all up, add one bias.
That is the entire node — every output number is one "weighted vote tally." Weight is stored
`[out=384, in=384]` row-major (`Linear.cs:36-39`); each row of W is one threshold gate's
weight list, and the output row is every gate's tally against the input row.

**2. How MiniLM uses it.** The workhorse node. In the chain itself it's not part of the
front half (that's lookups + a normalize) — instead it's the block every later stage is
built from, the ones posts 2–4 cover:

- attention projections **Q, K, V** (post 2),
- the FFN's two stages **384→1536** and **1536→384** (post 4),
- the classification head (post 4).

One instance is `384×384 + 384 bias = 147,840` params; MiniLM runs it **4×** inside each
attention (post 2) and **2×** inside each FFN (post 4). `Linear.Forward`,
`src/Nivara/AutoDiff/Nn/Linear.cs:55-68`:

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

Bias is a row-broadcast add (`AddBias`, `ReverseGradOperations.cs:49-102`); the shape
contract is `[aRows,aCols] × [bRows,bCols]` with `aCols == bRows`
(`ReverseGradOperations.cs:250-253`).

The threshold-vote picture:

- **Threshold-logic vote (the primary analogy):** a Linear layer is a row of threshold
  gates. Each input `x[i]` casts a weighted vote (`x[i]·W[j,i]`); the output `y[j]` is the
  running tally. The bias `b[j]` is the **comparator threshold offset** — the "bar for
  winning": the tally must clear it for the gate to fire. The whole layer is 384 gates side
  by side; the whole model is a matrix of them. This is the "the AND gate does this to the
  signals" view of a neural layer — threshold gates are a real logic-design element (the
  McCulloch–Pitts neuron).
- **Hardware truth (one line):** every vote is a multiply–accumulate, so a Linear layer is
  structurally a **systolic MAC array** — the same fabric as a TPU. A miniature TPU per
  layer.
- **Software-native aside:** if inputs and weights were bits, the vote is just
  `popcount(x AND w)` — counting bitwise agreements. The float dot product is the "soft"
  version of the same idea.

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

---

## Checkpoint: the encoder map

> Step back from the details — here's where we are on the whole line.

```
text ──→ ●tokenizer ──→ token ids [128]
      │  ●token + position + segment lookups, summed ──→ [128×384]
   ×6 │  each BERT layer:
      │    ○[attention] ──→ ○⊕ (residual) ──→ ○[LayerNorm] ──→ ○[widen+rectify+squeeze] ──→ ○⊕ (residual) ──→ ○[LayerNorm]
      ▼
      ○[CLS] row ──→ ○L2 normalize ──→ unit vector [384] ──→ cosine similarity (cat vs dog)

● covered so far · ○ still ahead
```

- **Added:** the front half — lexer, the three lookup tables, the gain-staging stage, and
  the threshold-logic vote (the workhorse that returns inside the still-unlit blocks).
- **Where you are:** text is now a `[128×384]` matrix of steady, leveled signals.
- **Still unlit:** everything inside the 6 stacked layers — attention, the bypass wire and
  gain stages around it, the widen/rectify/squeeze stage — plus the readout that turns it
  all into the cat-vs-dog cosine. Posts 2–4 light those.

## End hook → Post 2

> Next: the block that made ChatGPT possible — the soft crossbar switch. Q, K, V, three
> probes on the same signal, and a soft mux that routes it. Three line-equations, four
> named pieces, zero calculus.

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
- The threshold-gate picture: three inputs voting into an adder, the comparator sitting at
  the top setting the bar (bias).
- The recording-console visual: per-band LED meters all green, the combined peak meter in
  the red — "individually fine, collectively clipping."
- GELU-vs-ReLU curve is deferred to Post 4.

---

> **If you remember one thing from this post:** half the model is a lookup table —
> embeddings are memory, not math.
