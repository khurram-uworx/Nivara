# Post 4 — The Bypass Wire, the Rectifier, and Everything I Learned

> Series: *The Signal Chain: A Software Engineer's Field Guide to a BERT*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"The Bypass Wire, the Rectifier, and Everything I Learned"**
>
> What you'll learn: the two remaining blocks are the most engineer-flavored of all — the
> **bypass wire** (residual connection) and the **soft rectifier** (GELU) — plus the
> readout ([CLS] pooling + L2 normalization) that closes the loop on the series' running
> example, and a retrospective of what building the whole thing actually taught me.

## Hook (use as-is or rewrite)

> The last two blocks in the chain are the most engineer-flavored parts of the whole model:
> a wire that bypasses each stage, and a diode with a soft knee. They're also where the two
> most instructive bugs I hit lived — one was a choice between three look-alike
> nonlinearities, and it only showed up as a drift in the fifth decimal place of a
> sentiment score.

## 1. The FFN: the per-token "thinking" pass

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128, 384]` | `[128, 384]` | 1,181,568 | widen to 1536 → nonlinearity → squeeze back | two-stage amplifier with a rectifier |

- The attention stage routes information *between* tokens; the FFN transforms each token
  *independently* — no cross-token mixing, each of the 128 rows goes through the same
  two Linear layers (`BertLayer.Forward`, `samples/Nivara.Samples/BertModel.cs:177-189`):

```csharp
public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
{
    var h = attn.Forward(input);                  // attention (posts 2-3)
    h = ReverseGradOperations.Add(h, input);      // residual 1: the bypass wire
    h = ln1.Forward(h);                           // LayerNorm (post 1)

    var h2 = fc1.Forward(h);                      // Linear 384 → 1536  (widen)
    h2 = ReverseGradOperations.GeluExact(h2);     // the soft rectifier
    h2 = fc2.Forward(h2);                         // Linear 1536 → 384  (squeeze)
    h2 = ReverseGradOperations.Add(h2, h);        // residual 2
    h2 = ln2.Forward(h2);                         // LayerNorm
    return h2;
}
```

- `fc1` is `Linear(384, 1536)` and `fc2` is `Linear(1536, 384)` (`BertLayer` constructor,
  `BertModel.cs:159-175`). The intermediate size `1536` is 4× the hidden size
  (`BertConfig.IntermediateSize`, `BertModel.cs:13`).
- **Analogy:** expand the signal into a wider scratch register (more "headroom" to form
  intermediate features), apply a nonlinearity, then fold it back to the bus width. It's a
  two-stage amplifier where the rectifier sits between the stages.
- **Why widen to 1536, then squeeze back to 384?** Because a wider lane is where the model
  forms *combinations*. Each of the 384 input values is already a blend of meanings; to
  detect a *new* feature the model must weigh many inputs at once, and it needs separate
  lanes so different combinations don't trample each other. So it temporarily expands to a
  1536-wide workbench — each lane can specialize in one combination — the rectifier keeps
  each lane's answer from canceling, and the second Linear folds the useful lanes back into
  the 384-wide bus the next layer expects. The "4×" is not a law; it's a configuration
  choice, like a buffer size (`BertConfig.IntermediateSize`, `BertModel.cs:13`). Wide enough
  to form rich combinations, cheap enough to keep the math fast.
- Param note: the FFN (~1.18M per layer) is roughly twice the attention stack (~0.59M) —
  the "boring" layers are the expensive ones.
- (Why a rectifier at all? Because without one the whole stack collapses into a single
  Linear — the next section.)

## 2. GELU: the soft rectifier

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[384]` (or any) | same shape | 0 | smooth gate on positive values | a diode with a soft knee |

- **Why a rectifier at all?** The question nobody asks, so let's answer it before we meet
  the parts. A `Linear` layer is multiply-and-add; if you chain six of them with *no*
  rectifier, the chain silently collapses into **one** Linear — the matrices fold together
  into a single multiply-add, and six stacked stages buy you exactly nothing over one. The
  rectifier is the one step that isn't linear, and it's what makes a stack of stages able to
  represent things a single stage can't. In code terms: without it, stacking layers is just
  inlining the same function over and over; with it, each stage becomes a fresh decision
  layer that builds on the last.
- **ReLU** (`max(x, 0)`) is a hard diode: negatives → 0, positives pass through. Sharp
  corner at 0. Kernel: `ReverseGradOperations.Relu` at
  `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:879-905`.
- **GELU** is the soft version — it passes `x` scaled by "how confident we are that x is
  positive":

```
GELU(x) = x · Φ(x) = x · ½(1 + erf(x/√2))
```

  Don't let `erf` scare you — treat it as a library call: `x * 0.5 * (1 + erf(x/sqrt2))`
  is `x * SoftGate(x)`, where `SoftGate` scores "how positive is x" on a smooth 0→1 scale
  and you never need to read it. Φ (uppercase phi) is just the name of that soft gate.

  For large positive x, Φ(x)→1 so GELU(x)≈x. For large negative x, Φ(x)→0 so GELU(x)≈0.
  Around 0 it's smooth — small signals pass *partially*, unlike ReLU's hard cutoff.
- Exact-erf kernel (`src/Nivara/Tensors/NivaraTensorExtensions.cs:1366-1388`, float path —
  the formula at line 1386):

```csharp
// vectorized: result[i] = x[i] * 0.5 * (1 + erf(x[i] * 1/sqrt(2)))
var xv = Vector.LoadUnsafe(ref xRef, (nuint)i);
var erfv = ErfVector(xv * invSqrt2);                 // invSqrt2 = 0.70710678…
var onePlus = Vector<float>.One + erfv;
Vector.StoreUnsafe(xv * half * onePlus, ref rRef, (nuint)i);
```

### Three look-alike flavors, and the bug they caused

| Flavor | Formula | Used by |
|---|---|---|
| exact erf GELU | `x · ½(1 + erf(x/√2))` | BERT family — **MiniLM**, DistilBERT (`GeluExact`) |
| tanh-approx GELU | `x · ½(1 + tanh(√(2/π)·(x + 0.044715·x³)))` | GPT family, HF `gelu_new` (`ReverseGradOperations.Gelu`, `:907-946`) |
| ReLU | `max(x, 0)` | hard rectifier (`:879-905`) |

- The curves are visually almost identical and numerically different. **These are specs,
  not stylistic choices.**
- **The bug:** the SST-2 sentiment classifier head applies `pre_classifier (768→768)` →
  activation → `classifier (768→2)`. HuggingFace's architecture uses **ReLU** there; a
  naive port used exact GELU. Result: logits drifted by ~**0.05**, and argmax flipped on
  borderline sentences. Documented at `samples/NivaraInference/README.md:192`; the fixed
  head in `samples/Nivara.Samples/DistilBertModel.cs:27-39`:

```csharp
var encoded = encoder.ForwardBatched(inputIds, attentionMask, batchSize, seqLen);
var clsTokens = ExtractClsTokens(encoded, batchSize, seqLen);
var h = preClassifier.Forward(clsTokens);
h = ReverseGradOperations.Relu(h);            // ← must be ReLU, not GELU
var logits = classifier.Forward(h);
return logits;
```

- **How it was found:** the reference-comparison methodology (section 6) — run the same
  input through the PyTorch reference and the port, compare layer by layer; a per-layer
  diff, not eyeballing final accuracy, pinpointed the head activation.
- BERT-family encoder FFN uses exact erf (`BertModel.cs:184`, `:197`); the head uses ReLU.
  A model can use different flavors in different places — read the source, don't assume.

## 3. The bypass wire: residual connections

**Datasheet:**

| Input | Output | Parameters | One-liner | Analogy |
|---|---|---|---|---|
| `[128, 384]` + `[128, 384]` | `[128, 384]` | 0 | `out = block(x) + x` | bypass / feedback wire |

- Every sub-block is wrapped: `h = Attn(x) + x`, then `h2 = FFN(h) + h`
  (`BertModel.cs:180`, `:186`). The block only ever learns a **delta**; the input always
  passes through untouched.
- Two engineering reasons to tell the audience:

  1. **Signal integrity through depth.** Stack 6 (or 12, or 48) stages and any gain or
     offset drift compounds — like cascaded op-amp stages. The bypass carries the signal
     through, so the network's output never fully depends on the last stage's tiny error.
  2. **Gradient flow.** In reverse-mode training, error signals can route back through the
     bypass (identity derivative = 1), so gradients don't vanish across depth. Same reason
     a feedback path keeps an amplifier stable.
- **The "learn the error, not the answer" idea** is the historical why: residual
  connections are why networks stopped degrading once you stacked more than ~2 layers.
  It's one `Add` op — the cheapest, highest-leverage wire in the model.

## 4. The readout: [CLS] pooling + L2 normalization

- After 6 layers the output is `[128, 384]`. For a *sentence* embedding, the sample takes
  **row 0** — the `[CLS]` token's row (`MiniLMDistilled.Forward`,
  `BertModel.cs:338-343`):

```csharp
public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
{
    var hidden = encoder.Forward(input);
    var clsToken = ExtractRow(hidden, 0, config.HiddenSize);   // row 0 → [384]
    return L2Normalize(clsToken, config.HiddenSize);           // → unit vector
}
```

  In a batched classifier head it's the same idea: `Gather(encoded, [0, seqLen, 2·seqLen, …])`
  (`DistilBertModel.cs:44-50`). Analogy: pin 0 of the bus — the convention is that the
  first token's row carries the sentence summary. Why row 0 specifically? Because during
  training the model was *told* the `[CLS]` position must summarize the whole sentence, so
  it learned to route the useful bits there. A designated output pin — by contract, not by
  magic.

- **L2 normalization** (`BertModel.cs:376-397`): divide every element by the vector's
  magnitude, producing a **unit vector** (the sample prints `L2 norm: ~1.0`,
  `Program.cs:682`).
- Why: magnitude stops mattering, direction survives — so
  `cosine(a,b) = dot(a,b) / (|a|·|b|) = dot(a,b)` for unit vectors. Two sentences are
  "similar" iff their 384-d directions are close. This closes the loop on the running
  example: cat vs dog → high cosine; cat vs programming → low.

---

## For the software-only reader: the audio decoder ring

> The block analogies in this post are real circuits. If you skipped the electronics course,
> here's the same four ideas in Post 1's audio vocabulary and in code — one row per block,
> a single pass that rebuilds your mental model.

| Block | Analog circuit | Audio / digital for SW-only |
|---|---|---|
| FFN | two-stage amplifier + rectifier | **Audio:** split the signal into more EQ bands with extra headroom, drive them, mix back to the 384-wide bus — per-band shaping, not per-token magic. |
| GELU | diode with a soft knee | **Audio:** soft-knee vs hard-knee dynamics — ReLU cuts abruptly at 0 (hard knee), GELU turns on gradually (soft knee). **Code:** `Math.Max(x,0)` vs a smooth clamp. |
| Residual | bypass / feedback wire | **Audio:** true bypass — a wire carries the input straight through; the block only adds a delta. **Code:** `out = block(x) + x`. |
| Readout | pin 0 of the bus | **Audio:** a designated master/summary channel normalized to a reference level. **Code:** `Vector3.Normalize` — direction survives, so cosine == dot. |

## 5. What I actually learned (retrospective — the post's payoff)

1. **The comparison loop is the only way.** I never checked "does it work?" — I checked
   "which block moved the numbers?" Methodology:
   1. Generate a fixed input fixture once (`python samples/NivaraInference/Python/generate_input.py`).
   2. Run the reference (PyTorch) and the port on identical inputs.
   3. Diff **layer by layer** (max abs diff + cosine) — the sample has a `compare_diag`
      mode for vision models and `compare` modes for MiniLM/DistilBERT.
   4. Bisect to the first layer whose diff exceeds tolerance; fix and re-run.
   This is how the GELU/ReLU bug (logit drift ~0.05) and the ε issue surfaced. Never
   "close enough" — always "which block moved the numbers."
2. **Numerical correctness is a spec, not a luxury.** erf vs tanh GELU, softmax
   max-subtraction (Post 3), `layer_norm_eps` `1e-12` vs `1e-5` (Post 1) — every one is a
   "which mathematically-equivalent expression do I ship?" decision, and each one is part
   of the model contract. Verification numbers to cite: DistilBERT `last_hidden_state`
   matches HuggingFace to `max abs diff 5e-6` (cosine 0.99999988); SST-2 logits to
   `max abs diff 9.5e-7` with `argmax agreement 8/8` (`README.md:180`, `:196`).
3. **Shapes as a type system** (recap of Post 1) — the single discipline that made a
   22.6M-parameter model assemblable from a flat tensor dump with zero reflection.
4. **Inference and training are different programs.** No graph nodes outside a `Grad()`
   scope; the fused attention kernel allocates `savedWeights` only when tracking gradients
   (Post 3). This is a design decision, not an optimization afterthought.
5. **Honest performance.** MiniLM: PyTorch 11 ms vs Nivara 110 ms; DistilBERT 31 ms vs
   186 ms on the same CPU (`README.md:218-230`). Publish it. The gap is the matmul
   kernels (naive loops vs MKL), which is a known, bounded engineering problem — not a
   flaw in the architecture or the concepts being explained.
6. **The thesis.** None of these blocks are exotic: lookup table, threshold-logic vote,
   gain-staging stage, soft switch, bypass wire, rectifier. If you can trace shapes and
   read a datasheet, you can read a transformer.

---

## Checkpoint: the encoder map (the whole line, lit)

> Step back from the details — here's where we are on the whole line.

```
text ──→ ●tokenizer ──→ token ids [128]
      │  ●token + position + segment lookups, summed ──→ [128×384]
   ×6 │  each BERT layer:
      │    ●[attention] ──→ ●⊕ (residual) ──→ ●[LayerNorm] ──→ ●[widen+rectify+squeeze] ──→ ●⊕ (residual) ──→ ●[LayerNorm]
      ▼
      ●[CLS] row ──→ ●L2 normalize ──→ unit vector [384] ──→ cosine similarity (cat vs dog)

● covered so far · ○ still ahead
```

Every station is lit. Nothing left to dim: text in, a unit vector out, and the series'
running example lands on one cosine — cat vs dog high, cat vs programming low.

## Facts & numbers to reuse (checklist)

- FFN per layer: `fc1(384→1536)` → GELU → `fc2(1536→384)`; 1,181,568 params
  (≈2× the attention stack's 591,360).
- Why the rectifier: chained Linears with no nonlinearity collapse into one Linear; the
  rectifier is what makes stacking stages meaningful.
- Why widen→squeeze: a 1536-wide workbench lets lanes form different feature combos, then
  fold the useful ones back to the 384-wide bus; 4× is a config choice, not a law.
- GELU exact: `x·½(1+erf(x/√2))`; tanh approx: `x·½(1+tanh(√(2/π)·(x+0.044715x³)))`; ReLU: `max(x,0)`.
- MiniLM + DistilBERT encoder use exact GELU; the SST-2 head uses **ReLU**; GPT-style uses tanh-approx.
- GELU/ReLU bug: logits off by ~0.05, argmax flips on borderline cases (`README.md:192`).
- Residuals: `out = block(x) + x` after attention and FFN (`BertModel.cs:180,186`).
- Readout: row 0 = [CLS]; L2-normalize → unit vector; cosine == dot for unit vectors (`BertModel.cs:338-343,376-397`).
- Verification: DistilBERT hidden states max abs diff 5e-6 / cosine 0.99999988; SST-2 logits 9.5e-7, 8/8 argmax (`README.md:180,196`).
- Perf: MiniLM 11 ms vs 110 ms; DistilBERT 31 ms vs 186 ms (CPU, 128 tokens) (`README.md:218-230`).

## Source references

- `samples/Nivara.Samples/BertModel.cs` — BertLayer.Forward (177-203), BertConfig (8-16),
  MiniLMDistilled.Forward (338-343), L2Normalize (376-397)
- `samples/Nivara.Samples/DistilBertModel.cs` — classification head (27-39), ExtractClsTokens (44-50)
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — Relu (879-905), Gelu tanh (907-946), GeluExact (948-976)
- `src/Nivara/Tensors/NivaraTensorExtensions.cs` — GeluExactKernel (1323+), float kernel (1366-1388)
- `samples/NivaraInference/Program.cs` — RunMiniLMInference (632-686), similarity (1177-1231), CosineSimilarity (1233-1245)
- `samples/NivaraInference/README.md` — head activation note (192), verification (180,196), benchmarks (218-230)

## Visual ideas

- GELU vs ReLU curve: two lines, with a vertical shaded band around 0 showing where they
  differ most. (This is the post's hero graphic.)
- The bypass-wire schematic: block diagram where each stage has a wire going around it with
  an adder node.
- The readout flow: `[128,384] → [CLS] row → [384] → L2 normalize → unit sphere dot product`.
- A before/after table of the SST-2 logits with GELU vs ReLU head (illustrative numbers:
  the ~0.05 drift).

## Series wrap-up

The four posts form an arc: workhorses (lookup, vote, gain-staging) → attention concept
(the soft crossbar) → attention implementation (layout, stability, the -∞ mask) → residuals,
rectifier, and the meta-lessons. The running cat/dog example ties them together, and the
retrospective (section 5) is the essay a software engineer can take away: read a model the
way you'd read a schematic — datasheet, signal flow, shapes — and the maths stops being a
wall.

---

> **If you remember one thing from this post:** a stack of Linears with no rectifier
> collapses into a single Linear — the nonlinearity is what makes depth mean something.
