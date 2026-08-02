# Post 2 — Self-Attention, the Soft Crossbar Switch (concept)

> Series: *The Signal Chain: A Software Engineer's Field Guide to a BERT*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"Self-Attention, the Soft Crossbar Switch"**
>
> What you'll learn: the conceptual core of a transformer with exactly three equations
> (both are just dot products and a normalization), four named pieces — **Q, K, V, and the
> softmax** — and why multi-head means running 12 of these in parallel. No calculus, no
> backprop. The implementation dirty details are Post 3.

## Hook (use as-is or rewrite)

> The layer that made ChatGPT possible is, to a systems engineer, an old friend: a
> **crossbar switch** whose routing is decided by the data itself. Every input line can
> listen to every other input line, with the "amount of listening" computed on the fly.
> No magic, no understanding — just a weighted readout of a memory. Let me show you.

## 1. The problem fixed wiring can't solve

- Words need different things from each other depending on context. "The **bank** was
  muddy" vs "the **bank** approved the loan" — "bank" is the same token, same embedding row,
  but must *route* information from different companions (river vs money) to be useful.
- Fixed connections (like an AND-gate truth table) can't do this: the wiring would have to
  be different for every sentence ever written. Attention makes the wiring **data-dependent**
  — computed at runtime from the actual inputs.
- The shape intuition: input is `[128, 384]` (128 tokens, 384 features). After one
  attention layer it's still `[128, 384]`, but each of the 128 rows is now a *weighted blend
  of the other 127 rows*. The "context" everyone talks about.

## 2. The cast: Q, K, V — three probes on the same signal

From each token's embedding row, three **Linear** projections are computed (post 1's
amplifier, applied three times). Source: `BertSelfAttention.Forward`,
`samples/Nivara.Samples/BertModel.cs:65-71`:

```csharp
public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
{
    var Q = qProj.Forward(input);
    var K = kProj.Forward(input);
    var V = vProj.Forward(input);
    return oProj.Forward(MultiHeadAttention(Q, K, V, null));
}
```

| Letter | Name | Job in plain English | Analog in a memory subsystem |
|---|---|---|---|
| **Q** | query | "what am I looking for?" | the read address / the search key |
| **K** | key | "what do I advertise / contain?" | the stored address tag |
| **V** | value | "what do I actually carry?" | the stored content |

- All three are `[128, 384]` — the same rows as the input, re-mixed by three different
  Linear weight matrices. The model *learned* those three re-mixings during training.
- Analogy bank:
  - **Content-addressable memory (CAM):** Q is the probe, K is what each memory slot
    answers with, V is the payload returned.
  - **Database query:** Q = `SELECT`/WHERE, K = indexed column, V = the row you return.
  - **A meeting:** each person (token) broadcasts "who can tell me X?" (Q), others reply
    "I know about X" (K), and the ones who do hand over what they have (V).

## 3. The relevance matrix: dot(Q, K) says "listen"

- For every pair (query token *i*, key token *j*), compute how much *i* should listen to *j*:

```
score(i, j) = dot( Q[i], K[j] ) =  Σₖ Q[i,k] · K[j,k]
```

- This is the **same dot product / correlation** from Post 1's Linear — "how aligned are
  these two signal vectors?" Two tokens whose Q and K point in similar directions get a
  high score; orthogonal ones get ~0.
- Arranged as a matrix, this is `S = Q·Kᵀ`, shape `[128, 128]` — "who listens to whom."
  Row *i* holds token *i*'s opinion of every other token.
- MiniLM numbers: `[128,384] × [128,384]ᵀ → [128,128]`, i.e. 128×128 = 16,384 relevance
  scores, computed fresh on every forward pass.

## 4. Softmax = the soft mux

- A relevance score is a *score*, not a weight. We need each row to sum to 1 so the final
  output is a proper weighted blend. Normalize each row:

```
P[i,j] = exp( S[i,j] ) /  Σⱼ exp( S[i,j] )        (one softmax per row)
```

- **The hard switch vs the soft mux (the core analogy):**
  - `argmax` = a hard multiplexer: exactly one input selected, all others dropped
    (binary, AND-gate-like).
  - `softmax` = a *proportional mixer*: every input contributes, scaled by its weight,
    and the weights sum to 1. It's argmax "opened up" into a continuous, differentiable
    form. Same shape of answer (dominant winner), but smooth.
  - Why differentiability matters (one line, no calculus): the whole model is trained by
    gradient descent, and gradients need smooth functions; a hard `if`-style switch has no
    useful slope to follow. Softmax is the "the AND gate with a slope."
- The exponential is what exaggerates differences: double the logit → 7.4× the weight.
  Max-scoring token dominates, but the runner-ups still get a vote.
- Padding note for later: if a score is `-∞`, `exp(-∞) = 0` → that token gets **zero**
  weight no matter what. This one fact is the entire trick behind masking (Post 3).

## 5. The readout: blend the values

```
O[i] = Σⱼ P[i,j] · V[j]        →   O = P·V, shape [128, 384]
```

- Token *i*'s new vector is a weighted sum of *every* token's value vector, weights from
  softmax row *i*. This is the "soft mux output": a blend, not a pick.
- Analogy: everyone at the meeting ends up with a notebook containing a blend of the
  information other people had, weighted by how relevant they were to them.
- **This is why attention is sometimes called "soft lookup":** a hard lookup would return
  `V[argmax]` (one row); soft attention returns a convex blend of all rows.

## 6. Why scale by 1/√d (gain compensation)

- Dot products grow with dimension: 32 numbers summed → scores in a wider range than with
  d=1. Big scores push softmax toward a near-one-hot "hard switch" (saturated), where the
  gradient ≈ 0 — training stalls, and inference output becomes brittle.
- Fix: divide every score by `√(head_dim)` before softmax — keep the soft mux operating in
  its linear-ish region. Analogy: an amplifier stage with gain compensation so it never
  saturates the rails.

```
S = (Q · Kᵀ) / √d      d = head dimension (32 in MiniLM)
P = softmax(S, along rows)
O = P · V
```

- In the code, the scale is a constant baked into the module at construction:
  `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs:40` —
  `_attnScale = T.CreateChecked(1.0 / Math.Sqrt(_headDim));`
  and again in `BertSelfAttention` (`BertModel.cs:61`) for BERT-family models.
  **Lesson: the scale is part of the model's spec, not a free parameter.**

## 7. Multi-head: 12 parallel signal paths

- One attention decision per token is coarse. Instead: split the 384 features into 12
  chunks of **32** and run the whole Q/K/V pipeline 12 times in parallel — 12 "heads."
- Head *h* only sees features `[h·32, (h+1)·32)`. Each head learns a different notion of
  relevance (one might track "what noun does this adjective modify," another "what's the
  next word," etc. — you don't get to choose; training finds them).
- Outputs of all 12 heads are concatenated back to `[128, 384]` and passed through one more
  Linear (`oProj`) that mixes the 12 opinions into the final `[128, 384]`.
- Analogy: 12 crossbar switches wired in parallel, each carrying its own subset of the
  signal, followed by a combining mixer. MiniLM numbers: `head_dim = 384/12 = 32`.

## 8. The whole block in one picture

```
input x [128, 384]
   │ Linear(Wq)   Linear(Wk)   Linear(Wv)
   ▼      ▼           ▼           ▼
   Q [128,384]  K [128,384]  V [128,384]
        \          /
         \        /
       S = Q·Kᵀ / √d        [128,128]   "who listens to whom"
          │
       P = softmax(S)        [128,128]   rows sum to 1  (the soft mux)
          │
       O = P · V             [128,384]   "weighted blend of values"
          │
       Linear(oProj)          [128,384]
          ▼
       attn output [128, 384]

Attn(x) = oProj( softmax( QKᵀ / √d ) · V )
```

That's the entire "hard part." Three line-equations, four named pieces, zero calculus.

## 9. End hook → Post 3

> That's the concept. Shipping it is where the engineer's fun begins: heads aren't separate
> matrices, sentences must be padded to a fixed length and the padding must not vote, exp
> overflows if you're not careful, and every transpose you avoid is a cache miss you didn't
> have to take. Next: the dirty details of making this run.

## Facts & numbers to reuse (checklist)

- Q, K, V = three Linear projections of the same `[128,384]` input (`BertModel.cs:65-71`).
- `S = Q·Kᵀ / √d`, `P = softmax(S)` per row, `O = P·V`; shapes `[128,384]×[128,384]ᵀ→[128,128]→[128,384]`.
- head_dim = 384/12 = 32; scale `1/√32` baked into the module (`MultiheadAttention.cs:40`).
- Multi-head = 12 parallel paths + concat + `oProj` mixing (`BertModel.cs:45-71`).
- The -∞/exp(-∞)=0 fact is the seed of the padding mask (Post 3).
- 16,384 relevance scores per layer per forward (128×128).

## Source references

- `samples/Nivara.Samples/BertModel.cs` — BertSelfAttention (45-157), scale (61)
- `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs` — constructor & scale (27-58), Forward (60-116)
- `src/Nivara/AutoDiff/Nn/Linear.cs` — the projections are Linear layers (post 1)
- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — MultiHeadAttention entry (362-521)
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` — SoftmaxRows (53-88)

## Visual ideas

- The one-picture block diagram (section 8) — this should be the hero graphic.
- A tiny worked example with 3 tokens and d=2 so all numbers fit: Q/K/V 2-dim, compute the
  3×3 score matrix, softmax row, blend. Worked by hand in the post.
- Q/K/V "meeting" cartoon.
