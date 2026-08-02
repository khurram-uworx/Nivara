# Post 3 — Self-Attention, Implemented: The Dirty Details

> Series: *The Signal Chain: A Software Engineer's Field Guide to a BERT*
> Target: software engineers (0–10 yrs), math-averse. · Suggested title:
> **"Self-Attention, Implemented: The Dirty Details"**
>
> What you'll learn: the concept from Post 2 is four lines of math; shipping it is a stack
> of engineering decisions — real memory layout, packing heads so matmuls get contiguous
> spans, a transpose-free QKᵀ, softmax done without overflowing, and the `-∞` padding mask
> that makes masked attention free. Plus honest performance numbers.

## Hook (use as-is or rewrite)

> The concept was four lines. Shipping it was a list of decisions where everything that
> *could* go subtly wrong *did* — and the fixes were the kind of systems thinking you
> already have: respect the memory layout, don't pay for transposes, keep the exponent from
> overflowing, and hide the "don't care" inputs with a tri-state value. This is the post
> where a transformer starts to feel like a circuit.

> **Where we are:** the same soft crossbar from Post 2, now as a memory-layout and
> numerical-stability problem. Keep the Post 2 picture — pack → QKᵀ → scale → mask →
> softmax → ×V → scatter — in your head; this post is about making that picture fast and
> correct.

## 1. Real shapes & layout: there are no [heads] tensors

- The model never materializes `[batch, heads, seq, d]` cubes. Everything is a flat
  row-major `float[]`. Q/K/V come out of their Linears as `[128, 384]`.
- **Heads are columns, not matrices.** Head *h* owns contiguous columns
  `[h·32, (h+1)·32)` of the `[128, 384]` matrix. In row-major storage, token *r*, head *h*
  starts at element `r·384 + h·32`.
- The stride math that runs the whole layer (`AttentionKernels.GatherHead`,
  `src/Nivara/AutoDiff/Operations/AttentionKernels.cs:21-26`):

```csharp
public static void GatherHead(ReadOnlySpan<T> src, Span<T> dst, int rows, int D, int head, int headDim)
{
    int hs = head * headDim;                       // head's column offset
    for (int r = 0; r < rows; r++)
        src.Slice(r * D + hs, headDim).CopyTo(dst.Slice(r * headDim, headDim));
}
```

## 2. The per-head pipeline (the whole forward)

`ReverseGradOperations.MultiHeadAttention` — signature and validation at
`src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs:362-393`; the hot loop
(`:425-445`):

```csharp
for (int h = 0; h < numHeads; h++)
{
    var qh = qHeads.AsSpan(h * qLen * headDim, qLen * headDim);   // this head's packed Q
    var kh = kHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);
    var vh = vHeads.AsSpan(h * kvLen * headDim, kvLen * headDim);

    TensorsHelper.MultiplyCore(qh, kh, scores, qLen, headDim, kvLen, bTransposed: true); // Qh·Khᵀ
    var scoresSpan = scores.AsSpan(0, scoreLen);
    TensorPrimitives.Multiply(scoresSpan, scale, scoresSpan);     // ÷√d
    if (!maskSpan.IsEmpty)
        TensorPrimitives.Add(scoresSpan, maskSpan, scoresSpan);   // + additive mask (the -∞ trick)

    AttentionKernels<T>.SoftmaxRows(scoresSpan, qLen, kvLen);     // softmax per row

    if (savedWeights != null)
        scoresSpan.CopyTo(savedWeights.AsSpan(h * scoreLen, scoreLen)); // cache P for backward

    TensorsHelper.MultiplyCore(scores, vh, outHead, qLen, kvLen, headDim); // P·Vh
    AttentionKernels<T>.ScatterHead(outHead.AsSpan(0, qLen * headDim), output.AsSpan(), qLen, D, h, headDim);
}
```

Map this back to the Post 2 picture: **pack → QKᵀ → scale → +mask → softmax → ×V → scatter**,
repeated for each of 12 heads. `AttentionKernels.PackHeads` (`:42-46`) gathers all heads
into one contiguous `[12, 128, 32]` block up front; `ScatterHead` (`:32-37`) is the inverse
write-back.

## 3. Pack heads once — "layout is performance"

- Each head's 32 columns are strided in the `[128,384]` matrix (stride 384, not 32). If you
  ran matmuls directly on the strided layout you'd thrash caches and can't use SIMD cleanly.
- Fix: **pack** each head's columns into a contiguous span once, run all kernels on
  contiguous spans, then **scatter** results back. The packing is a cheap gather; the
  matmuls are the expensive part and they get contiguous memory.
- `PackHeads` (`AttentionKernels.cs:42-46`):

```csharp
public static void PackHeads(ReadOnlySpan<T> src, Span<T> dst, int rows, int numHeads, int headDim)
{
    for (int h = 0; h < numHeads; h++)
        GatherHead(src, dst.Slice(h * rows * headDim, rows * headDim), rows, numHeads * headDim, h, headDim);
}
```

- The comment above the class says it all (`AttentionKernels.cs:6-14`): packed once so every
  matmul feeds the SIMD `MultiplyCore` path "with zero per-head transposes (QKᵀ uses the
  transposed-B layout directly)." This is an engineer's cache-locality lesson, not an ML
  lesson.

## 4. Transpose-free QKᵀ

- `Qh·Khᵀ` wants K transposed, but K's raw layout is `[kvLen, headDim]` row-major. Rather
  than transpose K (a full extra pass + allocation), the kernel supports
  `bTransposed: true` — it reads K in transposed order internally.
- Same trick as Post 1's Linear (`MatMulTransposedB`), but inside the hottest loop of the
  model. Source note: `TensorsHelper.MultiplyCore(..., bTransposed: true)`, and the doc
  comment at `ReverseGradOperations.cs:346-355` states the transposed-B layout "equals K's
  raw layout."
- **Lesson:** avoid materializing transposes in a hot path; teach the kernel to read
  transposed instead. One fewer `[kvLen,headDim]` copy per head per layer.

## 5. Softmax done right: subtract the row max

- `exp(x)` overflows float at ~`x > 88`. Attention scores can exceed that.
- The fix is numerically trivial and universally missed in tutorials
  (`AttentionKernels.cs:59-70`, float path):

```csharp
static void SoftmaxRow(Span<T> row)
{
    T max = T.NegativeInfinity;
    for (int i = 0; i < row.Length; i++)
        if (row[i] > max) max = row[i];

    TensorPrimitives.Subtract(row, max, row);   // row - max  (max ≤ 0 now)
    TensorPrimitives.Exp(row, row);             // safe: exp(x - max) ≤ 1
    TensorPrimitives.Divide(row, TensorPrimitives.Sum(row), row); // normalize to sum=1
}
```

- Subtracting the max doesn't change the softmax output (it cancels in the numerator and
  denominator) — it only makes it *computable*. This is a "which mathematically-equivalent
  expression do I ship?" decision (a theme returned to in Post 4 with GELU).
- Non-float fallback (Half/BFloat16) does the same idea in scalar `double` math
  (`AttentionKernels.cs:72-88`).

## 6. The padding mask: the `-∞` / tri-state trick

- Sentences are padded to 128 with the padding token; those positions must **not** vote.
  The Post 2 fact does the work: `exp(-∞) = 0`, so a score of `-∞` becomes zero weight
  after softmax, no matter what.
- So instead of branching in the hot loop, the mask is a **pre-built additive matrix**
  of 0s and `-∞`. Adding it to the scores is just one vectorized add. The mask builder
  (`MultiheadAttention.CreatePaddingMask`, `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs:138-155`):

```csharp
ReverseGradTensor<T> CreatePaddingMask(ReverseGradTensor<T> paddingMask, int qLen, int kvLen)
{
    int maskLen = paddingMask.Length;
    var maskData = new T[qLen * kvLen];
    var negInf = T.CreateChecked(double.NegativeInfinity);
    for (int j = 0; j < maskLen; j++)
    {
        if (paddingMask.Data[j] == T.Zero)           // this key position is padding
        {
            for (int i = 0; i < qLen; i++)
                maskData[i * kvLen + j] = negInf;    // -∞ the whole column j
        }
    }
    var col = NivaraColumn<T>.Create(maskData);
    var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
    tensor.Reshape(qLen, kvLen);
    return tensor;
}
```

- Column-wise `-∞`: for query row *i*, key column *j*, the score at `maskData[i*kvLen + j]`
  is `-∞` exactly when key *j* is a padding position. Rows stay untouched; real tokens keep
  normal scores.
- **The elegance worth highlighting:** masked attention needs **no** special code path in
  the kernel — the mask is just another `float[]` added to the scores (`:434-435`). A `-∞`
  in, a zero weight out, fully vectorized.
- The same mechanism powers causal masks (a future/upper-triangle `-∞` block) — 
  `ModuleHelpers<T>.CreateCausalMask` is called when `causal` is set
  (`MultiheadAttention.cs:126-136`).

## 7. Inference vs training: two different programs

- The `MultiHeadAttention` op checks `GradientUtils.ShouldTrackGrad(query, key, value)`
  (`ReverseGradOperations.cs:395`). In inference it returns **leaf tensors** and allocates
  no graph nodes and no `savedWeights` (`:423`).
- Training saves the softmax weights per head to compute dQ/dK/dV later (`:439-440`, and the
  backward VJP at `:450-521`) — that memory is *only* allocated inside a `Grad()` scope.
- Big-picture lesson: per-op bookkeeping is real cost at these sizes. The model runs
  thousands of forwards per second in production; the graph is only built when you ask for
  gradients.

## 8. What I learned / the numbers

1. **Fusing the kernel paid real dividends.** Before: attention was decomposed into
   per-head Slice/Transpose/MatMul/Multiply/Add/Softmax/Concat graph ops. After: one fused
   `MultiHeadAttention` node with heads packed once. DistilBERT encoder inference went
   **~236 ms → ~186 ms (~21% faster)** (`samples/NivaraInference/README.md:230`).
2. **The `-∞` mask removed an entire code path.** No `if masked` branch anywhere in the
   kernel. The mask is data.
3. **Packing heads is the difference between SIMD-friendly and cache-hostile.** All the
   matmuls feed `TensorsHelper.MultiplyCore` over contiguous spans (`AttentionKernels.cs:6-14`).
4. **Softmax max-subtraction is non-negotiable** — scores genuinely exceed exp's range;
   the naive formula is a latent `NaN`.
5. **Honest performance (worth writing down):** MiniLM at 128 tokens: PyTorch **11 ms** vs
   Nivara **110 ms** on the same CPU (`README.md:218-230`). The gap is dominated by the
   matmul kernels (naive loops vs MKL), *not* by the attention math or the architecture.
   Publishing this number is a feature: it shows the concept is right and the optimization
   is a known, bounded problem.
6. **Constants are part of the model spec.** The scale `1/√32` is baked into the module
   constructor (`MultiheadAttention.cs:40`; `BertSelfAttention`, `BertModel.cs:61`). `d`,
   `eps`, `num_heads` are read from `config.json` at load time. Getting the scale wrong
   silently breaks training dynamics.

---

## Checkpoint: the encoder map

> Step back from the details — here's where we are on the whole line.

```
text ──→ ●tokenizer ──→ token ids [128]
      │  ●token + position + segment lookups, summed ──→ [128×384]
   ×6 │  each BERT layer:
      │    ●[attention] ──→ ○⊕ (residual) ──→ ○[LayerNorm] ──→ ○[widen+rectify+squeeze] ──→ ○⊕ (residual) ──→ ○[LayerNorm]
      ▼
      ○[CLS] row ──→ ○L2 normalize ──→ unit vector [384] ──→ cosine similarity (cat vs dog)

● covered so far · ○ still ahead
```

- **Added:** the soft crossbar is now real — packed heads, a transpose-free QKᵀ, overflow-
  safe softmax, and a `-∞` mask that makes padding a free no-op.
- **Where you are:** the attention station is lit and *implemented*; it runs ~110 ms for
  the whole model (vs 11 ms in PyTorch — the gap is the matmul kernels, not the ideas).
- **Still unlit:** a BERT layer is not just attention. The wiring around it — the bypass
  wire and the gain stage after each block — and the widen/rectify/squeeze stage that does
  the per-token "thinking," then the readout that closes the cat-vs-dog loop.

## End hook → Post 4

> The attention block is complete. But a BERT layer is more than attention: every block is
> wrapped in a bypass wire that carries the signal through untouched, and each layer ends
> with a widen-rectify-squeeze stage that thinks per token. Then we read out the `[CLS]` row,
> L2-normalize it, and the whole series lands on one cosine: cat vs dog. Next: the bypass
> wire, the rectifier, and everything I learned building it.

## Facts & numbers to reuse (checklist)

- One forward per layer: 12 heads × (pack + QKᵀ + scale + mask + softmax + ×V + scatter).
- Head dim 32; scoreLen = 128×128 = 16,384 per head; qHeadsLen = 12×128×32.
- Scores are `[128,128]` per head; scaled by `1/√32`; masked by additive `0`/`-∞` matrix.
- Softmax subtracts row max (overflow safety), exp, normalize by row sum.
- Padding mask = `-∞` in masked columns → zero weight via `exp(-∞)=0`; no kernel branch.
- Inference allocates no graph/saved weights; training caches P per head for the backward.
- DistilBERT fused kernel: ~236 ms → ~186 ms (~21%).
- MiniLM: PyTorch 11 ms vs Nivara 110 ms (CPU, 128 tokens, 3 warmup + 10 timed).

## Source references

- `src/Nivara/AutoDiff/Operations/ReverseGradOperations.cs` — MultiHeadAttention (362-521),
  head loop (425-445), backward VJP (450-521), doc comments (345-361)
- `src/Nivara/AutoDiff/Operations/AttentionKernels.cs` — class doc (6-14), GatherHead (21-26),
  ScatterHead (32-37), PackHeads (42-46), SoftmaxRows/SoftmaxRow (53-88), SoftmaxBackwardRows (95-116)
- `src/Nivara/AutoDiff/Nn/MultiheadAttention.cs` — scale (40), ComputeAttention (118-136),
  CreatePaddingMask (138-155)
- `samples/Nivara.Samples/BertModel.cs` — BertSelfAttention.Forward (65-71), scale (61),
  ForwardWithMask mask building (73-93), batched block-diagonal mask (105-149)
- `samples/NivaraInference/README.md` — fused kernel win (230), benchmark table (218-230)

## Visual ideas

- Side-by-side: the Post 2 concept diagram vs the implementation loop (pack/scatter arrows).
- Memory layout sketch: `[128,384]` with head columns shaded, and the packed `[12,128,32]`.
- The `-∞` mask diagram: a 128×128 grid with a few columns shaded; softmax row shown
  zeroing those weights.
- A tiny worked head (e.g., 4 tokens, d=2) showing scores → max-subtract → exp → sum=1.

---

> **If you remember one thing from this post:** the `-∞` padding mask means masked attention
> needs no special code path — the mask is data, and the kernel never branches on it.
