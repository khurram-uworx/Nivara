using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Nivara.AutoDiff.Operations;

/// <summary>
/// Span-level kernels for fused multi-head scaled dot-product attention.
///
/// Q/K/V are row-major [rows, D]; each head owns contiguous columns
/// [h*headDim, (h+1)*headDim). Head matrices are packed once into a
/// [numHeads, rows, headDim] contiguous layout so every matmul below feeds
/// the SIMD <see cref="TensorsHelper.MultiplyCore{T}"/> path with zero
/// per-head transposes (QK^T uses the transposed-B layout directly).
/// </summary>
internal static class AttentionKernels<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Gathers one head's contiguous columns from a [rows, D] matrix into a
    /// contiguous [rows, headDim] span.
    /// </summary>
    public static void GatherHead(ReadOnlySpan<T> src, Span<T> dst, int rows, int D, int head, int headDim)
    {
        int hs = head * headDim;
        for (int r = 0; r < rows; r++)
            src.Slice(r * D + hs, headDim).CopyTo(dst.Slice(r * headDim, headDim));
    }

    /// <summary>
    /// Scatters a contiguous [rows, headDim] span back into one head's columns
    /// of a [rows, D] matrix.
    /// </summary>
    public static void ScatterHead(ReadOnlySpan<T> src, Span<T> dst, int rows, int D, int head, int headDim)
    {
        int hs = head * headDim;
        for (int r = 0; r < rows; r++)
            src.Slice(r * headDim, headDim).CopyTo(dst.Slice(r * D + hs, headDim));
    }

    /// <summary>
    /// Packs a [rows, D] matrix into [numHeads, rows, headDim] head-major layout.
    /// </summary>
    public static void PackHeads(ReadOnlySpan<T> src, Span<T> dst, int rows, int numHeads, int headDim)
    {
        for (int h = 0; h < numHeads; h++)
            GatherHead(src, dst.Slice(h * rows * headDim, rows * headDim), rows, numHeads * headDim, h, headDim);
    }

    /// <summary>
    /// In-place row-wise softmax (max subtraction, exp, normalize). Delegates to
    /// <see cref="GradKernels.SoftmaxRowsInPlace{T}"/> so attention and the
    /// Softmax op share one kernel.
    /// </summary>
    public static void SoftmaxRows(Span<T> x, int rows, int cols)
        => GradKernels.SoftmaxRowsInPlace(x, rows, cols);

    /// <summary>
    /// In-place softmax backward: dS[i,j] = P[i,j] * (dP[i,j] - dot(P_i, dP_i)).
    /// Delegates to <see cref="GradKernels.SoftmaxGradient{T}"/> with the
    /// gradient span aliasing the output span (the per-row dot is computed
    /// before any write), so attention and the Softmax op share one kernel.
    /// </summary>
    public static void SoftmaxBackwardRows(ReadOnlySpan<T> weights, Span<T> dS, int rows, int cols)
        => SoftmaxBackwardRows(weights, dS, rows, cols, ReadOnlySpan<bool>.Empty);

    public static void SoftmaxBackwardRows(ReadOnlySpan<T> weights, Span<T> dS, int rows, int cols,
                                           ReadOnlySpan<bool> scoreIndependentRows = default)
    {
        if (!scoreIndependentRows.IsEmpty && scoreIndependentRows.Length != rows)
            throw new ArgumentException(
                $"Row flags length {scoreIndependentRows.Length} does not match rows {rows}.", nameof(scoreIndependentRows));

        GradKernels.SoftmaxGradient(weights, dS, dS, cols, scoreIndependentRows);
    }

    /// <summary>
    /// In-place additive attention mask over a flat [qLen, kvLen] score buffer.
    /// A cell whose mask entry is exactly <c>-inf</c> is suppressed.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The suppressed cell is <em>assigned</em> <c>-inf</c>, not summed into. Summing cannot
    /// suppress a non-finite score: <c>NaN + (-inf) = NaN</c> and <c>(+inf) + (-inf) = NaN</c>, so
    /// a score that already diverged (a NaN from q/k/v, or a <c>+inf</c> from an overflow in a
    /// badly-scaled layer) escapes the mask and poisons every query row in the frame on the next
    /// layer. Assigning discards the bad value instead of combining with it, so suppression is
    /// unconditional.
    /// </para>
    /// <para>
    /// Only the <c>-inf</c> cell counts as suppression. Every other entry stays additive, which is
    /// what keeps the two fill conventions in the tree behaving as they do:
    /// <list type="bullet">
    /// <item>A <c>NaN</c> or <c>+inf</c> mask cell still propagates to <c>NaN</c>. <c>NaN == -inf</c>
    /// is false (IEEE 754 unordered comparison), so such a cell takes the additive branch and the
    /// row poisons exactly as PyTorch's <c>score + mask</c> does — see
    /// <c>BandedAttention_NonFiniteMaskCells_PropagateAsNaNRows_MatchingPyTorch</c>, which is a
    /// parity fixture and must keep passing.</item>
    /// <item>A finite fill keeps its magnitude and still saturates. HuggingFace's
    /// <c>masking_utils</c> fills with <c>torch.finfo(dtype).min</c> rather than <c>-inf</c>, and
    /// that magnitude is load-bearing: at float32 the ULP at 3.4e38 is ~2e31, so
    /// <c>score + finfo.min</c> collapses a fully-masked row to one constant and the row softmax
    /// reduces to a uniform average of V. Discarding the magnitude would turn that into a raw
    /// softmax over undamped scores — see
    /// <c>BandedAttention_FinfoMinFill_Forward_MatchesPyTorch</c>.</item>
    /// </list>
    /// An unsuppressed cell therefore keeps today's behaviour bit for bit, including
    /// <c>NaN + 0 = NaN</c>: a diverged score at a position the mask does not suppress must still
    /// announce itself rather than be quietly swallowed.
    /// </para>
    /// <para>
    /// A scalar loop rather than a <c>TensorPrimitives</c> call, because BCL has no select/blend
    /// primitive and this mask cannot be expressed as arithmetic. Tracked as #480.
    /// </para>
    /// <para>
    /// The lengths must match exactly. <c>TensorPrimitives.Add</c>, which this replaces, enforced
    /// that; the loop on its own would only fault on a short mask and silently truncate a long
    /// one, so a mis-shaped mask would become a wrong answer instead of an exception.
    /// </para>
    /// </remarks>
    public static void ApplyMask(Span<T> scores, ReadOnlySpan<T> mask)
    {
        ApplyMask(scores, mask, Span<bool>.Empty, 0, 0);
    }

    /// <summary>
    /// In-place additive attention mask over a flat [qLen, kvLen] score buffer, recording
    /// which query rows became score-independent due to saturation by the mask fill.
    /// </summary>
    /// <param name="scores">Flat score buffer [rows * cols].</param>
    /// <param name="mask">Additive mask buffer [rows * cols].</param>
    /// <param name="scoreIndependentRows">
    /// Per-row flag where <c>true</c> means every cell in the row was annihilated
    /// by the mask (the row's pre-softmax score no longer depends on any input after
    /// masking). Pass <see cref="Span{Boolean}.Empty"/> to ignore.
    /// </param>
    /// <param name="rows">Query rows. If <paramref name="scoreIndependentRows"/> is not empty,
    /// this must equal its length.</param>
    /// <param name="cols">Keys per row.</param>
    /// <remarks>
    /// A cell is <em>dead</em> (score annihilated) if <c>mask[i] != T.Zero</c> and
    /// <c>(scores[i] + mask[i]) == mask[i]</c>. This is the definitive test: adding the fill
    /// does not change the stored value, so the final pre-softmax value carries no
    /// dependence on the original score. The <c>-inf</c> branch is subsumed by this test
    /// (adding <c>-inf</c> to a finite-or-infinite score produces <c>-inf</c> only in the
    /// suppressed case; the equality test detects annihilation consistently). A row is
    /// flagged when every cell in it is dead.
    /// </remarks>
    public static void ApplyMask(Span<T> scores, ReadOnlySpan<T> mask,
                                 Span<bool> scoreIndependentRows, int rows, int cols)
    {
        if (mask.Length != scores.Length)
            throw new ArgumentException(
                $"Mask length {mask.Length} does not match score length {scores.Length}.", nameof(mask));
        if (!scoreIndependentRows.IsEmpty)
        {
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols < 0) throw new ArgumentOutOfRangeException(nameof(cols));
            if (scoreIndependentRows.Length != rows)
                throw new ArgumentException(
                    $"Row flags length {scoreIndependentRows.Length} does not match rows {rows}.", nameof(scoreIndependentRows));
            if ((long)rows * cols != scores.Length)
                throw new ArgumentException(
                    $"rows*cols ({rows * cols}) must equal score length ({scores.Length}).");
        }

        if (scoreIndependentRows.IsEmpty)
        {
            for (int i = 0; i < scores.Length; i++)
            {
                T m = mask[i];
                if (m == T.NegativeInfinity)
                {
                    scores[i] = T.NegativeInfinity;
                }
                else
                {
                    T s = scores[i];
                    T t = s + m;
                    scores[i] = t;
                }
            }
            return;
        }

        scoreIndependentRows.Clear();
        int idx = 0;
        for (int r = 0; r < rows; r++)
        {
            bool rowIndependent = true;
            for (int c = 0; c < cols; c++, idx++)
            {
                T s = scores[idx];
                T m = mask[idx];
                T t;
                bool dead = false;
                if (m == T.NegativeInfinity)
                {
                    t = T.NegativeInfinity;
                    dead = true;
                }
                else
                {
                    t = s + m;
                    if (m != T.Zero && t == m)
                        dead = true;
                }
                scores[idx] = t;
                if (!dead)
                    rowIndependent = false;
            }
            scoreIndependentRows[r] = rowIndependent;
        }
    }

    /// <summary>
    /// Single-query GQA decode attention over a cached KV prefix, with no materialization.
    ///
    /// Query is one row <c>[numHeads * headDim]</c>; the key/value caches are row-major
    /// <c>[kvLen, numKvHeads * headDim]</c> (RoPE'd keys, head-interleaved per row). Each query
    /// head <c>qh</c> attends against its shared KV head <c>kv = qh / repeat</c>
    /// (<c>repeat = numHeads / numKvHeads</c>) — a <em>virtual</em> GQA mapping that reads the
    /// cache strided, so no GqaRepeatKV expansion, no head repack, and no BlockCopy of the
    /// cached prefix occur. Scores are computed per query head (each query head has its own Q),
    /// folded with the scale, softmaxed over the single row, then used to weight the strided V
    /// rows straight into the output span.
    ///
    /// This is the inference-only attention path for <c>LlamaCausalAttention.ForwardCached</c>;
    /// it builds no graph nodes and allocates only a rented score buffer (ArrayPool-reused).
    /// </summary>
    /// <param name="q">Single query row, <c>[numHeads * headDim]</c></param>
    /// <param name="kCache">RoPE'd key cache, <c>[kvLen * numKvHeads * headDim]</c></param>
    /// <param name="vCache">Value cache, <c>[kvLen * numKvHeads * headDim]</c></param>
    /// <param name="output">Attention output, <c>[numHeads * headDim]</c></param>
    /// <param name="kvLen">Number of cached positions (inclusive of the new token)</param>
    /// <param name="numHeads">Query head count (must be divisible by <paramref name="numKvHeads"/>)</param>
    /// <param name="numKvHeads">Key/value head count</param>
    /// <param name="headDim">Per-head dimension</param>
    /// <param name="scale">Attention scale (usually <c>1/sqrt(headDim)</c>)</param>
    public static void DecodeAttention(
        ReadOnlySpan<T> q,
        ReadOnlySpan<T> kCache,
        ReadOnlySpan<T> vCache,
        Span<T> output,
        int kvLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        T scale)
    {
        if (kvLen <= 0) throw new ArgumentOutOfRangeException(nameof(kvLen));
        if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be a positive multiple of numKvHeads ({numKvHeads}).");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        int repeat = numHeads / numKvHeads;
        int kvWidth = numKvHeads * headDim;
        if (q.Length < numHeads * headDim) throw new ArgumentOutOfRangeException(nameof(q));
        if (kCache.Length < kvLen * kvWidth || vCache.Length < kvLen * kvWidth)
            throw new ArgumentOutOfRangeException(nameof(kCache));
        if (output.Length < numHeads * headDim) throw new ArgumentOutOfRangeException(nameof(output));

        var scores = ArrayPool<T>.Shared.Rent(Math.Max(kvLen, 1));
        try
        {
            var scoresSpan = scores.AsSpan(0, kvLen);
            // Float accelerates on any SIMD-capable runtime; otherwise the scalar
            // loop below (the generic fallback) handles every T.
            bool floatVectors = typeof(T) == typeof(float)
                && (Vector512.IsHardwareAccelerated || Vector.IsHardwareAccelerated);
            output.Clear();
            for (int qh = 0; qh < numHeads; qh++)
            {
                int kvHead = qh / repeat;
                var qHead = q.Slice(qh * headDim, headDim);
                for (int j = 0; j < kvLen; j++)
                    scoresSpan[j] = scale * TensorPrimitives.Dot(qHead, kCache.Slice(j * kvWidth + kvHead * headDim, headDim));
                SoftmaxRows(scoresSpan, 1, kvLen);
                var outHead = output.Slice(qh * headDim, headDim);
                if (floatVectors)
                {
                    var scoresF = MemoryMarshal.Cast<T, float>(scoresSpan);
                    var vCacheF = MemoryMarshal.Cast<T, float>(vCache);
                    var outHeadF = MemoryMarshal.Cast<T, float>(outHead);
                    if (Vector512.IsHardwareAccelerated)
                        VPhaseFloat512(scoresF, vCacheF, outHeadF, kvLen, kvWidth, kvHead, headDim);
                    else
                        VPhaseFloat(scoresF, vCacheF, outHeadF, kvLen, kvWidth, kvHead, headDim);
                }
                else
                {
                    for (int j = 0; j < kvLen; j++)
                    {
                        var vRow = vCache.Slice(j * kvWidth + kvHead * headDim, headDim);
                        T w = scoresSpan[j];
                        for (int d = 0; d < headDim; d++)
                            outHead[d] += w * vRow[d];
                    }
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(scores);
        }
    }

    /// <summary>Vector512 (AVX-512) V-phase: each 16-lane block of the head
    /// accumulates across all KV rows in one zmm register via broadcast-FMA.
    /// Tail (headDim not a multiple of 16) is handled scalar.</summary>
    static void VPhaseFloat512(ReadOnlySpan<float> scores, ReadOnlySpan<float> vCache, Span<float> outHead, int kvLen, int kvWidth, int kvHead, int headDim)
    {
        int kvBase = kvHead * headDim;
        int width = Vector512<float>.Count;
        ref float vRef = ref MemoryMarshal.GetReference(vCache);
        ref float oRef = ref MemoryMarshal.GetReference(outHead);
        int d = 0;
        for (; d + width <= headDim; d += width)
        {
            var acc = Vector512<float>.Zero;
            for (int j = 0; j < kvLen; j++)
            {
                var wVec = Vector512.Create(scores[j]);
                var vRow = Vector512.LoadUnsafe(ref vRef, (nuint)(j * kvWidth + kvBase + d));
                acc = Vector512.FusedMultiplyAdd(wVec, vRow, acc);
            }
            Vector512.StoreUnsafe(acc, ref oRef, (nuint)d);
        }
        for (; d < headDim; d++)
        {
            float s = 0;
            for (int j = 0; j < kvLen; j++)
                s += scores[j] * vCache[j * kvWidth + kvBase + d];
            outHead[d] = s;
        }
    }

    /// <summary>Portable variable-width SIMD V-phase (<see cref="Vector{T}"/>, sized
    /// to the host — SSE2/AVX2/NEON on non-AVX-512 machines). Same d-blocked
    /// broadcast-FMA structure as the Vector512 kernel.</summary>
    static void VPhaseFloat(ReadOnlySpan<float> scores, ReadOnlySpan<float> vCache, Span<float> outHead, int kvLen, int kvWidth, int kvHead, int headDim)
    {
        int kvBase = kvHead * headDim;
        int width = Vector<float>.Count;
        ref float vRef = ref MemoryMarshal.GetReference(vCache);
        ref float oRef = ref MemoryMarshal.GetReference(outHead);
        int d = 0;
        for (; d + width <= headDim; d += width)
        {
            var acc = Vector<float>.Zero;
            for (int j = 0; j < kvLen; j++)
            {
                var wVec = new Vector<float>(scores[j]);
                var vRow = Vector.LoadUnsafe(ref vRef, (nuint)(j * kvWidth + kvBase + d));
                acc = Vector.FusedMultiplyAdd(wVec, vRow, acc);
            }
            Vector.StoreUnsafe(acc, ref oRef, (nuint)d);
        }
        for (; d < headDim; d++)
        {
            float s = 0;
            for (int j = 0; j < kvLen; j++)
                s += scores[j] * vCache[j * kvWidth + kvBase + d];
            outHead[d] = s;
        }
    }

    /// <summary>
    /// Batched (multi-row) GQA prefill attention over a chunk of <c>qLen</c> rows with a
    /// causal mask, mirroring <see cref="DecodeAttention"/>'s no-materialization contract.
    ///
    /// Query is <c>[qLen, numHeads * headDim]</c> (post-RoPE, contiguous rows). The key/value
    /// caches are row-major <c>[kvLen, numKvHeads * headDim]</c> holding this chunk's rows
    /// (post-RoPE, pre-repeat, per-KV-head layout exactly as captured by
    /// <c>LlamaCausalAttention.ForwardPrefill</c>). Per query head <c>qh</c> the shared KV head
    /// <c>kv = qh / repeat</c> (<c>repeat = numHeads / numKvHeads</c>) is attended with a
    /// <em>virtual</em> GQA mapping — the cache is read strided, so no GqaRepeatKV expansion
    /// occurs and only <c>numKvHeads</c> packs of K/V are gathered (not <c>numHeads</c>).
    ///
    /// Per head the work mirrors <see cref="ReverseGradOperations.MultiHeadAttention{T}"/>
    /// exactly: packed heads → <c>GradKernels.MatMulTransposedB</c> QK^T → scale → masked
    /// positions set to −∞ (the causal additive-mask values, folded in place) → full-row
    /// softmax → <c>GradKernels.MatMul</c> with V → scatter. The rows, scores, mask, and
    /// per-head output buffers are ArrayPool-rented and reused across heads and layers, so the
    /// only heap allocation is the output tensor itself. The absolute position offset is
    /// irrelevant because only this chunk's rows are visible to each other, exactly as the
    /// current <c>[qLen, qLen]</c> mask semantics.
    ///
    /// This is the inference-only prefill attention path for
    /// <c>LlamaCausalAttention.ForwardCore</c>; it builds no graph nodes.
    /// </summary>
    /// <param name="q">Query rows, <c>[qLen * numHeads * headDim]</c></param>
    /// <param name="kCache">RoPE'd key cache (this chunk's rows), <c>[qLen * numKvHeads * headDim]</c></param>
    /// <param name="vCache">Value cache (this chunk's rows), <c>[qLen * numKvHeads * headDim]</c></param>
    /// <param name="output">Attention output, <c>[qLen * numHeads * headDim]</c></param>
    /// <param name="qLen">Number of query rows in this chunk (must be ≥ 1)</param>
    /// <param name="numHeads">Query head count (must be divisible by <paramref name="numKvHeads"/>)</param>
    /// <param name="numKvHeads">Key/value head count</param>
    /// <param name="headDim">Per-head dimension</param>
    /// <param name="scale">Attention scale (usually <c>1/sqrt(headDim)</c>)</param>
    public static void BatchedAttention(
        ReadOnlySpan<T> q,
        ReadOnlySpan<T> kCache,
        ReadOnlySpan<T> vCache,
        Span<T> output,
        int qLen,
        int numHeads,
        int numKvHeads,
        int headDim,
        T scale)
    {
        if (qLen <= 0) throw new ArgumentOutOfRangeException(nameof(qLen));
        if (numHeads <= 0 || numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new ArgumentException($"numHeads ({numHeads}) must be a positive multiple of numKvHeads ({numKvHeads}).");
        if (headDim <= 0) throw new ArgumentOutOfRangeException(nameof(headDim));

        int repeat = numHeads / numKvHeads;
        int kvWidth = numKvHeads * headDim;
        int D = numHeads * headDim;
        if (q.Length < qLen * D) throw new ArgumentOutOfRangeException(nameof(q));
        if (kCache.Length < qLen * kvWidth || vCache.Length < qLen * kvWidth)
            throw new ArgumentOutOfRangeException(nameof(kCache));
        if (output.Length < qLen * D) throw new ArgumentOutOfRangeException(nameof(output));

        int qHeadsLen = numHeads * qLen * headDim;
        int kvHeadsLen = numKvHeads * qLen * headDim;
        int scoreLen = qLen * qLen;

        var qHeads = ArrayPool<T>.Shared.Rent(Math.Max(qHeadsLen, 1));
        var kHeads = ArrayPool<T>.Shared.Rent(Math.Max(kvHeadsLen, 1));
        var vHeads = ArrayPool<T>.Shared.Rent(Math.Max(kvHeadsLen, 1));
        var scores = ArrayPool<T>.Shared.Rent(Math.Max(scoreLen, 1));
        var outHead = ArrayPool<T>.Shared.Rent(Math.Max(qLen * headDim, 1));
        try
        {
            var qHeadsSpan = qHeads.AsSpan(0, qHeadsLen);
            var kHeadsSpan = kHeads.AsSpan(0, kvHeadsLen);
            var vHeadsSpan = vHeads.AsSpan(0, kvHeadsLen);
            var scoresSpan = scores.AsSpan(0, scoreLen);

            PackHeads(q, qHeadsSpan, qLen, numHeads, headDim);
            for (int h = 0; h < numKvHeads; h++)
            {
                GatherHead(kCache, kHeads.AsSpan(h * qLen * headDim, qLen * headDim), qLen, kvWidth, h, headDim);
                GatherHead(vCache, vHeads.AsSpan(h * qLen * headDim, qLen * headDim), qLen, kvWidth, h, headDim);
            }

            output.Clear();
            for (int qh = 0; qh < numHeads; qh++)
            {
                int kvHead = qh / repeat;
                var qhBuf = qHeadsSpan.Slice(qh * qLen * headDim, qLen * headDim);
                var khBuf = kHeadsSpan.Slice(kvHead * qLen * headDim, qLen * headDim);
                var vhBuf = vHeadsSpan.Slice(kvHead * qLen * headDim, qLen * headDim);

                // QK^T via the same bulk kernel MultiHeadAttention uses, then scale and fold the
                // causal mask (−∞ beyond the diagonal) directly into the score buffer so the
                // full-row softmax sees identical values to the additive-mask path.
                GradKernels.MatMulTransposedB(qhBuf, khBuf, scores, qLen, headDim, qLen);
                TensorPrimitives.Multiply(scoresSpan, scale, scoresSpan);
                for (int i = 0; i < qLen; i++)
                {
                    int rowStart = i * qLen;
                    for (int j = i + 1; j < qLen; j++)
                        scores[rowStart + j] = T.NegativeInfinity;
                }
                SoftmaxRows(scoresSpan, qLen, qLen);

                GradKernels.MatMul(scores, vhBuf, outHead, qLen, qLen, headDim);
                ScatterHead(outHead.AsSpan(0, qLen * headDim), output, qLen, D, qh, headDim);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(qHeads);
            ArrayPool<T>.Shared.Return(kHeads);
            ArrayPool<T>.Shared.Return(vHeads);
            ArrayPool<T>.Shared.Return(scores);
            ArrayPool<T>.Shared.Return(outHead);
        }
    }
}
