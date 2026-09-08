using System.Buffers;
using System.Numerics;
using System.Numerics.Tensors;

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
        => GradKernels.SoftmaxGradient(weights, dS, dS, cols);

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
            output.Clear();
            for (int qh = 0; qh < numHeads; qh++)
            {
                int kvHead = qh / repeat;
                var qHead = q.Slice(qh * headDim, headDim);
                for (int j = 0; j < kvLen; j++)
                    scoresSpan[j] = scale * TensorPrimitives.Dot(qHead, kCache.Slice(j * kvWidth + kvHead * headDim, headDim));
                SoftmaxRows(scoresSpan, 1, kvLen);
                var outHead = output.Slice(qh * headDim, headDim);
                for (int j = 0; j < kvLen; j++)
                {
                    var vRow = vCache.Slice(j * kvWidth + kvHead * headDim, headDim);
                    T w = scoresSpan[j];
                    for (int d = 0; d < headDim; d++)
                        outHead[d] += w * vRow[d];
                }
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(scores);
        }
    }
}
