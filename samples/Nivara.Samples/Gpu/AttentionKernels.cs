using ILGPU;
using ILGPU.Algorithms;

namespace Nivara.Samples.Gpu;

/// <summary>
/// Fused batched multi-head attention forward for the GPU encoder scenarios
/// (docs/BERT-GPU.md §3), mirroring the CPU AutoDiff BatchedMultiHeadAttention
/// exactly. Q/K/V live as head-interleaved [batch*seqLen, D] views (D = numHeads*headDim,
/// output of the q/k/v projections). One work item per (b, h, q):
///   score[b,h,q,j] = dot(Q[b,q,hd], K[b,j,hd]) over the head's headDim columns
///                  * scale; -inf where the key is suppressed
///   row softmax (max-subtract, exp, /sum) — same numerics as the CPU SoftmaxSingle
///   attnOut[b,q,h*headDim+d] = sum_j p_j * V[b,j,h*headDim+d]
/// A key is suppressed when the padding mask is 0 (mask[j] &lt; 0.5) or it falls outside the
/// sliding window: see <c>Keep</c> for the band convention (negative = global).
/// A row whose keys are all suppressed writes zeros rather than NaN, mirroring the CPU
/// safe-softmax clamp; such a row is an artifact of the mask constant and never reaches a
/// valid position, because a valid query never reads a padding key.
/// The score row is recomputed per pass (no per-work-item array), and the third pass
/// accumulates the probability-weighted V rows in a shared-memory tile instead of spilling
/// the p_j row to global scratch: [B, H, S, S] would be 268 MB at S=2048 and 4.3 GB at
/// S=8192, and only the per-(b,h,q) accumulator needs to be live. Each output element's
/// j loop stays sequential and ascending, so the accumulation order and precision shape
/// match the CPU kernel exactly.
/// </summary>
internal static class AttentionKernels
{
    public static void BatchedAttention(
        ArrayView<float> q,
        ArrayView<float> k,
        ArrayView<float> v,
        ArrayView<float> mask,
        ArrayView<float> attnOut,
        int batch,
        int seqLen,
        int numHeads,
        int headDim,
        int band,
        float scale)
    {
        int total = batch * numHeads * seqLen;
        int idx = Grid.GlobalIndex.X;
        if (idx >= total) return;

        int b = idx / (numHeads * seqLen);
        int rem = idx - b * (numHeads * seqLen);
        int h = rem / seqLen;
        int qPos = rem - h * seqLen;

        int D = numHeads * headDim;
        int dOffset = h * headDim;
        int qRow = b * seqLen + qPos;
        int maskBase = b * seqLen;
        int outBase = qRow * D;

        float max = float.NegativeInfinity;
        for (int j = 0; j < seqLen; j++)
        {
            float s = RowScore(q, k, qRow, b * seqLen + j, dOffset, D, headDim, scale);
            if (!Keep(mask, maskBase, qPos, j, band)) s = float.NegativeInfinity;
            if (s > max) max = s;
        }

        // Every key suppressed: the row max is -inf, so s - max is NaN and the whole output
        // goes NaN from here. Mirrors the CPU safe-softmax clamp (GradKernels.cs) — such a row
        // produces zeros, which is a mask-constant artifact that never reaches a valid
        // position. Reachable as soon as a caller can suppress an entire row, e.g. a
        // sliding-window band over a padded batch.
        if (max == float.NegativeInfinity)
        {
            for (int d = 0; d < headDim; d++)
                attnOut[outBase + dOffset + d] = 0f;
            return;
        }

        float sum = 0f;
        for (int j = 0; j < seqLen; j++)
        {
            float s = RowScore(q, k, qRow, b * seqLen + j, dOffset, D, headDim, scale);
            if (!Keep(mask, maskBase, qPos, j, band)) s = float.NegativeInfinity;
            sum += XMath.Exp(s - max);
        }

        // Shared-memory accumulator tile. ILGPU's SharedMemory is per work *group*, not per
        // work item, so the tile is [MaxHeadDim rows x AttentionGroupSize columns] and column
        // Group.IdxX is this thread's private slice - two work items in a group would otherwise
        // race on the same accumulators. DenseX keeps a thread's MaxHeadDim floats contiguous
        // (offset = d + lane * MaxHeadDim). Both dimensions are compile-time constants because
        // ILGPU requires a statically known allocation size and headDim/group size are runtime
        // arguments; only the first headDim rows are touched, and
        // GpuBuffers.ValidateAttentionLocalMemory rejects a larger headDim at construction.
        var acc = SharedMemory.Allocate2D<float, Stride2D.DenseX>(
            new Index2D(GpuBuffers.MaxHeadDim, GpuBuffers.AttentionGroupSize),
            new Stride2D.DenseX(GpuBuffers.MaxHeadDim));
        int lane = Group.IdxX;
        for (int d = 0; d < headDim; d++)
            acc[d, lane] = 0f;

        for (int j = 0; j < seqLen; j++)
        {
            float s = RowScore(q, k, qRow, b * seqLen + j, dOffset, D, headDim, scale);
            if (!Keep(mask, maskBase, qPos, j, band)) s = float.NegativeInfinity;
            float p = XMath.Exp(s - max) / sum;
            int vBase = (b * seqLen + j) * D + dOffset;
            for (int d = 0; d < headDim; d++)
                acc[d, lane] += p * v[vBase + d];
        }

        for (int d = 0; d < headDim; d++)
            attnOut[outBase + dOffset + d] = acc[d, lane];
    }

    /// <summary>
    /// Whether query <paramref name="qPos"/> may attend to key <paramref name="j"/>: the
    /// padding mask must be set, and when <paramref name="band"/> is non-negative the pair must
    /// also be within that many positions of each other.
    /// </summary>
    /// <remarks>
    /// A negative <paramref name="band"/> means global attention, matching the CPU's convention
    /// (<c>ModernBertMasks.Build</c> treats <c>band &lt; 0</c> as unbounded). The half-width
    /// window <c>|qPos - j| &lt;= band</c> is what the CPU's
    /// <c>firstAllowed = max(0, i - band)</c> / <c>lastAllowed = min(validLength - 1, i + band)</c>
    /// describe once the padding mask is applied on top.
    /// </remarks>
    static bool Keep(ArrayView<float> mask, int maskBase, int qPos, int j, int band)
        => mask[maskBase + j] >= 0.5f && (band < 0 || XMath.Abs(qPos - j) <= band);

    static float RowScore(
        ArrayView<float> q,
        ArrayView<float> k,
        int qRow,
        int jRow,
        int dOffset,
        int D,
        int headDim,
        float scale)
    {
        int qBase = qRow * D + dOffset;
        int kBase = jRow * D + dOffset;
        float acc = 0f;
        for (int d = 0; d < headDim; d++)
            acc += q[qBase + d] * k[kBase + d];
        return acc * scale;
    }
}