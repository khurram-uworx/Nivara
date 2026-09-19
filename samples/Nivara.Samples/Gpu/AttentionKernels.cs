using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace Nivara.Samples.Gpu;

/// <summary>
/// Fused batched multi-head attention forward for the DistilBERT GPU scenario
/// (docs/BERT-GPU.md §3), mirroring the CPU AutoDiff BatchedMultiHeadAttention
/// exactly. Q/K/V live as head-interleaved [batch*seqLen, D] views (D = numHeads*headDim,
/// output of the q/k/v projections). One work item per (b, h, q):
///   score[b,h,q,j] = dot(Q[b,q,hd], K[b,j,hd]) over the head's headDim columns
///                  * scale; -inf where the padding mask is 0 (mask[j] &lt; 0.5)
///   row softmax (max-subtract, exp, /sum) — same numerics as the CPU SoftmaxSingle
///   attnOut[b,q,h*headDim+d] = sum_j p_j * V[b,j,h*headDim+d]
/// The score row is recomputed per pass (no per-work-item array); the p_j row is
/// spilled to a per-(b,h,q) global scratch row so each output element's accumulation
/// is a plain sequential j loop — the same effective per-element order and precision
/// shape as the CPU kernel.
/// </summary>
internal static class AttentionKernels
{
    public static void BatchedAttention(
        ArrayView<float> q,
        ArrayView<float> k,
        ArrayView<float> v,
        ArrayView<float> mask,
        ArrayView<float> attnOut,
        ArrayView<float> scores,
        int batch,
        int seqLen,
        int numHeads,
        int headDim,
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

        float max = float.NegativeInfinity;
        for (int j = 0; j < seqLen; j++)
        {
            float s = RowScore(q, k, qRow, b * seqLen + j, dOffset, D, headDim, scale);
            if (mask[maskBase + j] < 0.5f) s = float.NegativeInfinity;
            if (s > max) max = s;
        }

        float sum = 0f;
        for (int j = 0; j < seqLen; j++)
        {
            float s = RowScore(q, k, qRow, b * seqLen + j, dOffset, D, headDim, scale);
            if (mask[maskBase + j] < 0.5f) s = float.NegativeInfinity;
            sum += XMath.Exp(s - max);
        }

        int scoreBase = (b * numHeads + h) * (seqLen * seqLen) + qPos * seqLen;
        for (int j = 0; j < seqLen; j++)
        {
            float s = RowScore(q, k, qRow, b * seqLen + j, dOffset, D, headDim, scale);
            if (mask[maskBase + j] < 0.5f) s = float.NegativeInfinity;
            scores[scoreBase + j] = XMath.Exp(s - max) / sum;
        }

        int outBase = qRow * D;
        for (int d = 0; d < headDim; d++)
        {
            float acc = 0f;
            for (int j = 0; j < seqLen; j++)
                acc += scores[scoreBase + j] * v[(b * seqLen + j) * D + dOffset + d];
            attnOut[outBase + dOffset + d] = acc;
        }
    }

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