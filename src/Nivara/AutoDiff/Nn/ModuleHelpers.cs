using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

internal static class ModuleHelpers<T> where T : struct, IFloatingPointIeee754<T>
{
    internal static ReadOnlySpan<T> GetSpan(ReverseGradTensor<T> tensor)
    {
        tensor.Data.TryGetSpan(out var span);
        return span;
    }

    internal static T[] CopyToTemp(NivaraColumn<T> column, int length)
    {
        var arr = new T[length];
        column.CopyTo(arr, T.Zero);
        return arr;
    }

    internal static ReverseGradTensor<T> CreateBlockDiagonalMask(
        ReverseGradTensor<T> attentionMask, int batchSize, int seqLen)
    {
        int N = batchSize * seqLen;
        var maskData = new T[N * N];
        var negInf = T.CreateChecked(double.NegativeInfinity);

        for (int b = 0; b < batchSize; b++)
        {
            int offset = b * seqLen;

            for (int j = 0; j < seqLen; j++)
            {
                int colIdx = offset + j;
                if (attentionMask.Data[colIdx] == T.Zero)
                {
                    for (int i = 0; i < seqLen; i++)
                    {
                        int row = offset + i;
                        maskData[row * N + colIdx] = negInf;
                    }
                }
            }

            for (int other = 0; other < batchSize; other++)
            {
                if (other == b) continue;
                int otherOffset = other * seqLen;
                for (int i = 0; i < seqLen; i++)
                {
                    int row = offset + i;
                    for (int j = 0; j < seqLen; j++)
                    {
                        int otherCol = otherOffset + j;
                        maskData[row * N + otherCol] = negInf;
                    }
                }
            }
        }

        var maskCol = NivaraColumn<T>.CreateFromOwnedArray(maskData);
        var tensor = new ReverseGradTensor<T>(maskCol, requiresGrad: false);
        tensor.Reshape(N, N);
        return tensor;
    }

    internal static ReverseGradTensor<T> CreateCausalMask(int L)
    {
        return CreateCausalMask(L, L);
    }

    internal static ReverseGradTensor<T> CreateCausalMask(int qLen, int kvLen)
    {
        var maskData = new T[qLen * kvLen];
        for (int i = 0; i < qLen; i++)
            for (int j = 0; j < kvLen; j++)
                if (j > i)
                    maskData[i * kvLen + j] = T.CreateChecked(double.NegativeInfinity);

        var col = NivaraColumn<T>.CreateFromOwnedArray(maskData);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(qLen, kvLen);
        return tensor;
    }

    internal static (ReverseGradTensor<T>? runningMean, ReverseGradTensor<T>? runningVar, ReverseGradTensor<T>? numBatchesTracked)
        UpdateRunningStats(
            ReverseGradTensor<T>? runningMean,
            ReverseGradTensor<T>? runningVar,
            ReverseGradTensor<T>? numBatchesTracked,
            T[] batchMean,
            T[] batchInvStd,
            int numFeatures,
            T momentum,
            T eps)
    {
        if (runningMean == null || runningVar == null || numBatchesTracked == null)
            return (runningMean, runningVar, numBatchesTracked);

        var rmData = new T[numFeatures];
        runningMean.Data.CopyTo(rmData, T.Zero);
        var rvData = new T[numFeatures];
        runningVar.Data.CopyTo(rvData, T.Zero);

        var oneMinusMomentum = T.One - momentum;

        for (int i = 0; i < numFeatures; i++)
        {
            T variance = T.One / (batchInvStd[i] * batchInvStd[i]) - eps;
            rmData[i] = rmData[i] * oneMinusMomentum + batchMean[i] * momentum;
            rvData[i] = rvData[i] * oneMinusMomentum + variance * momentum;
        }

        var newMean = ReverseGradTensor<T>.FromArray(rmData, requiresGrad: false);
        var newVar = ReverseGradTensor<T>.FromArray(rvData, requiresGrad: false);
        var count = new T[] { numBatchesTracked[0] + T.One };
        var newCount = ReverseGradTensor<T>.FromArray(count, requiresGrad: false);

        return (newMean, newVar, newCount);
    }

    internal static ReverseGradTensor<T> Reparameterize(
        ReverseGradTensor<T> mu,
        ReverseGradTensor<T> logVar,
        bool isTraining,
        int? seed)
    {
        if (!isTraining)
            return mu;
        return ReverseGradOperations.SampleNormal(mu, logVar, seed);
    }

    /// <summary>
    /// Builds the input list for a module's <see cref="OpNode{T}"/>, appending only the
    /// optional parameter tensors that actually exist.
    /// </summary>
    /// <remarks>
    /// Under the <c>OpNode.Inputs</c> contract a node lists every tensor whose gradient its
    /// backward is responsible for, so a module must list the parameters it accumulates into.
    /// The gates differ per module (some use <c>useBias</c>, some <c>affine</c>, some only null
    /// checks, some nothing), so the conditional lives here once instead of being re-derived —
    /// and therefore able to drift from the closure — at each site.
    /// </remarks>
    /// <param name="input">The module input, always listed.</param>
    /// <param name="second">First optional tensor, typically the weight. Omitted when null.</param>
    /// <param name="third">Second optional tensor, typically the bias. Omitted when null.</param>
    internal static ReverseGradTensor<T>[] NodeInputs(
        ReverseGradTensor<T> input,
        ReverseGradTensor<T>? second = null,
        ReverseGradTensor<T>? third = null)
    {
        if (second == null)
            return [input];
        if (third == null)
            return [input, second];
        return [input, second, third];
    }
}
