using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

internal static class ModuleHelpers<T> where T : struct, INumber<T>
{
    internal static ReadOnlySpan<T> GetSpan(ReverseGradTensor<T> tensor)
    {
        if (tensor.Data.TryGetSpan(out var span))
            return span;
        var arr = new T[tensor.Length];
        tensor.Data.CopyTo(arr, T.Zero);
        return arr;
    }

    internal static T[] CopyToTemp(NivaraColumn<T> column, int length)
    {
        var arr = new T[length];
        column.CopyTo(arr, T.Zero);
        return arr;
    }

    internal static ReverseGradTensor<T> CreateCausalMask(int L)
    {
        var maskData = new T[L * L];
        for (int i = 0; i < L; i++)
            for (int j = 0; j < L; j++)
                if (j > i)
                    maskData[i * L + j] = T.CreateChecked(double.NegativeInfinity);

        var col = NivaraColumn<T>.Create(maskData);
        var tensor = new ReverseGradTensor<T>(col, requiresGrad: false);
        tensor.Reshape(L, L);
        return tensor;
    }

    internal static ReverseGradTensor<T> MultiHeadAttention(
        ReverseGradTensor<T> Q,
        ReverseGradTensor<T> K,
        ReverseGradTensor<T> V,
        int numHeads,
        int headDim,
        ReverseGradTensor<T> scaleTensor,
        ReverseGradTensor<T>? mask)
    {
        var heads = new ReverseGradTensor<T>[numHeads];

        for (int h = 0; h < numHeads; h++)
        {
            int hs = h * headDim;

            var Q_h = ReverseGradOperations.Slice(Q, hs, headDim);
            var K_h = ReverseGradOperations.Slice(K, hs, headDim);
            var V_h = ReverseGradOperations.Slice(V, hs, headDim);

            var K_h_T = ReverseGradOperations.Transpose(K_h);
            var scores = ReverseGradOperations.MatMul(Q_h, K_h_T);

            scores = ReverseGradOperations.Multiply(scores, scaleTensor);

            if (mask != null)
                scores = ReverseGradOperations.Add(scores, mask);

            var weights = ReverseGradOperations.Softmax(scores);

            heads[h] = ReverseGradOperations.MatMul(weights, V_h);
        }

        return ReverseGradOperations.Concat(heads, axis: 1);
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
}
