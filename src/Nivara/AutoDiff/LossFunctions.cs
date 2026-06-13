using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Buffers;
using System.Numerics;

namespace Nivara.AutoDiff;

public static class LossFunctions
{
    public static ReverseGradTensor<T> MSE<T>(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
        where T : struct, INumber<T>
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var diff = GradOperations.Subtract(predictions, targets);
        var squared = GradOperations.Multiply(diff, diff);
        return GradOperations.Sum(squared);
    }

    public static ReverseGradTensor<T> L1<T>(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
        where T : struct, INumber<T>
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var diff = GradOperations.Subtract(predictions, targets);
        var abs = GradOperations.Abs(diff);
        return GradOperations.Sum(abs);
    }

    public static ReverseGradTensor<T> BCE<T>(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets, double eps = 1e-7)
        where T : struct, INumber<T>
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var epsT = T.CreateChecked(eps);
        var clamped = GradOperations.Clip(predictions, epsT, T.One - epsT);
        var one = new ReverseGradTensor<T>(NivaraColumn<T>.Create([T.One]),
            requiresGrad: false, predictions.shape);
        var logPred = GradOperations.Log(clamped);
        var log1mPred = GradOperations.Log(GradOperations.Subtract(one, clamped));
        var loss = GradOperations.Negate(GradOperations.Add(
            GradOperations.Multiply(targets, logPred),
            GradOperations.Multiply(GradOperations.Subtract(one, targets), log1mPred)));
        return GradOperations.Sum(loss);
    }

    public static ReverseGradTensor<T> BCEWithLogits<T>(ReverseGradTensor<T> logits, ReverseGradTensor<T> targets)
        where T : struct, INumber<T>
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        int n = logits.Length;
        var oneBuf = ArrayPool<T>.Shared.Rent(n);
        NivaraColumn<T> oneCol;
        try
        {
            oneBuf.AsSpan(0, n).Fill(T.One);
            oneCol = NivaraColumn<T>.Create(oneBuf.AsSpan(0, n));
        }
        finally
        {
            ArrayPool<T>.Shared.Return(oneBuf, clearArray: true);
        }
        var one = new ReverseGradTensor<T>(oneCol,
            requiresGrad: false, logits.shape);

        var maxX = GradOperations.Relu(logits);
        var negAbs = GradOperations.Negate(GradOperations.Abs(logits));
        var negExpNegAbs = GradOperations.Exp(negAbs);
        var log1pExp = GradOperations.Log(GradOperations.Add(one, negExpNegAbs));

        var loss = GradOperations.Add(
            GradOperations.Subtract(maxX, GradOperations.Multiply(logits, targets)),
            log1pExp);
        return GradOperations.Sum(loss);
    }

    public static ReverseGradTensor<T> CrossEntropy<T>(ReverseGradTensor<T> logits, ReverseGradTensor<T> targets)
        where T : struct, INumber<T>
    {
        if (logits == null) throw new ArgumentNullException(nameof(logits));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var logSoftmax = GradOperations.LogSoftmax(logits);
        var nll = GradOperations.Negate(GradOperations.Sum(GradOperations.Multiply(logSoftmax, targets)));
        var batchSize = T.CreateChecked(logits.shape[0]);
        var scaleTensor = GradientUtils.Full(logits.Length, batchSize);
        return GradOperations.Divide(nll, scaleTensor);
    }
}
