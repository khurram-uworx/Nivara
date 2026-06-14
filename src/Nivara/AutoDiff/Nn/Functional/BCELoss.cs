using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class BCELoss<T> where T : struct, INumber<T>
{
    readonly T eps;

    public BCELoss(double eps = 1e-7)
    {
        this.eps = T.CreateChecked(eps);
    }

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var clamped = ReverseGradOperations.Clip(predictions, eps, T.One - eps);
        var one = new ReverseGradTensor<T>(NivaraColumn<T>.Create([T.One]),
            requiresGrad: false, predictions.shape);
        var logPred = ReverseGradOperations.Log(clamped);
        var log1mPred = ReverseGradOperations.Log(ReverseGradOperations.Subtract(one, clamped));
        var loss = ReverseGradOperations.Negate(ReverseGradOperations.Add(
            ReverseGradOperations.Multiply(targets, logPred),
            ReverseGradOperations.Multiply(ReverseGradOperations.Subtract(one, targets), log1mPred)));
        return ReverseGradOperations.Sum(loss);
    }
}
