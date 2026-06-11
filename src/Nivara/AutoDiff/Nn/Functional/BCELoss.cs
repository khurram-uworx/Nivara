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

        var clamped = GradOperations.Clip(predictions, eps, T.One - eps);
        var one = new ReverseGradTensor<T>(NivaraColumn<T>.Create([T.One]),
            requiresGrad: false, predictions.shape);
        var logPred = GradOperations.Log(clamped);
        var log1mPred = GradOperations.Log(GradOperations.Subtract(one, clamped));
        var loss = GradOperations.Negate(GradOperations.Add(
            GradOperations.Multiply(targets, logPred),
            GradOperations.Multiply(GradOperations.Subtract(one, targets), log1mPred)));
        return GradOperations.Sum(loss);
    }
}
