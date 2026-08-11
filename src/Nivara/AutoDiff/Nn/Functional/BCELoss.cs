using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class BCELoss<T> : Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly T eps;

    public BCELoss(Reduction reduction = Reduction.Mean, double eps = 1e-7)
        : base(reduction)
    {
        this.eps = T.CreateChecked(eps);
    }

    public override ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> predictions,
        ReverseGradTensor<T> targets,
        Reduction reduction)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var clamped = ReverseGradOperations.Clip(predictions, eps, T.One - eps);
        var one = GradientUtils.Full(predictions.Length, T.One);
        one.Reshape(predictions.shape);
        var logPred = ReverseGradOperations.Log(clamped);
        var log1mPred = ReverseGradOperations.Log(ReverseGradOperations.Subtract(one, clamped));
        var loss = ReverseGradOperations.Negate(ReverseGradOperations.Add(
            ReverseGradOperations.Multiply(targets, logPred),
            ReverseGradOperations.Multiply(ReverseGradOperations.Subtract(one, targets), log1mPred)));
        return Reduce(loss, reduction);
    }
}
