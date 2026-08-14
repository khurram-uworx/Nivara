using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

/// <summary>
/// Binary cross-entropy loss: <c>-(targets·log(pred) + (1-targets)·log(1-pred))</c>.
/// Predictions are clamped to <c>[eps, 1-eps]</c> before taking logs.
/// </summary>
public sealed class BCELoss<T> : Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly T eps;

    /// <summary>
    /// Creates a binary cross-entropy loss.
    /// </summary>
    /// <param name="reduction">The default reduction (mean by default)</param>
    /// <param name="eps">Clamping value used to keep log arguments away from zero</param>
    public BCELoss(Reduction reduction = Reduction.Mean, double eps = 1e-7)
        : base(reduction)
    {
        this.eps = T.CreateChecked(eps);
    }

    /// <summary>
    /// Computes the binary cross-entropy with an explicit reduction.
    /// </summary>
    /// <param name="predictions">The predicted probabilities</param>
    /// <param name="targets">The binary targets</param>
    /// <param name="reduction">How to reduce the element-wise loss</param>
    /// <returns>The (possibly reduced) loss</returns>
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
