using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

/// <summary>
/// Mean absolute (L1) error loss: <c>mean / sum |predictions - targets|</c>.
/// </summary>
public sealed class L1Loss<T> : Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>
    /// Creates an L1 loss.
    /// </summary>
    /// <param name="reduction">The default reduction (mean by default)</param>
    public L1Loss(Reduction reduction = Reduction.Mean) : base(reduction)
    {
    }

    /// <summary>
    /// Computes the mean absolute error with an explicit reduction.
    /// </summary>
    /// <param name="predictions">The predicted values</param>
    /// <param name="targets">The target values</param>
    /// <param name="reduction">How to reduce the element-wise loss</param>
    /// <returns>The (possibly reduced) loss</returns>
    public override ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> predictions,
        ReverseGradTensor<T> targets,
        Reduction reduction)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var diff = ReverseGradOperations.Subtract(predictions, targets);
        var abs = ReverseGradOperations.Abs(diff);
        return Reduce(abs, reduction);
    }
}
