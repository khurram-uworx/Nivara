using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

/// <summary>
/// Abstract base class for losses. Stores a default <see cref="Reduction"/> (mean, matching
/// PyTorch) and centralizes the Sum/Mean/None reduction logic in <see cref="Reduce"/>.
/// Subclasses implement the three-argument <see cref="Forward(ReverseGradTensor{T}, ReverseGradTensor{T}, Reduction)"/>.
/// </summary>
public abstract class Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>Gets the default reduction applied by the two-argument Forward.</summary>
    public Reduction Reduction { get; }

    /// <summary>
    /// Creates a loss with a default reduction.
    /// </summary>
    /// <param name="reduction">The default reduction for the two-argument Forward</param>
    protected Loss(Reduction reduction)
    {
        if (!Enum.IsDefined(reduction))
            throw new ArgumentOutOfRangeException(nameof(reduction), $"Unknown Reduction value: {reduction}.");
        Reduction = reduction;
    }

    /// <summary>
    /// Computes the loss using the stored default reduction.
    /// </summary>
    /// <param name="predictions">The predicted values</param>
    /// <param name="targets">The target values</param>
    /// <returns>The reduced loss</returns>
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
        => Forward(predictions, targets, Reduction);

    // Subclasses override only this 3-arg overload; the 2-arg Forward above is
    // non-virtual and delegates with the stored Reduction so both paths share
    // one Reduce call chain. Do not override the 2-arg overload.
    /// <summary>
    /// Computes the loss with an explicit reduction.
    /// </summary>
    /// <param name="predictions">The predicted values</param>
    /// <param name="targets">The target values</param>
    /// <param name="reduction">How to reduce the element-wise loss</param>
    /// <returns>The (possibly reduced) loss</returns>
    public abstract ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> predictions,
        ReverseGradTensor<T> targets,
        Reduction reduction);

    /// <summary>
    /// Applies a reduction to an element-wise loss. Sum and Mean return a scalar; None
    /// returns the input unchanged. Mean divides by <paramref name="divisor"/> when provided,
    /// otherwise by the element count.
    /// </summary>
    /// <param name="elementwiseLoss">The element-wise loss tensor</param>
    /// <param name="reduction">How to reduce the loss</param>
    /// <param name="divisor">Optional divisor for Mean (defaults to the element count)</param>
    /// <returns>The reduced loss</returns>
    protected static ReverseGradTensor<T> Reduce(
        ReverseGradTensor<T> elementwiseLoss,
        Reduction reduction,
        int? divisor = null)
    {
        if (elementwiseLoss == null) throw new ArgumentNullException(nameof(elementwiseLoss));

        switch (reduction)
        {
            case Reduction.None:
                return elementwiseLoss;

            case Reduction.Sum:
                return ReverseGradOperations.Sum(elementwiseLoss);

            case Reduction.Mean:
                int count = divisor ?? elementwiseLoss.Length;
                return ReverseGradOperations.DivideScalar(
                    ReverseGradOperations.Sum(elementwiseLoss), T.CreateChecked(count));

            default:
                throw new ArgumentOutOfRangeException(nameof(reduction), $"Unknown Reduction value: {reduction}.");
        }
    }
}
