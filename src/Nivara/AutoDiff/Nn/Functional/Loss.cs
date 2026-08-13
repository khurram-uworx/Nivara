using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public abstract class Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    public Reduction Reduction { get; }

    protected Loss(Reduction reduction)
    {
        if (!Enum.IsDefined(reduction))
            throw new ArgumentOutOfRangeException(nameof(reduction), $"Unknown Reduction value: {reduction}.");
        Reduction = reduction;
    }

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
        => Forward(predictions, targets, Reduction);

    // Subclasses override only this 3-arg overload; the 2-arg Forward above is
    // non-virtual and delegates with the stored Reduction so both paths share
    // one Reduce call chain. Do not override the 2-arg overload.
    public abstract ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> predictions,
        ReverseGradTensor<T> targets,
        Reduction reduction);

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
                var scale = GradientUtils.Full(1, T.CreateChecked(count));
                return ReverseGradOperations.Divide(ReverseGradOperations.Sum(elementwiseLoss), scale);

            default:
                throw new ArgumentOutOfRangeException(nameof(reduction), $"Unknown Reduction value: {reduction}.");
        }
    }
}
