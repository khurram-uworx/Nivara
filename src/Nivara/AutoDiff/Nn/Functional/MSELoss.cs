using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class MSELoss<T> : Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    public MSELoss(Reduction reduction = Reduction.Mean) : base(reduction)
    {
    }

    public override ReverseGradTensor<T> Forward(
        ReverseGradTensor<T> predictions,
        ReverseGradTensor<T> targets,
        Reduction reduction)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var diff = ReverseGradOperations.Subtract(predictions, targets);
        var squared = ReverseGradOperations.Multiply(diff, diff);
        return Reduce(squared, reduction);
    }
}
