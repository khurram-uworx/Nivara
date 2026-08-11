using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class L1Loss<T> : Loss<T> where T : struct, IFloatingPointIeee754<T>
{
    public L1Loss(Reduction reduction = Reduction.Mean) : base(reduction)
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
        var abs = ReverseGradOperations.Abs(diff);
        return Reduce(abs, reduction);
    }
}
