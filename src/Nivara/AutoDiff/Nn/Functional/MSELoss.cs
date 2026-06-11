using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class MSELoss<T> where T : struct, INumber<T>
{
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var diff = GradOperations.Subtract(predictions, targets);
        var squared = GradOperations.Multiply(diff, diff);
        return GradOperations.Sum(squared);
    }
}
