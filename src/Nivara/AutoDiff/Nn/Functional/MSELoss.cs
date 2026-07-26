using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class MSELoss<T> where T : struct, INumber<T>
{
    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets)
        => Forward(predictions, targets, reduceToMean: false);

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> predictions, ReverseGradTensor<T> targets, bool reduceToMean)
    {
        if (predictions == null) throw new ArgumentNullException(nameof(predictions));
        if (targets == null) throw new ArgumentNullException(nameof(targets));

        var diff = ReverseGradOperations.Subtract(predictions, targets);
        var squared = ReverseGradOperations.Multiply(diff, diff);
        var sumLoss = ReverseGradOperations.Sum(squared);

        if (!reduceToMean)
            return sumLoss;

        var lengthTensor = GradientUtils.Full(1, T.CreateChecked(predictions.Length));
        return ReverseGradOperations.Divide(sumLoss, lengthTensor);
    }
}
