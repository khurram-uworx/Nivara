using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Functional;

public sealed class LogSoftmax<T> where T : struct, INumber<T>
{
    readonly int dim;

    public LogSoftmax(int dim = -1)
    {
        this.dim = dim;
    }

    public ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        return GradOperations.LogSoftmax(input);
    }
}
