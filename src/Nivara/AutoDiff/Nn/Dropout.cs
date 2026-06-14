using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Dropout<T> : Module<T> where T : struct, INumber<T>
{
    readonly double probability;

    public double Probability => probability;

    public Dropout(double probability = 0.5)
    {
        if (probability < 0.0 || probability >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(probability), "Dropout probability must be in [0, 1).");
        this.probability = probability;
    }

    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        return ReverseGradOperations.Dropout(input, probability, IsTraining);
    }
}
