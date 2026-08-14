using Nivara.AutoDiff.Operations;
using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// Randomly zeroes a fraction of input elements during training and rescales the
/// remaining elements by <c>1 / (1 - p)</c>. In evaluation mode the input is returned
/// unchanged.
/// </summary>
public sealed class Dropout<T> : Module<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly double probability;

    /// <summary>Gets the dropout probability <c>p</c> in <c>[0, 1)</c>.</summary>
    public double Probability => probability;

    /// <summary>
    /// Creates a dropout layer.
    /// </summary>
    /// <param name="probability">The dropout probability <c>p</c> in <c>[0, 1)</c></param>
    public Dropout(double probability = 0.5)
    {
        if (probability < 0.0 || probability >= 1.0)
            throw new ArgumentOutOfRangeException(nameof(probability), "Dropout probability must be in [0, 1).");
        this.probability = probability;
    }

    /// <summary>
    /// Applies dropout when the module is in training mode; otherwise returns the input unchanged.
    /// </summary>
    /// <param name="input">The input tensor</param>
    /// <returns>The dropped-out (or unchanged) tensor</returns>
    public override ReverseGradTensor<T> Forward(ReverseGradTensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));

        return ReverseGradOperations.Dropout(input, probability, IsTraining);
    }
}
