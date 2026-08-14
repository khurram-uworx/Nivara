using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// Initializes parameters with values drawn uniformly from <c>[lower, upper]</c>.
/// </summary>
public sealed class UniformInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly T lower;
    readonly T upper;

    /// <summary>Creates a uniform initializer over <c>[-1, 1]</c>.</summary>
    public UniformInitializer() : this(-T.One, T.One) { }

    /// <summary>
    /// Creates a uniform initializer over the given range.
    /// </summary>
    /// <param name="lower">The inclusive lower bound</param>
    /// <param name="upper">The inclusive upper bound</param>
    public UniformInitializer(T lower, T upper)
    {
        this.lower = lower;
        this.upper = upper;
    }

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var range = upper - lower;
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble()) * range + lower;

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}
