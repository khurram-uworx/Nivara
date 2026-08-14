using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// Kaiming/He uniform initializer: values uniform in <c>[-bound, bound]</c> with
/// <c>bound = sqrt(6 / fanIn)</c>, where fanIn is the second shape dimension.
/// No-op for tensors with fewer than two dimensions.
/// </summary>
public sealed class KaimingUniformInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>A shared singleton instance.</summary>
    public static readonly KaimingUniformInitializer<T> Instance = new();

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var bound = T.CreateChecked(Math.Sqrt(6.0 / fanIn));
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}
