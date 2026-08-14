using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// PyTorch-compatible default initialization for linear layers: values uniform in
/// <c>[-bound, bound]</c> with <c>bound = 1 / sqrt(fanIn)</c>, where fanIn is the second
/// shape dimension. No-op for tensors with fewer than two dimensions.
/// </summary>
public sealed class PyTorchDefaultInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>A shared singleton instance.</summary>
    public static readonly PyTorchDefaultInitializer<T> Instance = new();

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var bound = T.CreateChecked(1.0 / Math.Sqrt(fanIn));
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}
