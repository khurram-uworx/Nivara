using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// Xavier/Glorot uniform initializer: values uniform in <c>[-bound, bound]</c> with
/// <c>bound = sqrt(6 / (fanIn + fanOut))</c>, where fanIn/fanOut are the first two shape
/// dimensions. No-op for tensors with fewer than two dimensions.
/// </summary>
public sealed class XavierUniformInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>A shared singleton instance.</summary>
    public static readonly XavierUniformInitializer<T> Instance = new();

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var fanOut = shape[0];
        var bound = T.CreateChecked(Math.Sqrt(6.0 / (fanIn + fanOut)));
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}
