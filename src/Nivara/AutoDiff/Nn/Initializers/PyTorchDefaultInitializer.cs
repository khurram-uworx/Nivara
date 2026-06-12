using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public sealed class PyTorchDefaultInitializer<T> : IInitializer<T> where T : struct, INumber<T>
{
    public static readonly PyTorchDefaultInitializer<T> Instance = new();

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

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}
