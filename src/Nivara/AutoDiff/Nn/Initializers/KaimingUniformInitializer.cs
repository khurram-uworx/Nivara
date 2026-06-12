using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public sealed class KaimingUniformInitializer<T> : IInitializer<T> where T : struct, INumber<T>
{
    public static readonly KaimingUniformInitializer<T> Instance = new();

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

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}
