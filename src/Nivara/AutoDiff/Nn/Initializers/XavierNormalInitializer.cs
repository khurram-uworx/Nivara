using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public sealed class XavierNormalInitializer<T> : IInitializer<T> where T : struct, INumber<T>
{
    public static readonly XavierNormalInitializer<T> Instance = new();

    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var fanOut = shape[0];
        var std = T.CreateChecked(Math.Sqrt(2.0 / (fanIn + fanOut)));
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
        {
            var u1 = Random.Shared.NextDouble();
            var u2 = Random.Shared.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = T.CreateChecked(normal) * std;
        }

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}
