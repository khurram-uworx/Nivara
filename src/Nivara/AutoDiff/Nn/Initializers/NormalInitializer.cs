using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public sealed class NormalInitializer<T> : IInitializer<T> where T : struct, INumber<T>
{
    readonly T mean;
    readonly T std;

    public NormalInitializer() : this(T.Zero, T.One) { }

    public NormalInitializer(T mean, T std)
    {
        this.mean = mean;
        this.std = std;
    }

    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
        {
            var u1 = Random.Shared.NextDouble();
            var u2 = Random.Shared.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = T.CreateChecked(normal) * std + mean;
        }

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}
