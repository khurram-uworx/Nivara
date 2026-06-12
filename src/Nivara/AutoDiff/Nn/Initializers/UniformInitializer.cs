using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

public sealed class UniformInitializer<T> : IInitializer<T> where T : struct, INumber<T>
{
    readonly T lower;
    readonly T upper;

    public UniformInitializer() : this(-T.One, T.One) { }

    public UniformInitializer(T lower, T upper)
    {
        this.lower = lower;
        this.upper = upper;
    }

    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var range = upper - lower;
        var random = Random.Shared;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble()) * range + lower;

        var column = NivaraColumn<T>.Create(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}
