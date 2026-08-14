using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// Initializes parameters with values drawn from a Gaussian distribution with the given
/// mean and standard deviation (via the Box-Muller transform).
/// </summary>
public sealed class NormalInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly T mean;
    readonly T std;

    /// <summary>Creates a standard normal initializer (mean 0, std 1).</summary>
    public NormalInitializer() : this(T.Zero, T.One) { }

    /// <summary>
    /// Creates a normal initializer with the given distribution parameters.
    /// </summary>
    /// <param name="mean">The distribution mean</param>
    /// <param name="std">The distribution standard deviation</param>
    public NormalInitializer(T mean, T std)
    {
        this.mean = mean;
        this.std = std;
    }

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
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

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}
