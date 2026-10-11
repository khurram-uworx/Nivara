using Nivara.Helpers;
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
    readonly Random random;

    /// <summary>
    /// Creates a standard normal initializer (mean 0, std 1). A <paramref name="seed"/> makes
    /// initialization reproducible: the same seed produces identical weights on the same runtime.
    /// A null seed draws from <see cref="Random.Shared"/> (the current default behavior).
    /// </summary>
    /// <param name="seed">Optional RNG seed; when null, draws from <see cref="Random.Shared"/></param>
    /// <remarks>
    /// The seeded form holds its own <see cref="Random"/> and is not thread-safe; the null-seed
    /// default uses <see cref="Random.Shared"/> and is thread-safe. A bare
    /// <c>new NormalInitializer&lt;T&gt;(null)</c> is ambiguous between the <c>int?</c> and
    /// <see cref="Random"/> overloads — pass <c>seed:</c>/<c>random:</c> by name or omit the argument.
    /// </remarks>
    public NormalInitializer(int? seed = null) : this(T.Zero, T.One, seed) { }

    /// <summary>
    /// Creates a standard normal initializer (mean 0, std 1) drawing from the given RNG.
    /// </summary>
    /// <param name="random">The RNG to draw from; must not be null</param>
    /// <remarks>Not thread-safe: the same instance must not be used concurrently.</remarks>
    public NormalInitializer(Random random) : this(T.Zero, T.One, random) { }

    /// <summary>
    /// Creates a normal initializer with the given distribution parameters. A
    /// <paramref name="seed"/> makes initialization reproducible: the same seed produces
    /// identical weights on the same runtime. A null seed draws from
    /// <see cref="Random.Shared"/> (the current default behavior).
    /// </summary>
    /// <param name="mean">The distribution mean</param>
    /// <param name="std">The distribution standard deviation</param>
    /// <param name="seed">Optional RNG seed; when null, draws from <see cref="Random.Shared"/></param>
    /// <remarks>
    /// The seeded form holds its own <see cref="Random"/> and is not thread-safe; the null-seed
    /// default uses <see cref="Random.Shared"/> and is thread-safe. A bare
    /// <c>new NormalInitializer&lt;T&gt;(null)</c> is ambiguous between the <c>int?</c> and
    /// <see cref="Random"/> overloads — pass <c>seed:</c>/<c>random:</c> by name or omit the argument.
    /// </remarks>
    public NormalInitializer(T mean, T std, int? seed = null)
        : this(mean, std, RandomGeneration.Create(seed))
    {
    }

    /// <summary>
    /// Creates a normal initializer with the given distribution parameters, drawing from the
    /// given RNG.
    /// </summary>
    /// <param name="mean">The distribution mean</param>
    /// <param name="std">The distribution standard deviation</param>
    /// <param name="random">The RNG to draw from; must not be null</param>
    /// <remarks>Not thread-safe: the same instance must not be used concurrently.</remarks>
    public NormalInitializer(T mean, T std, Random random)
    {
        this.mean = mean;
        this.std = std;
        this.random = random ?? throw new ArgumentNullException(nameof(random));
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
            var u1 = random.NextDouble();
            var u2 = random.NextDouble();
            var normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            data[i] = T.CreateChecked(normal) * std + mean;
        }

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}
