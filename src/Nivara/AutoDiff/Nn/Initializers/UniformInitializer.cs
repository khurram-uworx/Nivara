using Nivara.Helpers;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// Initializes parameters with values drawn uniformly from <c>[lower, upper]</c>.
/// </summary>
public sealed class UniformInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    readonly T lower;
    readonly T upper;
    readonly Random random;

    /// <summary>
    /// Creates a uniform initializer over <c>[-1, 1]</c>. A <paramref name="seed"/> makes
    /// initialization reproducible: the same seed produces identical weights on the same runtime.
    /// A null seed draws from <see cref="Random.Shared"/> (the current default behavior).
    /// </summary>
    /// <param name="seed">Optional RNG seed; when null, draws from <see cref="Random.Shared"/></param>
    /// <remarks>
    /// The seeded form holds its own <see cref="Random"/> and is not thread-safe; the null-seed
    /// default uses <see cref="Random.Shared"/> and is thread-safe. A bare
    /// <c>new UniformInitializer&lt;T&gt;(null)</c> is ambiguous between the <c>int?</c> and
    /// <see cref="Random"/> overloads — pass <c>seed:</c>/<c>random:</c> by name or omit the argument.
    /// </remarks>
    public UniformInitializer(int? seed = null) : this(-T.One, T.One, seed) { }

    /// <summary>
    /// Creates a uniform initializer over <c>[-1, 1]</c> drawing from the given RNG.
    /// </summary>
    /// <param name="random">The RNG to draw from; must not be null</param>
    /// <remarks>Not thread-safe: the same instance must not be used concurrently.</remarks>
    public UniformInitializer(Random random) : this(-T.One, T.One, random) { }

    /// <summary>
    /// Creates a uniform initializer over the given range. A <paramref name="seed"/> makes
    /// initialization reproducible: the same seed produces identical weights on the same runtime.
    /// A null seed draws from <see cref="Random.Shared"/> (the current default behavior).
    /// </summary>
    /// <param name="lower">The inclusive lower bound</param>
    /// <param name="upper">The inclusive upper bound</param>
    /// <param name="seed">Optional RNG seed; when null, draws from <see cref="Random.Shared"/></param>
    /// <remarks>
    /// The seeded form holds its own <see cref="Random"/> and is not thread-safe; the null-seed
    /// default uses <see cref="Random.Shared"/> and is thread-safe. A bare
    /// <c>new UniformInitializer&lt;T&gt;(null)</c> is ambiguous between the <c>int?</c> and
    /// <see cref="Random"/> overloads — pass <c>seed:</c>/<c>random:</c> by name or omit the argument.
    /// </remarks>
    public UniformInitializer(T lower, T upper, int? seed = null)
        : this(lower, upper, RandomGeneration.Create(seed))
    {
    }

    /// <summary>
    /// Creates a uniform initializer over the given range, drawing from the given RNG.
    /// </summary>
    /// <param name="lower">The inclusive lower bound</param>
    /// <param name="upper">The inclusive upper bound</param>
    /// <param name="random">The RNG to draw from; must not be null</param>
    /// <remarks>Not thread-safe: the same instance must not be used concurrently.</remarks>
    public UniformInitializer(T lower, T upper, Random random)
    {
        this.lower = lower;
        this.upper = upper;
        this.random = random ?? throw new ArgumentNullException(nameof(random));
    }

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var range = upper - lower;
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble()) * range + lower;

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, tensor.Shape);
    }
}