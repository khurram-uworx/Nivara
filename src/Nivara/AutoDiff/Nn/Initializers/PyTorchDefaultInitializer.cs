using Nivara.Helpers;
using System.Numerics;

namespace Nivara.AutoDiff.Nn.Initializers;

/// <summary>
/// PyTorch-compatible default initialization for linear layers: values uniform in
/// <c>[-bound, bound]</c> with <c>bound = 1 / sqrt(fanIn)</c>, where fanIn is the second
/// shape dimension. No-op for tensors with fewer than two dimensions.
/// </summary>
public sealed class PyTorchDefaultInitializer<T> : IInitializer<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>A shared singleton instance drawing from <see cref="Random.Shared"/>.</summary>
    public static readonly PyTorchDefaultInitializer<T> Instance = new();

    readonly Random random;

    /// <summary>
    /// Creates a PyTorch default initializer. A <paramref name="seed"/> makes initialization
    /// reproducible: the same seed produces identical weights on the same runtime. A null seed
    /// draws from <see cref="Random.Shared"/> (the current default behavior).
    /// </summary>
    /// <param name="seed">Optional RNG seed; when null, draws from <see cref="Random.Shared"/></param>
    /// <remarks>
    /// The seeded form holds its own <see cref="Random"/> and is not thread-safe; the null-seed
    /// default uses <see cref="Random.Shared"/> and is thread-safe.
    /// </remarks>
    public PyTorchDefaultInitializer(int? seed = null)
        : this(RandomGeneration.Create(seed))
    {
    }

    /// <summary>
    /// Creates a PyTorch default initializer drawing from the given RNG.
    /// </summary>
    /// <param name="random">The RNG to draw from; must not be null</param>
    /// <remarks>Not thread-safe: the same instance must not be used concurrently.</remarks>
    public PyTorchDefaultInitializer(Random random)
    {
        this.random = random ?? throw new ArgumentNullException(nameof(random));
    }

    /// <summary>Initializes the parameter in place.</summary>
    /// <param name="parameter">The parameter to initialize</param>
    public void Initialize(Parameter<T> parameter)
    {
        var tensor = parameter.Tensor;
        var shape = tensor.Shape;
        if (shape.Length < 2) return;

        var fanIn = shape[1];
        var bound = T.CreateChecked(1.0 / Math.Sqrt(fanIn));
        var n = tensor.Length;
        var data = new T[n];

        for (int i = 0; i < n; i++)
            data[i] = T.CreateChecked(random.NextDouble() * 2.0 - 1.0) * bound;

        var column = NivaraColumn<T>.CreateFromOwnedArray(data);
        parameter.Tensor = new ReverseGradTensor<T>(column, tensor.RequiresGrad, shape);
    }
}