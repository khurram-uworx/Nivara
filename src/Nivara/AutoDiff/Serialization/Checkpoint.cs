using System.Numerics;

namespace Nivara.AutoDiff.Serialization;

/// <summary>
/// A snapshot of a model and optimizer state at a point in training, produced by
/// <see cref="ModelSerializer.LoadCheckpoint{T}"/>.
/// </summary>
public sealed class Checkpoint<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>The epoch recorded at save time.</summary>
    public int Epoch { get; init; }

    /// <summary>The loss recorded at save time.</summary>
    public double Loss { get; init; }

    /// <summary>The model parameters keyed by name.</summary>
    public IReadOnlyDictionary<string, ParameterData<T>> Parameters { get; init; }
        = new Dictionary<string, ParameterData<T>>();

    /// <summary>The optimizer state buffers keyed by name.</summary>
    public IReadOnlyDictionary<string, T[]> OptimizerState { get; init; }
        = new Dictionary<string, T[]>();
}

/// <summary>Serialized shape and values for a single model parameter.</summary>
public sealed class ParameterData<T> where T : struct, IFloatingPointIeee754<T>
{
    /// <summary>The parameter's tensor shape.</summary>
    public int[] Shape { get; init; } = [];

    /// <summary>The parameter's element values.</summary>
    public T[] Values { get; init; } = [];
}
