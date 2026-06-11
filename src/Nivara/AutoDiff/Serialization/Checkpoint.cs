using System.Numerics;

namespace Nivara.AutoDiff.Serialization;

public sealed class Checkpoint<T> where T : struct, INumber<T>
{
    public int Epoch { get; init; }
    public double Loss { get; init; }
    public IReadOnlyDictionary<string, ParameterData<T>> Parameters { get; init; }
        = new Dictionary<string, ParameterData<T>>();
}

public sealed class ParameterData<T> where T : struct, INumber<T>
{
    public int[] Shape { get; init; } = [];
    public T[] Values { get; init; } = [];
    public bool[]? NullMask { get; init; }
}
