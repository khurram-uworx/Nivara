using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Parameter<T> where T : struct, INumber<T>
{
    public string Name { get; }
    public ReverseGradTensor<T> Tensor { get; internal set; }

    public Parameter(string name, int size, bool requiresGrad = true)
        : this(name, new T[size], requiresGrad)
    {
    }

    public Parameter(string name, T[] data, bool requiresGrad = true)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Tensor = ReverseGradTensor<T>.FromArray(data, requiresGrad);
    }

    public Parameter(string name, ReverseGradTensor<T> tensor)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    public int Length => Tensor.Length;
    public int[] Shape => Tensor.Shape;
    public int Rank => Tensor.Rank;

    public override string ToString() => $"Parameter({Name})";
}
