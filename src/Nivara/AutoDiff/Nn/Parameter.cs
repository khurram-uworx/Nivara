using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Parameter<T> : IDisposable where T : struct, INumber<T>
{
    ReverseGradTensor<T> tensor;
    bool disposed;

    public string Name { get; }
    public ReverseGradTensor<T> Tensor
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return tensor;
        }
        set
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            tensor = value ?? throw new ArgumentNullException(nameof(value));
        }
    }

    public Parameter(string name, int size, bool requiresGrad = true)
        : this(name, new T[size], requiresGrad)
    {
    }

    public Parameter(string name, T[] data, bool requiresGrad = true)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        tensor = ReverseGradTensor<T>.FromArray(data, requiresGrad);
    }

    public Parameter(string name, ReverseGradTensor<T> tensor)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        this.tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    public int Length => Tensor.Length;
    public int[] Shape => Tensor.Shape;
    public int Rank => Tensor.Rank;

    public override string ToString() => $"Parameter({Name})";

    public void Dispose()
    {
        if (disposed) return;

        tensor.Dispose();
        disposed = true;
    }
}
