using System.Numerics;

namespace Nivara.AutoDiff.Nn;

public sealed class Parameter<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    ReverseGradTensor<T> tensor;
    long version;
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
            version++;
        }
    }

    /// <summary>
    /// Monotonic version stamp incremented whenever <see cref="Tensor"/> is
    /// replaced. Modules cache derived views of the parameter (e.g. transposed
    /// weights) and invalidate them on version change.
    /// </summary>
    public long Version
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return version;
        }
    }

    /// <summary>
    /// Invalidates the version stamp without replacing <see cref="Tensor"/>.
    /// Call after mutating the tensor's underlying data in place.
    /// </summary>
    public void Touch()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        version++;
    }

    public Parameter(string name, int size, bool requiresGrad = true)
        : this(name, new T[size], requiresGrad)
    {
    }

    /// <summary>
    /// Creates a parameter wrapping <paramref name="data"/> zero-copy.
    /// The caller must not mutate <paramref name="data"/> afterward — the parameter's
    /// tensor shares the array, so mutating it corrupts the parameter.
    /// </summary>
    /// <param name="name">The parameter name</param>
    /// <param name="data">The initial values; ownership transfers to the parameter</param>
    /// <param name="requiresGrad">Whether gradients should be tracked for this parameter</param>
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
