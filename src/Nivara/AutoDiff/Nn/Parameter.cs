using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// A named, trainable parameter wrapping a <see cref="ReverseGradTensor{T}"/>.
/// Parameters are the leaf nodes of the computation graph and the units registered
/// with modules and optimizers.
/// </summary>
public sealed class Parameter<T> : IDisposable where T : struct, IFloatingPointIeee754<T>
{
    ReverseGradTensor<T> tensor;
    long version;
    bool disposed;

    /// <summary>
    /// Gets the parameter name. The name is the key used in module state dictionaries.
    /// </summary>
    public string Name { get; }
    /// <summary>
    /// Gets or sets the underlying tensor. Assigning replaces the tensor and bumps
    /// <see cref="Version"/> so cached derived views are invalidated.
    /// </summary>
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

    /// <summary>
    /// Creates a parameter of <paramref name="size"/> zero-filled elements.
    /// </summary>
    /// <param name="name">The parameter name</param>
    /// <param name="size">The number of elements</param>
    /// <param name="requiresGrad">Whether gradients should be tracked for this parameter</param>
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

    /// <summary>
    /// Creates a parameter wrapping the given tensor.
    /// </summary>
    /// <param name="name">The parameter name</param>
    /// <param name="tensor">The underlying tensor</param>
    public Parameter(string name, ReverseGradTensor<T> tensor)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        this.tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    /// <summary>Gets the number of elements in the parameter.</summary>
    public int Length => Tensor.Length;
    /// <summary>Gets the shape of the parameter tensor.</summary>
    public int[] Shape => Tensor.Shape;
    /// <summary>Gets the rank (dimensionality) of the parameter tensor.</summary>
    public int Rank => Tensor.Rank;

    /// <summary>Returns a string describing the parameter (e.g. <c>Parameter(Weight)</c>).</summary>
    public override string ToString() => $"Parameter({Name})";

    /// <summary>
    /// Releases the underlying tensor.
    /// </summary>
    public void Dispose()
    {
        if (disposed) return;

        tensor.Dispose();
        disposed = true;
    }
}
