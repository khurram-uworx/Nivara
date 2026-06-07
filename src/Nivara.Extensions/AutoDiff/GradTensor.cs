using Nivara.Extensions.AutoDiff.Utilities;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Extensions.AutoDiff;

/// <summary>
/// Base tensor wrapper that holds data and shape metadata.
/// Serves as the shared foundation for all automatic differentiation flavors
/// (reverse-mode, forward-mode, etc.).
/// </summary>
/// <typeparam name="T">The numeric type that implements INumber&lt;T&gt;</typeparam>
public class GradTensor<T> : IDisposable where T : struct, INumber<T>
{
    internal bool disposed;
    internal int[] shape;

    /// <summary>
    /// Gets the underlying data as a NivaraColumn&lt;T&gt;
    /// </summary>
    public NivaraColumn<T> Data { get; }

    /// <summary>
    /// Gets the number of elements in the tensor
    /// </summary>
    public int Length => Data.Length;

    /// <summary>
    /// Gets a value indicating whether this tensor contains any null values
    /// </summary>
    public bool HasNulls => Data.HasNulls;

    /// <summary>
    /// Gets the shape of the tensor as a read-only copy of dimension sizes.
    /// Default is 1D: [Length].
    /// Use <see cref="Reshape"/> to set matrix dimensions for MatMul/Transpose.
    /// </summary>
    public int[] Shape => (int[])shape.Clone();

    /// <summary>
    /// Gets the number of dimensions (rank) of the tensor.
    /// 1 for vectors, 2 for matrices, etc.
    /// </summary>
    public int Rank => shape.Length;

    /// <summary>
    /// Initializes a new instance of GradTensor with the specified data.
    /// Shape defaults to 1D: [data.Length].
    /// </summary>
    /// <param name="data">The underlying data as a NivaraColumn&lt;T&gt;</param>
    /// <exception cref="ArgumentNullException">Thrown when data is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type for automatic differentiation</exception>
    protected GradTensor(NivaraColumn<T> data)
    {
        TypeValidator.ValidateNumericType<T>();
        Data = data ?? throw new ArgumentNullException(nameof(data));
        shape = new[] { data.Length };
    }

    /// <summary>
    /// Initializes a new instance of GradTensor with shape metadata.
    /// </summary>
    internal GradTensor(NivaraColumn<T> data, int[] shape)
    {
        TypeValidator.ValidateNumericType<T>();
        Data = data ?? throw new ArgumentNullException(nameof(data));
        this.shape = shape ?? throw new ArgumentNullException(nameof(shape));
    }

    /// <summary>
    /// Reshapes the tensor dimensions. The product of all dimensions must equal <see cref="Length"/>.
    /// This only changes metadata; the underlying flat data is unchanged.
    /// </summary>
    /// <param name="dims">The new dimension sizes</param>
    /// <exception cref="ArgumentException">Thrown when the product of dims doesn't match Length</exception>
    public void Reshape(params int[] dims)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (dims == null || dims.Length == 0)
            throw new ArgumentException("Shape must have at least one dimension", nameof(dims));

        int total = 1;
        foreach (var d in dims)
        {
            if (d <= 0)
                throw new ArgumentException($"All dimensions must be positive, got {d}", nameof(dims));
            total *= d;
        }

        if (total != Length)
            throw new ArgumentException(
                $"New shape ({string.Join(", ", dims)}) has {total} elements but tensor has {Length} elements",
                nameof(dims));

        shape = (int[])dims.Clone();
    }

    /// <summary>
    /// Converts this GradTensor back to a NivaraColumn&lt;T&gt;
    /// </summary>
    /// <returns>The underlying NivaraColumn&lt;T&gt;</returns>
    public NivaraColumn<T> ToColumn()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return Data;
    }

    /// <summary>
    /// Converts this GradTensor to a NivaraSeries&lt;T&gt;
    /// </summary>
    /// <returns>A new NivaraSeries&lt;T&gt; wrapping the data</returns>
    public NivaraSeries<T> ToSeries()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new NivaraSeries<T>(Data);
    }

    /// <summary>
    /// Gets a tensor view of the data for operations requiring tensor semantics
    /// </summary>
    /// <returns>A Tensor&lt;T&gt; view of the data</returns>
    /// <exception cref="InvalidOperationException">Thrown when the data contains null values</exception>
    public Tensor<T> AsTensor()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (HasNulls)
        {
            throw new InvalidOperationException("Cannot create tensor view from data with null values. Use ToColumn() or ToSeries() for null-aware operations.");
        }

        var series = ToSeries();
        return series.ToTensor();
    }

    /// <summary>
    /// Gets the element at the specified index
    /// </summary>
    /// <param name="index">The zero-based index</param>
    /// <returns>The element at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return Data[index];
        }
    }

    /// <summary>
    /// Determines whether the element at the specified index is null
    /// </summary>
    /// <param name="index">The zero-based index to check</param>
    /// <returns>true if the element is null; otherwise, false</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    public bool IsNull(int index)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return Data.IsNull(index);
    }

    /// <summary>
    /// Creates a string representation of this GradTensor
    /// </summary>
    /// <returns>A string representation showing data and shape info</returns>
    public override string ToString()
    {
        if (disposed)
            return "GradTensor<T> (disposed)";

        var shapeStr = string.Join("x", shape);
        return $"GradTensor<{typeof(T).Name}>[{shapeStr}]";
    }

    /// <summary>
    /// Releases all resources used by this GradTensor
    /// </summary>
    public void Dispose()
    {
        if (!disposed)
        {
            Data?.Dispose();
            disposed = true;
        }
    }
}
