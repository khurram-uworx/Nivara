using Nivara.Diagnostics;
using System.Numerics.Tensors;

namespace Nivara.Storage;

/// <summary>
/// Tensor-backed storage implementation for vectorizable types.
/// Uses System.Numerics.Tensors for high-performance vectorized operations.
/// </summary>
/// <typeparam name="T">The unmanaged type of elements to store</typeparam>
internal sealed class TensorStorage<T> : IColumnStorage<T> where T : unmanaged
{
    readonly Tensor<T> data;
    readonly Tensor<bool>? nullMask;
    T[]? _flattenedCache;
    bool[]? _nullMaskCache;
    bool disposed;

    /// <summary>
    /// Initializes a new instance of TensorStorage with the specified values
    /// </summary>
    /// <param name="values">The values to store</param>
    public TensorStorage(ReadOnlySpan<T> values)
    {
        if (values.IsEmpty)
        {
            data = Tensor.Create<T>(Array.Empty<T>());
            nullMask = null;
        }
        else
        {
            data = Tensor.Create(values.ToArray(), [values.Length]);
            nullMask = null;
        }
    }

    /// <summary>
    /// Initializes a new instance of TensorStorage with nullable values
    /// </summary>
    /// <param name="values">The nullable values to store</param>
    public TensorStorage(ReadOnlySpan<T?> values)
    {
        if (values.IsEmpty)
        {
            data = Tensor.Create<T>(Array.Empty<T>());
            nullMask = null;
            return;
        }

        var dataArray = new T[values.Length];
        var nullMaskArray = new bool[values.Length];
        bool hasAnyNulls = false;

        for (int i = 0; i < values.Length; i++)
        {
            if (values[i].HasValue)
            {
                dataArray[i] = values[i]!.Value;
                nullMaskArray[i] = false;
            }
            else
            {
                dataArray[i] = default(T); // Store default value for null positions
                nullMaskArray[i] = true;
                hasAnyNulls = true;
            }
        }

        data = Tensor.Create(dataArray, [values.Length]);
        nullMask = hasAnyNulls ? Tensor.Create(nullMaskArray, [values.Length]) : null;
    }

    /// <summary>
    /// Initializes a new instance of TensorStorage with existing tensor data and null mask
    /// </summary>
    /// <param name="data">The tensor containing the data</param>
    /// <param name="nullMask">The optional null mask tensor</param>
    internal TensorStorage(Tensor<T> data, Tensor<bool>? nullMask = null)
    {
        this.data = data ?? throw new ArgumentNullException(nameof(data));
        this.nullMask = nullMask;

        // Validate that null mask has same length as data if provided
        if (this.nullMask != null && this.nullMask.FlattenedLength != this.data.FlattenedLength)
        {
            throw new ArgumentException("Null mask length must match data length", nameof(nullMask));
        }
    }

    /// <inheritdoc />
    public int Length => (int)data.FlattenedLength;

    /// <inheritdoc />
    public bool IsVectorizable => true;

    /// <inheritdoc />
    public bool HasNulls => nullMask != null;

    /// <inheritdoc />
    public StorageType StorageType => StorageType.Tensor;

    /// <inheritdoc />
    public ReadOnlySpan<bool> NullMask
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return GetFlattenedNullMask();
        }
    }

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (index < 0 || index >= Length)
                throw new IndexOutOfRangeException($"Index {index} is out of range for storage of length {Length}");

            return GetFlattenedSpan()[index];
        }
    }

    /// <inheritdoc />
    public IColumnStorage<T> Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (start < 0)
            throw new ArgumentOutOfRangeException(nameof(start), "Start index cannot be negative");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length cannot be negative");
        if (start + length > Length)
            throw new ArgumentOutOfRangeException(nameof(length), "Start + length exceeds storage bounds");

        if (length == 0)
        {
            return new TensorStorage<T>(ReadOnlySpan<T>.Empty);
        }

        // Get data as span and slice it
        var slicedDataArray = GetFlattenedSpan().Slice(start, length).ToArray();
        var slicedData = Tensor.Create(slicedDataArray, [length]);

        // Create sliced null mask if it exists
        Tensor<bool>? slicedNullMask = null;
        if (nullMask != null)
        {
            var slicedNullMaskArray = GetFlattenedNullMask().Slice(start, length).ToArray();
            slicedNullMask = Tensor.Create(slicedNullMaskArray, [length]);
        }

        return new TensorStorage<T>(slicedData, slicedNullMask);
    }

    /// <summary>
    /// Gets the underlying data tensor for vectorized operations
    /// </summary>
    internal Tensor<T> Data
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return data;
        }
    }

    /// <summary>
    /// Gets the underlying null mask tensor for vectorized null operations
    /// </summary>
    internal Tensor<bool>? NullMaskTensor
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return nullMask;
        }
    }

    /// <summary>
    /// Gets a tensor span over the underlying tensor data when no null mask is present.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown when the storage contains null values.</exception>
    internal TensorSpan<T> AsTensorSpanIfNoNulls()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (HasNulls)
            throw new InvalidOperationException("Cannot expose TensorSpan for tensor storage with null values.");

        return data.AsTensorSpan();
    }

    /// <summary>
    /// Gets the data as a span, using a cached flattened buffer when available.
    /// </summary>
    internal ReadOnlySpan<T> GetFlattenedSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (_flattenedCache != null)
            return _flattenedCache.AsSpan();

        var buffer = new T[data.FlattenedLength];
        data.FlattenTo(buffer);
        _flattenedCache = buffer;
        return buffer.AsSpan();
    }

    /// <summary>
    /// Gets the null mask as a span, using a cached flattened buffer when available.
    /// </summary>
    internal ReadOnlySpan<bool> GetFlattenedNullMask()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (nullMask == null) return [];

        if (_nullMaskCache != null)
            return _nullMaskCache.AsSpan();

        var buffer = new bool[nullMask.FlattenedLength];
        nullMask.FlattenTo(buffer);
        _nullMaskCache = buffer;
        return buffer.AsSpan();
    }

    /// <inheritdoc />
    ReadOnlySpan<T> IColumnStorage<T>.AsSpan()
    {
        return GetFlattenedSpan();
    }

    /// <inheritdoc />
    Span<T> IColumnStorage<T>.AsWritableSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        var buffer = new T[data.FlattenedLength];
        data.FlattenTo(buffer);
        return buffer.AsSpan();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            _flattenedCache = null;
            _nullMaskCache = null;
            disposed = true;
        }
    }
}
