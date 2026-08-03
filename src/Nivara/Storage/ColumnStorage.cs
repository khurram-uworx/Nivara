using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Nivara.Diagnostics;

namespace Nivara.Storage;

/// <summary>
/// Unified column storage: a sole-owner contiguous <typeparamref name="T"/>[] plus an
/// optional bool[] null mask. Whether a column dispatches to vectorized kernels is decided
/// at runtime by <see cref="KernelSelector"/>, never by the storage class itself.
/// Slices are zero-copy shared-buffer views (mutations to the source array are visible
/// through the slice). <see cref="AsTensor"/> exposes a lazy zero-copy
/// <see cref="Tensor{T}"/> view for unmanaged element types.
/// </summary>
/// <typeparam name="T">The type of elements to store</typeparam>
sealed class ColumnStorage<T> : IColumnStorage<T>
{
    readonly T[] data;
    readonly int dataStart;
    readonly int dataLength;
    readonly bool[]? nullMask;
    readonly int maskStart;
    bool disposed;
    Tensor<T>? tensorView;

    /// <summary>
    /// Initializes a new instance of ColumnStorage with the specified values
    /// </summary>
    /// <param name="values">The values to store</param>
    /// <param name="detectNulls">Whether to detect and track null values (for reference types)</param>
    public ColumnStorage(ReadOnlySpan<T> values, bool detectNulls = false)
    {
        if (values.IsEmpty)
        {
            data = [];
            dataStart = 0;
            dataLength = 0;
            nullMask = null;
            maskStart = 0;
            return;
        }

        var dataArray = values.ToArray();

        if (detectNulls && !typeof(T).IsValueType)
        {
            bool hasNulls = false;
            bool[]? nullMaskArray = null;

            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] == null)
                {
                    if (!hasNulls)
                    {
                        hasNulls = true;
                        nullMaskArray = new bool[values.Length];
                    }
                    nullMaskArray![i] = true;
                }
                else if (hasNulls)
                {
                    nullMaskArray![i] = false;
                }
            }

            data = dataArray;
            dataStart = 0;
            dataLength = dataArray.Length;
            nullMask = hasNulls ? nullMaskArray : null;
            maskStart = 0;
        }
        else
        {
            data = dataArray;
            dataStart = 0;
            dataLength = dataArray.Length;
            nullMask = null;
            maskStart = 0;
        }
    }

    /// <summary>
    /// Initializes a new instance of ColumnStorage sharing the caller's memory.
    /// The caller must not mutate the memory after this call.
    /// </summary>
    /// <param name="data">The memory containing the data (zero-copy shared)</param>
    /// <param name="nullMask">The optional null mask memory (zero-copy shared)</param>
    internal ColumnStorage(ReadOnlyMemory<T> data, ReadOnlyMemory<bool>? nullMask = null)
    {
        var (array, offset) = ExtractArray(data);
        this.data = array;
        dataStart = offset;
        dataLength = data.Length;
        (this.nullMask, maskStart) = ExtractMask(nullMask);
    }

    /// <summary>
    /// Initializes a new instance of ColumnStorage wrapping an array owned by the caller.
    /// The caller must not mutate the array after this call.
    /// </summary>
    /// <param name="ownedData">The data array (zero-copy wrapped, not copied)</param>
    /// <param name="nullMask">The optional null mask memory (zero-copy shared)</param>
    internal ColumnStorage(T[] ownedData, ReadOnlyMemory<bool>? nullMask = null)
    {
        ArgumentNullException.ThrowIfNull(ownedData);
        data = ownedData;
        dataStart = 0;
        dataLength = ownedData.Length;
        (this.nullMask, maskStart) = ExtractMask(nullMask);
    }

    ColumnStorage(T[] data, int dataStart, int dataLength, bool[]? nullMask, int maskStart)
    {
        this.data = data;
        this.dataStart = dataStart;
        this.dataLength = dataLength;
        this.nullMask = nullMask;
        this.maskStart = maskStart;
    }

    /// <inheritdoc />
    public int Length => dataLength;

    /// <inheritdoc />
    public bool IsVectorizable => ColumnStorageFactory.IsVectorizable<T>();

    /// <inheritdoc />
    public bool HasNulls => nullMask is not null;

    /// <inheritdoc />
    public bool ProvidesZeroCopySpanAccess => true;

    /// <inheritdoc />
    public StorageType StorageType => StorageType.Memory;

    /// <inheritdoc />
    public ReadOnlySpan<bool> NullMask
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return nullMask is null ? ReadOnlySpan<bool>.Empty : nullMask.AsSpan(maskStart, dataLength);
        }
    }

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (index < 0 || index >= dataLength)
                throw new IndexOutOfRangeException($"Index {index} is out of range for storage of length {dataLength}");

            return data[dataStart + index];
        }
    }

    /// <summary>
    /// Creates a new storage containing a slice of this storage.
    /// </summary>
    /// <remarks>
    /// Returns a true zero-copy view: the slice shares this storage's underlying buffer and
    /// null mask. Mutating the source array is visible through the slice (shared-buffer
    /// semantics, matching the historical MemoryStorage behavior).
    /// </remarks>
    /// <param name="start">The starting index of the slice</param>
    /// <param name="length">The number of elements in the slice</param>
    /// <returns>A new storage instance representing a zero-copy view of the requested range</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when start or length are invalid</exception>
    public IColumnStorage<T> Slice(int start, int length)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (start < 0)
            throw new ArgumentOutOfRangeException(nameof(start), "Start index cannot be negative");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "Length cannot be negative");
        if (start + length > dataLength)
            throw new ArgumentOutOfRangeException(nameof(length), "Start + length exceeds storage bounds");

        if (length == 0)
            return new ColumnStorage<T>(data, dataStart + start, 0, nullMask, maskStart + start);

        return new ColumnStorage<T>(data, dataStart + start, length, nullMask, maskStart + start);
    }

    /// <summary>
    /// Gets the underlying data memory for operations
    /// </summary>
    internal ReadOnlyMemory<T> Data
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return data.AsMemory(dataStart, dataLength);
        }
    }

    /// <summary>
    /// Gets the underlying null mask memory for null operations
    /// </summary>
    internal ReadOnlyMemory<bool>? NullMaskMemory
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return nullMask is null ? null : new ReadOnlyMemory<bool>(nullMask, maskStart, dataLength);
        }
    }

    /// <summary>
    /// Gets a lazy zero-copy <see cref="Tensor{T}"/> view over this storage's data.
    /// </summary>
    /// <remarks>
    /// The view shares the underlying array (verified via <see cref="Tensor{T}.TryGetSpan"/>).
    /// Callers must check <see cref="HasNulls"/> first; null positions hold <c>default(T)</c>.
    /// Requires unmanaged <typeparamref name="T"/>; reference-containing types throw.
    /// </remarks>
    /// <exception cref="InvalidOperationException">Thrown when <typeparamref name="T"/> is not unmanaged</exception>
    internal Tensor<T> AsTensor()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (tensorView is not null)
            return tensorView;

        if (RuntimeHelpers.IsReferenceOrContainsReferences<T>())
            throw new InvalidOperationException(
                $"Cannot create a Tensor view for non-unmanaged type {typeof(T).Name}; " +
                "ColumnStorage<T>.AsTensor() requires unmanaged T.");

        tensorView = dataLength == 0
            ? Tensor.Create<T>([], [], [])
            : Tensor.Create(data, dataStart, [dataLength], [1]);
        return tensorView;
    }

    /// <inheritdoc />
    ReadOnlySpan<T> IColumnStorage<T>.AsSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return data.AsSpan(dataStart, dataLength);
    }

    /// <inheritdoc />
    bool IColumnStorage<T>.TryGetSpan(out ReadOnlySpan<T> span)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (HasNulls)
        {
            span = default;
            return false;
        }

        span = data.AsSpan(dataStart, dataLength);
        return true;
    }

    /// <inheritdoc />
    Span<T> IColumnStorage<T>.AsWritableSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        return data.AsSpan(dataStart, dataLength).ToArray();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            disposed = true;
            tensorView = null;
        }
    }

    static (T[] Array, int Offset) ExtractArray(ReadOnlyMemory<T> memory)
    {
        if (MemoryMarshal.TryGetArray(memory, out ArraySegment<T> segment) && segment.Array is not null)
            return (segment.Array, segment.Offset);

        return (memory.ToArray(), 0);
    }

    static (bool[]? Mask, int Offset) ExtractMask(ReadOnlyMemory<bool>? mask)
    {
        if (mask is not { Length: > 0 } m)
            return (null, 0);

        if (MemoryMarshal.TryGetArray(m, out ArraySegment<bool> segment) && segment.Array is not null)
            return (segment.Array, segment.Offset);

        return (m.ToArray(), 0);
    }
}
