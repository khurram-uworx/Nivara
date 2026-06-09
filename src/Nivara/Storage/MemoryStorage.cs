using Nivara.Diagnostics;

namespace Nivara.Storage;

/// <summary>
/// Memory-backed storage implementation for non-vectorizable types.
/// Uses Memory&lt;T&gt; for efficient memory management without vectorization.
/// </summary>
/// <typeparam name="T">The type of elements to store</typeparam>
sealed class MemoryStorage<T> : IColumnStorage<T>
{
    readonly ReadOnlyMemory<T> data;
    readonly ReadOnlyMemory<bool>? nullMask;
    bool disposed;

    /// <summary>
    /// Initializes a new instance of MemoryStorage with the specified values
    /// </summary>
    /// <param name="values">The values to store</param>
    /// <param name="detectNulls">Whether to detect and track null values (for reference types)</param>
    public MemoryStorage(ReadOnlySpan<T> values, bool detectNulls = false)
    {
        if (values.IsEmpty)
        {
            data = ReadOnlyMemory<T>.Empty;
            nullMask = null;
            return;
        }

        var dataArray = values.ToArray();

        if (detectNulls && !typeof(T).IsValueType)
        {
            bool hasNulls = false;
            bool[]? nullMaskArray = null;

            // Check for nulls in reference types
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

            data = new ReadOnlyMemory<T>(dataArray);
            nullMask = hasNulls ? new ReadOnlyMemory<bool>(nullMaskArray!) : null;
        }
        else
        {
            data = new ReadOnlyMemory<T>(dataArray);
            nullMask = null;
        }
    }

    /// <summary>
    /// Initializes a new instance of MemoryStorage with existing memory and null mask
    /// </summary>
    /// <param name="data">The memory containing the data</param>
    /// <param name="nullMask">The optional null mask memory</param>
    internal MemoryStorage(ReadOnlyMemory<T> data, ReadOnlyMemory<bool>? nullMask = null)
    {
        this.data = data;
        this.nullMask = nullMask;
    }

    /// <inheritdoc />
    public int Length => data.Length;

    /// <inheritdoc />
    public bool IsVectorizable => false;

    /// <inheritdoc />
    public bool HasNulls => nullMask.HasValue && nullMask.Value.Length > 0;

    /// <inheritdoc />
    public StorageType StorageType => StorageType.Memory;

    /// <inheritdoc />
    public ReadOnlySpan<bool> NullMask
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return nullMask.HasValue ? nullMask.Value.Span : ReadOnlySpan<bool>.Empty;
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

            return data.Span[index];
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
            return new MemoryStorage<T>(ReadOnlyMemory<T>.Empty);
        }

        // Create sliced data memory
        var slicedData = data.Slice(start, length);

        // Create sliced null mask if it exists
        ReadOnlyMemory<bool>? slicedNullMask = null;
        if (nullMask.HasValue && nullMask.Value.Length > 0)
        {
            slicedNullMask = nullMask.Value.Slice(start, length);
        }

        return new MemoryStorage<T>(slicedData, slicedNullMask);
    }

    /// <summary>
    /// Gets the underlying data memory for operations
    /// </summary>
    internal ReadOnlyMemory<T> Data
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return data;
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
            return nullMask;
        }
    }

    /// <inheritdoc />
    ReadOnlySpan<T> IColumnStorage<T>.AsSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return data.Span;
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

        span = data.Span;
        return true;
    }

    /// <inheritdoc />
    Span<T> IColumnStorage<T>.AsWritableSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        // For memory storage, we need to create a writable copy
        // since our data is ReadOnlyMemory<T>
        var writableArray = data.ToArray();
        return writableArray.AsSpan();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (!disposed)
        {
            // Memory<T> doesn't require explicit disposal, but we mark as disposed
            // to prevent further access
            disposed = true;
        }
    }
}
