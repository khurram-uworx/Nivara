namespace Nivara.Storage;

/// <summary>
/// A chunk-structured <see cref="IColumnStorage{T}"/>: an ordered list of at least two
/// <see cref="ColumnStorage{T}"/> chunks. Structure (chunk boundaries) is carried through
/// <see cref="Slice"/> and concatenation; compute materialises one contiguous copy on first
/// request and caches it. Chunks are never disposed by this type — they are views over buffers
/// owned elsewhere, and there is no ownership transfer.
/// </summary>
/// <remarks>
/// A single chunk never produces a <see cref="ChunkedColumn{T}"/>: the factories collapse it to
/// its <see cref="ColumnStorage{T}"/>. That invariant is what keeps the flat hot path untouched.
/// </remarks>
/// <typeparam name="T">The element type</typeparam>
sealed class ChunkedColumn<T> : IColumnStorage<T>
{
    readonly ColumnStorage<T>[] chunkStorages;
    readonly Chunk<T>[] chunks;
    readonly int[] starts;
    readonly int length;
    readonly bool hasNulls;
    readonly object materializeLock = new();
    bool disposed;
    T[]? materializedData;
    bool[]? materializedMask;

    internal ChunkedColumn(IReadOnlyList<ColumnStorage<T>> chunkStorages)
    {
        ArgumentNullException.ThrowIfNull(chunkStorages);

        if (chunkStorages.Count < 2)
            throw new ArgumentException("A chunked column requires at least two chunks.", nameof(chunkStorages));

        this.chunkStorages = new ColumnStorage<T>[chunkStorages.Count];
        chunks = new Chunk<T>[chunkStorages.Count];
        starts = new int[chunkStorages.Count + 1];

        bool anyNulls = false;
        for (int i = 0; i < chunkStorages.Count; i++)
        {
            var storage = chunkStorages[i] ?? throw new ArgumentException("Chunks cannot be null.", nameof(chunkStorages));
            if (storage.Length == 0)
                throw new ArgumentException("Chunks cannot be empty.", nameof(chunkStorages));

            this.chunkStorages[i] = storage;
            chunks[i] = new Chunk<T>(storage);
            starts[i + 1] = starts[i] + storage.Length;
            anyNulls |= storage.HasNulls;
        }

        length = starts[^1];
        hasNulls = anyNulls;
    }

    /// <summary>
    /// Gets a value indicating whether the element at the specified index is null.
    /// Resolved per chunk without materialising the contiguous mask.
    /// </summary>
    internal bool IsNull(int index)
    {
        ObjectDisposedException.ThrowIf(disposed, this);

        if (index < 0 || index >= length)
            throw new IndexOutOfRangeException($"Index {index} is out of range for chunked storage of length {length}");

        int chunkIndex = FindChunk(index);
        return chunks[chunkIndex].IsNull(index - starts[chunkIndex]);
    }

    /// <summary>
    /// Gets the chunks that make up this column, in order.
    /// </summary>
    internal ReadOnlySpan<ColumnStorage<T>> ChunkStorages => chunkStorages;

    /// <summary>
    /// Gets a value indicating whether the contiguous copy has been materialised.
    /// </summary>
    internal bool IsMaterialized => materializedData is not null;

    /// <inheritdoc />
    public int Length
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return length;
        }
    }

    /// <inheritdoc />
    public bool IsVectorizable => ColumnStorageFactory.IsVectorizable<T>();

    /// <inheritdoc />
    public bool HasNulls
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return hasNulls;
        }
    }

    /// <inheritdoc />
    public ReadOnlySpan<bool> NullMask
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (!hasNulls)
                return ReadOnlySpan<bool>.Empty;

            EnsureMaterialized();
            return materializedMask;
        }
    }

    /// <inheritdoc />
    public T this[int index]
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (index < 0 || index >= length)
                throw new IndexOutOfRangeException($"Index {index} is out of range for chunked storage of length {length}");

            int chunkIndex = FindChunk(index);
            return chunks[chunkIndex][index - starts[chunkIndex]];
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
        if (start + length > this.length)
            throw new ArgumentOutOfRangeException(nameof(length), "Start + length exceeds storage bounds");

        if (length == 0)
            return new ColumnStorage<T>(ReadOnlySpan<T>.Empty);

        var slice = new List<ColumnStorage<T>>();
        int remaining = length;
        int position = start;
        int chunkIndex = FindChunk(start);

        while (remaining > 0)
        {
            int withinChunk = position - starts[chunkIndex];
            int take = Math.Min(remaining, chunks[chunkIndex].Length - withinChunk);
            slice.Add(chunks[chunkIndex].Slice(withinChunk, take));
            position += take;
            remaining -= take;
            chunkIndex++;
        }

        return slice.Count == 1 ? slice[0] : new ChunkedColumn<T>(slice);
    }

    /// <inheritdoc />
    public ReadOnlyMemory<T> Data
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            EnsureMaterialized();
            return materializedData;
        }
    }

    /// <inheritdoc />
    ReadOnlyMemory<bool>? IColumnStorage<T>.NullMaskMemory
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);

            if (!hasNulls)
                return null;

            EnsureMaterialized();
            return materializedMask;
        }
    }

    /// <inheritdoc />
    bool IColumnStorage<T>.TryGetHashCodes(out ReadOnlySpan<int> hashes)
    {
        hashes = default;
        return false;
    }

    /// <inheritdoc />
    ReadOnlySpan<T> IColumnStorage<T>.AsSpan()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        EnsureMaterialized();
        return materializedData;
    }

    /// <inheritdoc />
    bool IColumnStorage<T>.TryGetSpan(out ReadOnlySpan<T> span)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        span = default;
        return false;
    }

    /// <inheritdoc />
    Span<T> IColumnStorage<T>.AsWritableSpan()
        => throw new NotSupportedException(
            "Cannot expose a writable span over chunked storage; writing to a materialised copy would be silent data loss.");

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
            return;

        disposed = true;

        lock (materializeLock)
        {
            materializedData = null;
            materializedMask = null;
        }

        // Chunks are views over buffers owned elsewhere; dropping the references is all this type owns.
        // Both arrays hold the same ColumnStorage<T> references, so clear both or the struct array
        // would keep every chunk reachable.
        Array.Clear(chunks);
        Array.Clear(chunkStorages);
    }

    /// <summary>
    /// Collapses a list of chunk storages into the smallest representation: empty, a flat
    /// <see cref="ColumnStorage{T}"/>, or a <see cref="ChunkedColumn{T}"/> with at least two chunks.
    /// </summary>
    internal static IColumnStorage<T> FromChunkStorages(IEnumerable<ColumnStorage<T>> storages)
    {
        var nonEmpty = new List<ColumnStorage<T>>();
        foreach (var storage in storages)
        {
            if (storage.Length > 0)
                nonEmpty.Add(storage);
        }

        return nonEmpty.Count switch
        {
            0 => new ColumnStorage<T>(ReadOnlySpan<T>.Empty),
            1 => nonEmpty[0],
            _ => new ChunkedColumn<T>(nonEmpty),
        };
    }

    /// <summary>
    /// Flattens a list of storages (flat or chunked) into one chunk-structured storage,
    /// preserving existing chunk boundaries without copying.
    /// </summary>
    internal static IColumnStorage<T> Create(IEnumerable<IColumnStorage<T>> storages)
    {
        var flattened = new List<ColumnStorage<T>>();
        foreach (var storage in storages)
        {
            switch (storage)
            {
                case ChunkedColumn<T> chunked:
                    flattened.AddRange(chunked.chunkStorages);
                    break;
                case ColumnStorage<T> flat:
                    flattened.Add(flat);
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported storage type {storage.GetType().Name} for chunked concatenation.");
            }
        }

        return FromChunkStorages(flattened);
    }

    void EnsureMaterialized()
    {
        if (materializedData is not null)
            return;

        lock (materializeLock)
        {
            if (materializedData is not null)
                return;

            var data = new T[length];
            bool[]? mask = hasNulls ? new bool[length] : null;

            int offset = 0;
            for (int chunkIndex = 0; chunkIndex < chunks.Length; chunkIndex++)
            {
                var storage = chunkStorages[chunkIndex];
                storage.Data.Span.CopyTo(data.AsSpan(offset));

                if (mask is not null)
                {
                    var chunkMask = storage.NullMask;
                    if (chunkMask.Length > 0)
                        chunkMask.CopyTo(mask.AsSpan(offset));
                }

                offset += storage.Length;
            }

            materializedMask = mask;
            materializedData = data;
        }
    }

    int FindChunk(int index)
    {
        int lo = 0;
        int hi = chunks.Length - 1;

        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (starts[mid] <= index)
                lo = mid;
            else
                hi = mid - 1;
        }

        return lo;
    }
}
