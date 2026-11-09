namespace Nivara.Storage;

/// <summary>
/// One already-sliced <see cref="ColumnStorage{T}"/> inside a <see cref="ChunkedColumn{T}"/>.
/// The storage itself carries the offset and length, so a chunk needs no offset field.
/// </summary>
/// <typeparam name="T">The element type</typeparam>
readonly struct Chunk<T>
{
    internal Chunk(ColumnStorage<T> storage)
    {
        Storage = storage;
    }

    /// <summary>
    /// Gets the underlying storage for this chunk. It is a zero-copy view over the parent buffer.
    /// </summary>
    internal ColumnStorage<T> Storage { get; }

    /// <summary>
    /// Gets the number of elements in this chunk.
    /// </summary>
    internal int Length => Storage.Length;

    /// <summary>
    /// Gets the element at the specified index within this chunk.
    /// </summary>
    internal T this[int index] => Storage[index];

    /// <summary>
    /// Gets a value indicating whether the element at the specified index within this chunk is null.
    /// </summary>
    internal bool IsNull(int index)
    {
        var mask = Storage.NullMask;
        return mask.Length > 0 && mask[index];
    }

    /// <summary>
    /// Creates a zero-copy slice of this chunk.
    /// </summary>
    internal ColumnStorage<T> Slice(int start, int length) => (ColumnStorage<T>)Storage.Slice(start, length);
}
