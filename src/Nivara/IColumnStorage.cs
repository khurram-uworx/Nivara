using Nivara.Diagnostics;

namespace Nivara;

/// <summary>
/// Internal storage abstraction for column data.
/// Provides unified interface for both tensor-backed and memory-backed storage.
/// </summary>
/// <typeparam name="T">The type of elements stored in the column</typeparam>
internal interface IColumnStorage<T> : IDisposable
{
    /// <summary>
    /// Gets the number of elements in the storage
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Gets a value indicating whether this storage supports vectorized operations
    /// </summary>
    bool IsVectorizable { get; }

    /// <summary>
    /// Gets a value indicating whether this storage contains any null values
    /// </summary>
    bool HasNulls { get; }

    /// <summary>
    /// Gets the null mask indicating which positions contain null values.
    /// True indicates null, false indicates non-null.
    /// </summary>
    ReadOnlySpan<bool> NullMask { get; }

    /// <summary>
    /// Gets the element at the specified index
    /// </summary>
    /// <param name="index">The zero-based index of the element to get</param>
    /// <returns>The element at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    T this[int index] { get; }

    /// <summary>
    /// Creates a slice of this storage containing elements from the specified range
    /// </summary>
    /// <param name="start">The starting index of the slice</param>
    /// <param name="length">The number of elements in the slice</param>
    /// <returns>A new storage instance representing the slice</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when start or length are invalid</exception>
    IColumnStorage<T> Slice(int start, int length);

    /// <summary>
    /// Gets diagnostic information about this storage implementation.
    /// Used for performance analysis and kernel selection.
    /// </summary>
    StorageType StorageType { get; }

    /// <summary>
    /// Gets a read-only span view of the underlying data.
    /// Provides zero-copy access to the storage data for high-performance operations.
    /// </summary>
    /// <returns>A read-only span over the storage data</returns>
    /// <exception cref="InvalidOperationException">Thrown when the storage doesn't support span access</exception>
    internal ReadOnlySpan<T> AsSpan();

    /// <summary>
    /// Attempts to get a read-only span view of the underlying data when no nulls are present.
    /// Zero-copy when the column is null-free; returns false when nulls exist.
    /// </summary>
    /// <param name="span">When this method returns, contains the read-only span if successful</param>
    /// <returns>true if a span was obtained (no nulls present), false otherwise</returns>
    internal bool TryGetSpan(out ReadOnlySpan<T> span);

    /// <summary>
    /// Gets a writable span view of the underlying data.
    /// Provides zero-copy access for scenarios requiring data mutation.
    /// </summary>
    /// <returns>A writable span over the storage data</returns>
    /// <exception cref="InvalidOperationException">Thrown when the storage doesn't support writable span access</exception>
    internal Span<T> AsWritableSpan();
}