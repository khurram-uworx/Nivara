using Nivara.Diagnostics;

namespace Nivara;

/// <summary>
/// Non-generic base interface for columns, used by the query engine.
/// Provides type-erased access to column operations.
/// </summary>
public interface IColumn : IDisposable
{
    /// <summary>
    /// Gets the number of elements in the column
    /// </summary>
    int Length { get; }

    /// <summary>
    /// Gets a value indicating whether this column contains any null values
    /// </summary>
    bool HasNulls { get; }

    /// <summary>
    /// Gets the element type of the column
    /// </summary>
    Type ElementType { get; }

    /// <summary>
    /// Gets the element at the specified index as an object
    /// </summary>
    /// <param name="index">The zero-based index of the element to get</param>
    /// <returns>The element at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    object? GetValue(int index);

    /// <summary>
    /// Determines whether the element at the specified index is null
    /// </summary>
    /// <param name="index">The zero-based index to check</param>
    /// <returns>true if the element at the specified index is null; otherwise, false</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    bool IsNull(int index);
}

/// <summary>
/// Public interface for strongly-typed columns in Nivara.
/// Provides read-only access to column data with null handling.
/// </summary>
/// <typeparam name="T">The type of elements in the column</typeparam>
public interface IColumn<T> : IColumn
{
    /// <summary>
    /// Gets the element at the specified index
    /// </summary>
    /// <param name="index">The zero-based index of the element to get</param>
    /// <returns>The element at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    T this[int index] { get; }
}

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

/// <summary>
/// Public interface for NivaraFrame, providing read-only access to multi-column data structures.
/// Serves as the primary interface for DataFrame-like operations in Nivara.
/// </summary>
public interface IFrame : IDisposable
{
    /// <summary>
    /// Gets the number of rows in the frame
    /// </summary>
    int RowCount { get; }

    /// <summary>
    /// Gets the number of columns in the frame
    /// </summary>
    int ColumnCount { get; }

    /// <summary>
    /// Gets the names of all columns in the frame
    /// </summary>
    IReadOnlyList<string> ColumnNames { get; }

    /// <summary>
    /// Gets the schema information for this frame
    /// </summary>
    Schema Schema { get; }

    /// <summary>
    /// Gets a strongly-typed column by name
    /// </summary>
    /// <typeparam name="T">The expected type of the column</typeparam>
    /// <param name="name">The name of the column</param>
    /// <returns>The column as a NivaraColumn&lt;T&gt;</returns>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    /// <exception cref="ColumnTypeMismatchException">Thrown when the column type doesn't match T</exception>
    NivaraColumn<T> GetColumn<T>(string name);

    /// <summary>
    /// Checks if a column with the specified name exists
    /// </summary>
    /// <param name="name">The name of the column to check</param>
    /// <returns>True if the column exists, false otherwise</returns>
    bool HasColumn(string name);
}
