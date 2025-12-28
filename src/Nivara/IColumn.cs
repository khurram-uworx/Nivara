namespace Nivara;

/// <summary>
/// Public interface for strongly-typed columns in Nivara.
/// Provides read-only access to column data with null handling.
/// </summary>
/// <typeparam name="T">The type of elements in the column</typeparam>
public interface IColumn<T>
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
    /// Gets the element at the specified index
    /// </summary>
    /// <param name="index">The zero-based index of the element to get</param>
    /// <returns>The element at the specified index</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    T this[int index] { get; }

    /// <summary>
    /// Determines whether the element at the specified index is null
    /// </summary>
    /// <param name="index">The zero-based index to check</param>
    /// <returns>true if the element at the specified index is null; otherwise, false</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of bounds</exception>
    bool IsNull(int index);
}