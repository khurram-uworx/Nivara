using Nivara.Exceptions;

namespace Nivara;

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