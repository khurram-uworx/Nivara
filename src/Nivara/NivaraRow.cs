using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// A typed, allocation-free view over a single row of a <see cref="NivaraFrame"/>. Constructed by
/// the framework and passed to the predicate of <see cref="NivaraFrameExtensions.Where"/>; the
/// struct is read-only and carries the row's position plus a shared column array and name map.
/// </summary>
/// <remarks>
/// The zero state (<c>default(NivaraRow)</c>) is valid and throws a clear
/// <see cref="InvalidOperationException"/> on access so a defaulted value is never misread as data.
/// </remarks>
public readonly struct NivaraRow
{
    readonly IColumn[] columns;
    readonly IReadOnlyDictionary<string, int>? map;
    readonly int rowIndex;

    internal NivaraRow(IColumn[] columns, IReadOnlyDictionary<string, int> map, int rowIndex)
    {
        this.columns = columns;
        this.map = map;
        this.rowIndex = rowIndex;
    }

    /// <summary>
    /// Gets the zero-based position of this row within the source frame
    /// </summary>
    public int RowIndex => rowIndex;

    internal IColumn[] Columns => columns;

    /// <summary>
    /// Gets the column names for this row's source frame
    /// </summary>
    public string[] ColumnNames => map is not null
        ? [.. map.Keys]
        : [];

    /// <summary>
    /// Gets the value at the specified column as an object, or <c>null</c> for a null cell
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The value at the column, or <c>null</c> when the cell is null</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public object? this[string columnName] => GetColumn(columnName).GetValue(rowIndex);

    /// <summary>
    /// Gets the value at the specified column strongly typed. Returns the stored value on null
    /// cells (matching the <c>NivaraColumn{T}</c> indexer contract); use <see cref="IsNull"/> to
    /// detect nulls.
    /// </summary>
    /// <typeparam name="T">The expected column element type</typeparam>
    /// <param name="columnName">The name of the column</param>
    /// <returns>The value at the column</returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    /// <exception cref="ColumnTypeMismatchException">Thrown when the column element type does not match <typeparamref name="T"/></exception>
    public T GetValue<T>(string columnName)
    {
        var column = GetColumn(columnName);

        if (column.ElementType != typeof(T))
            throw new ColumnTypeMismatchException(columnName, typeof(T), column.ElementType);

        return ((IColumn<T>)column)[rowIndex];
    }

    /// <summary>
    /// Attempts to get the value at the specified column strongly typed
    /// </summary>
    /// <typeparam name="T">The expected column element type</typeparam>
    /// <param name="columnName">The name of the column</param>
    /// <param name="value">When this method returns <c>true</c>, the value at the column</param>
    /// <returns><c>true</c> when the column exists and its element type matches <typeparamref name="T"/>; otherwise <c>false</c></returns>
    public bool TryGetValue<T>(string columnName, out T value)
    {
        if (!TryGetColumn(columnName, out var column) || column!.ElementType != typeof(T))
        {
            value = default!;
            return false;
        }

        value = ((IColumn<T>)column)[rowIndex];
        return true;
    }

    /// <summary>
    /// Determines whether the cell at the specified column is null
    /// </summary>
    /// <param name="columnName">The name of the column</param>
    /// <returns><c>true</c> when the cell is null; otherwise <c>false</c></returns>
    /// <exception cref="InvalidOperationException">Thrown when the row is the default struct value</exception>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    /// <exception cref="ColumnNotFoundException">Thrown when the column is not found</exception>
    public bool IsNull(string columnName) => GetColumn(columnName).IsNull(rowIndex);

    IColumn GetColumn(string columnName)
    {
        if (columns is null)
            throw new InvalidOperationException("NivaraRow is the default value and does not reference a frame row.");

        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        if (map is null || !map.TryGetValue(columnName, out var index))
            throw new ColumnNotFoundException(columnName, map?.Keys ?? []);

        return columns[index];
    }

    bool TryGetColumn(string columnName, out IColumn? column)
    {
        column = null;

        if (columns is null || map is null || string.IsNullOrWhiteSpace(columnName))
            return false;

        if (!map.TryGetValue(columnName, out var index))
            return false;

        column = columns[index];
        return true;
    }
}
