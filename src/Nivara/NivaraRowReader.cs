using Nivara.Exceptions;

namespace Nivara;

/// <summary>
/// Binds a set of columns to a schema once, allowing columns to be pre-resolved by name or index
/// and read through <see cref="NivaraColumnReader{T}"/> without per-cell string lookups.
/// </summary>
public sealed class NivaraRowReader
{
    readonly IColumn[] columns;
    readonly string[] columnNames;
    readonly Dictionary<string, int> map;

    public NivaraRowReader(NivaraFrame frame)
        : this(frame.ColumnNames.ToArray(), frame.ColumnNames.Select(n => frame.GetColumn(n)).ToArray())
    {
    }

    public NivaraRowReader(string[] columnNames, IColumn[] columns)
    {
        if (columnNames is null) throw new ArgumentNullException(nameof(columnNames));
        if (columns is null) throw new ArgumentNullException(nameof(columns));
        if (columnNames.Length != columns.Length)
            throw new ArgumentException("Column names and columns must have the same length.");

        this.columnNames = columnNames;
        this.columns = columns;
        this.map = new Dictionary<string, int>(columnNames.Length, StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < columnNames.Length; i++)
            map[columnNames[i]] = i;
    }

    public NivaraRowReader(IColumn[] columns, string[] columnNames)
        : this(columnNames, columns)
    {
    }

    /// <summary>
    /// Gets the number of columns.
    /// </summary>
    public int ColumnCount => columns.Length;

    /// <summary>
    /// Gets the column names.
    /// </summary>
    public IReadOnlyList<string> ColumnNames => columnNames;

    /// <summary>
    /// Resolves a column name to its zero-based index.
    /// </summary>
    public int GetColumnIndex(string columnName)
    {
        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        if (!map.TryGetValue(columnName, out var index))
            throw new ColumnNotFoundException(columnName, columnNames);

        return index;
    }

    /// <summary>
    /// Attempts to resolve a column name to its zero-based index.
    /// </summary>
    public bool TryGetColumnIndex(string columnName, out int index)
    {
        index = -1;
        if (string.IsNullOrWhiteSpace(columnName))
            return false;
        return map.TryGetValue(columnName, out index);
    }

    /// <summary>
    /// Gets the column at the specified zero-based index.
    /// </summary>
    public IColumn GetColumn(int columnIndex)
    {
        if (columnIndex < 0 || columnIndex >= columns.Length)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));
        return columns[columnIndex];
    }

    /// <summary>
    /// Gets the column with the specified name.
    /// </summary>
    public IColumn GetColumn(string columnName) => columns[GetColumnIndex(columnName)];

    /// <summary>
    /// Attempts to get the column with the specified name.
    /// </summary>
    public bool TryGetColumn(string columnName, out IColumn? column)
    {
        column = null;
        if (!TryGetColumnIndex(columnName, out var index))
            return false;
        column = columns[index];
        return true;
    }

    /// <summary>
    /// Creates a typed reader for the column with the specified name.
    /// </summary>
    public NivaraColumnReader<T> Column<T>(string columnName)
        => new NivaraColumnReader<T>(GetColumn(columnName), columnName);

    /// <summary>
    /// Creates a typed reader for the column at the specified index.
    /// </summary>
    public NivaraColumnReader<T> Column<T>(int columnIndex)
        => new NivaraColumnReader<T>(columns[columnIndex], columnNames[columnIndex]);

    /// <summary>
    /// Creates a typed reader for the column with the specified name.
    /// </summary>
    public NivaraColumnReader<T> GetColumn<T>(string columnName) => Column<T>(columnName);

    /// <summary>
    /// Creates a typed reader for the column at the specified index.
    /// </summary>
    public NivaraColumnReader<T> GetColumn<T>(int columnIndex) => Column<T>(columnIndex);
}
