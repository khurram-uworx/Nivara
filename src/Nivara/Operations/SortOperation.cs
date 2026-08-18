using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Represents the sort direction for a column
/// </summary>
public enum SortDirection
{
    /// <summary>
    /// Sort in ascending order (smallest to largest)
    /// </summary>
    Ascending,

    /// <summary>
    /// Sort in descending order (largest to smallest)
    /// </summary>
    Descending
}

/// <summary>
/// Represents how null values should be ordered in sorting
/// </summary>
public enum NullOrdering
{
    /// <summary>
    /// Place null values first (before non-null values)
    /// </summary>
    NullsFirst,

    /// <summary>
    /// Place null values last (after non-null values)
    /// </summary>
    NullsLast
}

/// <summary>
/// Represents a sort key with column name, direction, and null ordering
/// </summary>
public sealed class SortKey
{
    /// <summary>
    /// Initializes a new instance of SortKey
    /// </summary>
    /// <param name="columnName">The name of the column to sort by</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <exception cref="ArgumentException">Thrown when columnName is null or whitespace</exception>
    public SortKey(string columnName, SortDirection direction = SortDirection.Ascending, NullOrdering nullOrdering = NullOrdering.NullsLast)
    {
        if (string.IsNullOrWhiteSpace(columnName))
            throw new ArgumentException("Column name cannot be null or whitespace", nameof(columnName));

        ColumnName = columnName;
        Direction = direction;
        NullOrdering = nullOrdering;
    }

    /// <summary>
    /// Gets the name of the column to sort by
    /// </summary>
    public string ColumnName { get; }

    /// <summary>
    /// Gets the sort direction
    /// </summary>
    public SortDirection Direction { get; }

    /// <summary>
    /// Gets the null ordering strategy
    /// </summary>
    public NullOrdering NullOrdering { get; }

    /// <summary>
    /// Returns a string representation of the sort key
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var directionStr = Direction == SortDirection.Ascending ? "ASC" : "DESC";
        var nullStr = NullOrdering == NullOrdering.NullsFirst ? "NULLS FIRST" : "NULLS LAST";
        return $"{ColumnName} {directionStr} {nullStr}";
    }
}

/// <summary>
/// Represents a sort operation that orders rows by one or more columns
/// </summary>
sealed class SortOperation : IQueryOperation, IParallelSortOperation
{
    readonly List<SortKey> sortKeys;
    readonly bool stable;

    /// <summary>
    /// Initializes a new instance of SortOperation
    /// </summary>
    /// <param name="sortKeys">The sort keys defining the sort order</param>
    /// <param name="stable">Whether to use stable sorting (preserves relative order of equal elements)</param>
    /// <exception cref="ArgumentNullException">Thrown when sortKeys is null</exception>
    /// <exception cref="ArgumentException">Thrown when no sort keys are provided</exception>
    public SortOperation(IEnumerable<SortKey> sortKeys, bool stable = true)
    {
        if (sortKeys == null)
            throw new ArgumentNullException(nameof(sortKeys));

        this.sortKeys = sortKeys.ToList();

        if (this.sortKeys.Count == 0)
            throw new ArgumentException("Must specify at least one sort key", nameof(sortKeys));

        this.stable = stable;
    }

    /// <summary>
    /// Initializes a new instance of SortOperation with a single sort key
    /// </summary>
    /// <param name="columnName">The name of the column to sort by</param>
    /// <param name="direction">The sort direction</param>
    /// <param name="nullOrdering">How to order null values</param>
    /// <param name="stable">Whether to use stable sorting</param>
    public SortOperation(string columnName, SortDirection direction = SortDirection.Ascending,
        NullOrdering nullOrdering = NullOrdering.NullsLast, bool stable = true)
        : this(new[] { new SortKey(columnName, direction, nullOrdering) }, stable)
    {
    }

    /// <summary>
    /// Gets the sort keys
    /// </summary>
    public IReadOnlyList<SortKey> SortKeys => sortKeys;

    /// <summary>
    /// Gets whether stable sorting is used
    /// </summary>
    public bool IsStable => stable;

    public string OperationType => Query.OperationType.Sort;

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate all sort keys exist in the schema
        foreach (var sortKey in sortKeys)
        {
            if (!inputSchema.HasColumn(sortKey.ColumnName))
            {
                throw new SchemaValidationException(
                    $"Sort key column '{sortKey.ColumnName}' not found in schema. Available columns: {string.Join(", ", inputSchema.ColumnNames)}");
            }

            // Validate that the column type is comparable
            var columnType = inputSchema.GetColumnType(sortKey.ColumnName);
            if (!IsComparableType(columnType))
            {
                throw new SchemaValidationException(
                    $"Column '{sortKey.ColumnName}' of type '{columnType.Name}' is not comparable and cannot be used for sorting");
            }
        }

        // Sort doesn't change the schema structure, only the row order
        return inputSchema;
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (input.Count == 0)
            return input;

        try
        {
            // Get the row count from any column
            var rowCount = input.Values.First().Length;

            if (rowCount <= 1)
            {
                // No need to sort if we have 0 or 1 rows
                return input;
            }

            // Compute sort indices
            var sortIndices = ComputeSortIndices(input, rowCount);

            // Reorder all columns using the computed indices
            var sortedColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
            {
                var reorderedColumn = ReorderColumn(kvp.Value, sortIndices);
                sortedColumns[kvp.Key] = reorderedColumn;
            }

            return sortedColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Sort operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Computes the sort indices for reordering rows.
    /// Fast path: pre-captures all column references once via delegates,
    /// eliminating per-comparison dictionary lookup and type-switch dispatch.
    /// </summary>
    private int[] ComputeSortIndices(IReadOnlyDictionary<string, IColumn> input, int rowCount)
    {
        var indices = Enumerable.Range(0, rowCount).ToArray();

        if (SortKeyComparerFactory.TryCreatePreCapturedComparer(input, sortKeys, out var comparer))
        {
            Array.Sort(indices, comparer);
            return indices;
        }

        var fallback = new MultiColumnComparer(input, sortKeys);
        if (stable)
            indices = indices.OrderBy(i => i, fallback).ToArray();
        else
            Array.Sort(indices, fallback);

        return indices;
    }

    /// <summary>
    /// Reorders a column using the specified indices
    /// </summary>
    /// <param name="column">The column to reorder</param>
    /// <param name="indices">The indices specifying the new order</param>
    /// <returns>A reordered column</returns>
    public static IColumn ReorderColumn(IColumn column, int[] indices)
        => ColumnFilterHelper.ReorderColumn(column, indices);

    /// <summary>
    /// Checks if a type is comparable and can be used for sorting
    /// </summary>
    /// <param name="type">The type to check</param>
    /// <returns>True if the type is comparable</returns>
    internal static bool IsComparableType(Type type)
    {
        // Handle nullable types
        var underlyingType = Nullable.GetUnderlyingType(type) ?? type;

        // Check if the type implements IComparable or IComparable<T>
        if (typeof(IComparable).IsAssignableFrom(underlyingType))
            return true;

        var comparableInterface = typeof(IComparable<>).MakeGenericType(underlyingType);
        if (comparableInterface.IsAssignableFrom(underlyingType))
            return true;

        return false;
    }

    /// <summary>
    /// Returns a string representation of the sort operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var keysStr = string.Join(", ", sortKeys);
        var stableStr = stable ? " (stable)" : "";
        return $"Sort({keysStr}){stableStr}";
    }
}

/// <summary>
/// Comparer that handles multiple sort keys with proper null handling
/// </summary>
internal sealed class MultiColumnComparer : IComparer<int>
{
    /// <summary>
    /// Compares two values from a column at the specified indices
    /// </summary>
    /// <param name="column">The column containing the values</param>
    /// <param name="indexX">The index of the first value</param>
    /// <param name="indexY">The index of the second value</param>
    /// <param name="sortKey">The sort key defining how to compare</param>
    /// <returns>A comparison result</returns>
    static int compareValues(IColumn column, int indexX, int indexY, SortKey sortKey)
        => column switch
        {
            NivaraColumn<bool> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<char> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<byte> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<sbyte> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<short> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<ushort> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<int> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<uint> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<long> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<ulong> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<nint> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<nuint> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<Int128> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<UInt128> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<float> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<double> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<Half> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<decimal> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<string> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<Guid> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<DateTime> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<DateTimeOffset> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<TimeSpan> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<DateOnly> c => compareTyped(c, indexX, indexY, sortKey),
            NivaraColumn<TimeOnly> c => compareTyped(c, indexX, indexY, sortKey),
            _ => compareBoxed(column, indexX, indexY, sortKey)
        };

    /// <summary>
    /// Compares two values from a typed column without boxing, via <see cref="Comparer{T}.Default"/>.
    /// Nulls are read from the column's null mask and honor the sort key's <see cref="NullOrdering"/>.
    /// </summary>
    static int compareTyped<T>(NivaraColumn<T> column, int indexX, int indexY, SortKey sortKey)
        where T : IComparable<T>
    {
        bool xNull = column.IsNull(indexX);
        bool yNull = column.IsNull(indexY);

        if (xNull && yNull) return 0;
        if (xNull) return sortKey.NullOrdering == NullOrdering.NullsFirst ? -1 : 1;
        if (yNull) return sortKey.NullOrdering == NullOrdering.NullsFirst ? 1 : -1;

        int comparison = Comparer<T>.Default.Compare(column[indexX], column[indexY]);
        return sortKey.Direction == SortDirection.Ascending ? comparison : -comparison;
    }

    /// <summary>
    /// Boxed fallback for non-<see cref="NivaraColumn{T}"/> object columns and custom element types:
    /// compares via <see cref="IComparable"/> and falls back to ordinal string comparison.
    /// </summary>
    static int compareBoxed(IColumn column, int indexX, int indexY, SortKey sortKey)
    {
        var valueX = column.GetValue(indexX);
        var valueY = column.GetValue(indexY);

        if (valueX == null && valueY == null)
            return 0;

        if (valueX == null)
            return sortKey.NullOrdering == NullOrdering.NullsFirst ? -1 : 1;

        if (valueY == null)
            return sortKey.NullOrdering == NullOrdering.NullsFirst ? 1 : -1;

        int comparison;
        if (valueX is IComparable comparableX)
            comparison = comparableX.CompareTo(valueY);
        else
            comparison = string.Compare(valueX.ToString(), valueY.ToString(), StringComparison.Ordinal);

        return sortKey.Direction == SortDirection.Ascending ? comparison : -comparison;
    }

    readonly IReadOnlyDictionary<string, IColumn> columns;
    readonly IReadOnlyList<SortKey> sortKeys;

    public MultiColumnComparer(IReadOnlyDictionary<string, IColumn> columns, IReadOnlyList<SortKey> sortKeys)
    {
        this.columns = columns;
        this.sortKeys = sortKeys;
    }

    public int Compare(int x, int y)
    {
        for (int i = 0; i < sortKeys.Count; i++)
        {
            var sortKey = sortKeys[i];
            var column = columns[sortKey.ColumnName];
            var result = compareValues(column, x, y, sortKey);

            if (result != 0)
            {
                return result;
            }
        }

        // All sort keys are equal
        return 0;
    }
}
