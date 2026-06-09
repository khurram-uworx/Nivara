using Nivara.Exceptions;
using Nivara.Execution;
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

    /// <inheritdoc />
    public string OperationType => "Sort";

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
    /// Computes the sort indices for reordering rows
    /// </summary>
    /// <param name="input">The input columns</param>
    /// <param name="rowCount">The number of rows</param>
    /// <returns>An array of indices representing the sort order</returns>
    private int[] ComputeSortIndices(IReadOnlyDictionary<string, IColumn> input, int rowCount)
    {
        // Create an array of indices to sort
        var indices = Enumerable.Range(0, rowCount).ToArray();

        // Create a comparer that uses all sort keys
        var comparer = new MultiColumnComparer(input, sortKeys);

        // Sort the indices using the comparer
        if (stable)
        {
            // Use stable sort (Array.Sort is not stable, so we use OrderBy which is stable)
            indices = indices.OrderBy(i => i, comparer).ToArray();
        }
        else
        {
            // Use unstable sort for potentially better performance
            Array.Sort(indices, comparer);
        }

        return indices;
    }

    /// <summary>
    /// Reorders a column using the specified indices
    /// </summary>
    /// <param name="column">The column to reorder</param>
    /// <param name="indices">The indices specifying the new order</param>
    /// <returns>A reordered column</returns>
    public static IColumn ReorderColumn(IColumn column, int[] indices)
    {
        var elementType = column.ElementType;

        // Use dynamic dispatch to create the appropriate column type
        return elementType switch
        {
            Type t when t == typeof(int) => ReorderColumnTyped<int>(column, indices),
            Type t when t == typeof(double) => ReorderColumnTyped<double>(column, indices),
            Type t when t == typeof(float) => ReorderColumnTyped<float>(column, indices),
            Type t when t == typeof(long) => ReorderColumnTyped<long>(column, indices),
            Type t when t == typeof(string) => ReorderColumnTyped<string>(column, indices),
            Type t when t == typeof(bool) => ReorderColumnTyped<bool>(column, indices),
            Type t when t == typeof(decimal) => ReorderColumnTyped<decimal>(column, indices),
            Type t when t == typeof(byte) => ReorderColumnTyped<byte>(column, indices),
            Type t when t == typeof(short) => ReorderColumnTyped<short>(column, indices),
            Type t when t == typeof(DateTime) => ReorderColumnTyped<DateTime>(column, indices),
            _ => ReorderColumnGeneric(column, indices)
        };
    }

    /// <summary>
    /// Reorders a column for a specific type
    /// </summary>
    public static IColumn ReorderColumnTyped<T>(IColumn column, int[] indices)
    {
        // Check if T is a value type to determine which creation method to use
        if (typeof(T).IsValueType)
        {
            // For value types, create nullable array and use CreateFromNullable
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var reorderedArray = System.Array.CreateInstance(nullableType, indices.Length);

            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value != null)
                {
                    var nullableInstance = Activator.CreateInstance(nullableType, value);
                    reorderedArray.SetValue(nullableInstance, i);
                }
                // null values remain null in the array
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { reorderedArray })!;
        }
        else
        {
            // For reference types, create regular array and use CreateForReferenceType
            var reorderedArray = new T[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                var value = column.GetValue(indices[i]);
                reorderedArray[i] = (T)value!; // Reference types can be null
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { reorderedArray })!;
        }
    }

    /// <summary>
    /// Reorders a column for unknown types using object column
    /// </summary>
    public static IColumn ReorderColumnGeneric(IColumn column, int[] indices)
    {
        var reorderedArray = new object[indices.Length];

        for (int i = 0; i < indices.Length; i++)
        {
            reorderedArray[i] = column.GetValue(indices[i])!;
        }

        return NivaraColumn<object>.Create(reorderedArray);
    }

    /// <summary>
    /// Checks if a type is comparable and can be used for sorting
    /// </summary>
    /// <param name="type">The type to check</param>
    /// <returns>True if the type is comparable</returns>
    private static bool IsComparableType(Type type)
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
public sealed class MultiColumnComparer : IComparer<int>
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
    {
        var valueX = column.GetValue(indexX);
        var valueY = column.GetValue(indexY);

        // Handle null values according to null ordering
        if (valueX == null && valueY == null)
            return 0;

        if (valueX == null)
            return sortKey.NullOrdering == NullOrdering.NullsFirst ? -1 : 1;

        if (valueY == null)
            return sortKey.NullOrdering == NullOrdering.NullsFirst ? 1 : -1;

        // Both values are non-null, compare them
        int comparison;
        if (valueX is IComparable comparableX)
            comparison = comparableX.CompareTo(valueY);
        else
            // Fallback to string comparison
            comparison = string.Compare(valueX.ToString(), valueY.ToString(), StringComparison.Ordinal);

        // Apply sort direction
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
        // Compare using each sort key in order
        foreach (var sortKey in sortKeys)
        {
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
