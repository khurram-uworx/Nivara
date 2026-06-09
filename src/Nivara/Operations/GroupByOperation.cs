using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Query;

namespace Nivara;

/// <summary>
/// Represents grouped data with efficient access to groups and their indices
/// </summary>
public sealed class GroupedData
{
    readonly Dictionary<GroupKey, List<int>> groups;
    readonly string[] keyColumnNames;
    readonly IReadOnlyDictionary<string, IColumn> sourceColumns;

    /// <summary>
    /// Initializes a new instance of GroupedData
    /// </summary>
    /// <param name="groups">The groups with their row indices</param>
    /// <param name="keyColumnNames">The names of the key columns</param>
    /// <param name="sourceColumns">The source columns</param>
    internal GroupedData(Dictionary<GroupKey, List<int>> groups, string[] keyColumnNames, IReadOnlyDictionary<string, IColumn> sourceColumns)
    {
        this.groups = groups ?? throw new ArgumentNullException(nameof(groups));
        this.keyColumnNames = keyColumnNames ?? throw new ArgumentNullException(nameof(keyColumnNames));
        this.sourceColumns = sourceColumns ?? throw new ArgumentNullException(nameof(sourceColumns));
    }

    /// <summary>
    /// Gets the number of groups
    /// </summary>
    public int GroupCount => groups.Count;

    /// <summary>
    /// Gets the names of the key columns
    /// </summary>
    public IReadOnlyList<string> KeyColumnNames => keyColumnNames;

    /// <summary>
    /// Gets all group keys
    /// </summary>
    public IEnumerable<GroupKey> GroupKeys => groups.Keys;

    /// <summary>
    /// Gets the row indices for a specific group
    /// </summary>
    /// <param name="key">The group key</param>
    /// <returns>The row indices for the group</returns>
    public IReadOnlyList<int> GetGroupIndices(GroupKey key)
    {
        return groups.TryGetValue(key, out var indices) ? indices : Array.Empty<int>();
    }

    /// <summary>
    /// Gets all groups with their indices
    /// </summary>
    /// <returns>An enumerable of group key and indices pairs</returns>
    public IEnumerable<(GroupKey Key, IReadOnlyList<int> Indices)> GetAllGroups()
    {
        return groups.Select(kvp => (kvp.Key, (IReadOnlyList<int>)kvp.Value));
    }

    /// <summary>
    /// Gets the source columns
    /// </summary>
    public IReadOnlyDictionary<string, IColumn> SourceColumns => sourceColumns;

    /// <summary>
    /// Gets the internal groups dictionary (for parallel execution merge)
    /// </summary>
    internal Dictionary<GroupKey, List<int>> Groups => groups;
}

/// <summary>
/// Represents a composite key for grouping operations with proper equality and hashing
/// </summary>
public sealed class GroupKey : IEquatable<GroupKey>
{
    readonly object?[] values;
    readonly int hashCode;

    /// <summary>
    /// Initializes a new instance of GroupKey
    /// </summary>
    /// <param name="values">The key values</param>
    public GroupKey(params object?[] values)
    {
        this.values = values ?? throw new ArgumentNullException(nameof(values));
        hashCode = ComputeHashCode(values);
    }

    /// <summary>
    /// Gets the key values
    /// </summary>
    public IReadOnlyList<object?> Values => values;

    /// <inheritdoc />
    public bool Equals(GroupKey? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (values.Length != other.values.Length) return false;

        for (int i = 0; i < values.Length; i++)
            if (!Equals(values[i], other.values[i]))
                return false;

        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as GroupKey);

    /// <inheritdoc />
    public override int GetHashCode() => hashCode;

    /// <summary>
    /// Computes a hash code for the given values
    /// </summary>
    /// <param name="values">The values to hash</param>
    /// <returns>The computed hash code</returns>
    static int ComputeHashCode(object?[] values)
    {
        var hash = new HashCode();
        foreach (var value in values)
            hash.Add(value);

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var valueStrings = values.Select(v => v?.ToString() ?? "null");
        return $"({string.Join(", ", valueStrings)})";
    }
}

/// <summary>
/// Represents a group by operation that groups rows by specified columns with hash-based grouping
/// </summary>
public sealed class GroupByOperation : IQueryOperation
{
    readonly ColumnExpression[] groupByColumns;

    /// <summary>
    /// Initializes a new instance of GroupByOperation
    /// </summary>
    /// <param name="groupByColumns">The column expressions to group by</param>
    /// <exception cref="ArgumentNullException">Thrown when groupByColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public GroupByOperation(ColumnExpression[] groupByColumns)
    {
        this.groupByColumns = groupByColumns ?? throw new ArgumentNullException(nameof(groupByColumns));

        if (groupByColumns.Length == 0)
            throw new ArgumentException("Must specify at least one column expression for grouping", nameof(groupByColumns));
    }

    /// <summary>
    /// Gets the column expressions to group by
    /// </summary>
    public IReadOnlyList<ColumnExpression> GroupByColumns => groupByColumns;

    /// <inheritdoc />
    public string OperationType => "GroupBy";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate all group by column expressions against the schema
        foreach (var column in GroupByColumns)
        {
            try
            {
                column.Validate(inputSchema);
            }
            catch (SchemaValidationException ex)
            {
                throw new SchemaValidationException($"GroupBy column validation failed for '{column.Name}': {ex.Message}");
            }
        }

        // For now, GroupBy returns only the grouped columns
        // In a full implementation, this would include aggregation columns
        var groupedColumns = new List<(string Name, Type Type)>();

        foreach (var column in GroupByColumns)
        {
            var columnName = GetColumnName(column, inputSchema);
            var columnType = GetColumnType(column, inputSchema);
            groupedColumns.Add((columnName, columnType));
        }

        return new Schema(groupedColumns);
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
            // Create grouped data using hash-based grouping
            var keyColumnNames = GroupByColumns.Select(expr => GetColumnName(expr, input)).ToArray();
            var groupedData = CreateGroupsInternal(input, keyColumnNames);

            // Create result columns with distinct key values
            var resultColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

            foreach (var keyColumnName in keyColumnNames)
            {
                var sourceColumn = input[keyColumnName];
                var distinctValues = ExtractDistinctKeyValues(groupedData, keyColumnName, sourceColumn);
                resultColumns[keyColumnName] = distinctValues;
            }

            return resultColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"GroupBy operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Creates grouped data using hash-based grouping with vectorized key comparison
    /// </summary>
    /// <param name="input">The input columns</param>
    /// <param name="keyColumns">The key column names</param>
    /// <returns>The grouped data</returns>
    internal static GroupedData CreateGroupsInternal(IReadOnlyDictionary<string, IColumn> input, string[] keyColumns, int offset = 0)
    {
        var firstColumn = input.Values.First();
        var rowCount = firstColumn.Length;
        var groups = new Dictionary<GroupKey, List<int>>();

        // Get key columns
        var keyColumnData = keyColumns.Select(name => input[name]).ToArray();

        // Group rows by creating composite keys
        for (int rowIndex = 0; rowIndex < rowCount; rowIndex++)
        {
            // Create composite key for this row
            var keyValues = new object?[keyColumns.Length];
            for (int keyIndex = 0; keyIndex < keyColumns.Length; keyIndex++)
                keyValues[keyIndex] = keyColumnData[keyIndex].GetValue(rowIndex);

            var groupKey = new GroupKey(keyValues);

            // Add row to appropriate group
            if (!groups.TryGetValue(groupKey, out var rowIndices))
            {
                rowIndices = new List<int>();
                groups[groupKey] = rowIndices;
            }

            rowIndices.Add(rowIndex + offset);
        }

        return new GroupedData(groups, keyColumns, input);
    }

    /// <summary>
    /// Extracts distinct key values from grouped data for a specific column
    /// </summary>
    /// <param name="groupedData">The grouped data</param>
    /// <param name="columnName">The column name</param>
    /// <param name="sourceColumn">The source column</param>
    /// <returns>A column with distinct key values</returns>
    internal static IColumn ExtractDistinctKeyValues(GroupedData groupedData, string columnName, IColumn sourceColumn)
    {
        var keyColumnIndex = Array.IndexOf(groupedData.KeyColumnNames.ToArray(), columnName);
        if (keyColumnIndex == -1)
            throw new ArgumentException($"Column '{columnName}' is not a key column", nameof(columnName));

        var distinctValues = groupedData.GroupKeys
            .Select(key => key.Values[keyColumnIndex])
            .ToArray();

        return CreateColumnFromValues(sourceColumn.ElementType, distinctValues);
    }

    /// <summary>
    /// Creates a column from an array of values with proper type handling
    /// </summary>
    /// <param name="elementType">The element type</param>
    /// <param name="values">The values</param>
    /// <returns>A new column</returns>
    internal static IColumn CreateColumnFromValues(Type elementType, object?[] values)
    {
        return elementType switch
        {
            Type t when t == typeof(int) => NivaraColumn<int>.Create(values.Cast<int>().ToArray()),
            Type t when t == typeof(double) => NivaraColumn<double>.Create(values.Cast<double>().ToArray()),
            Type t when t == typeof(float) => NivaraColumn<float>.Create(values.Cast<float>().ToArray()),
            Type t when t == typeof(long) => NivaraColumn<long>.Create(values.Cast<long>().ToArray()),
            Type t when t == typeof(string) => NivaraColumn<string>.Create(values.Cast<string>().ToArray()),
            Type t when t == typeof(bool) => NivaraColumn<bool>.Create(values.Cast<bool>().ToArray()),
            Type t when t == typeof(decimal) => NivaraColumn<decimal>.Create(values.Cast<decimal>().ToArray()),
            Type t when t == typeof(byte) => NivaraColumn<byte>.Create(values.Cast<byte>().ToArray()),
            Type t when t == typeof(short) => NivaraColumn<short>.Create(values.Cast<short>().ToArray()),
            Type t when t == typeof(DateTime) => NivaraColumn<DateTime>.Create(values.Cast<DateTime>().ToArray()),
            _ => NivaraColumn<object>.Create(values.Where(v => v != null).ToArray()!)
        };
    }

    /// <summary>
    /// Gets the name for a column expression in the result schema
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="inputSchema">The input schema</param>
    /// <returns>The column name to use in the result</returns>
    static string GetColumnName(ColumnExpression expression, Schema inputSchema)
    {
        // For simple column references, use the original column name
        if (expression is ColumnReference columnRef)
            return columnRef.ColumnName;

        // For complex expressions, use the expression's display name
        return expression.Name;
    }

    /// <summary>
    /// Gets the name for a column expression in the result (runtime version)
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>The column name to use in the result</returns>
    static string GetColumnName(ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        // For simple column references, use the original column name
        if (expression is ColumnReference columnRef)
            return columnRef.ColumnName;

        // For complex expressions, use the expression's display name
        return expression.Name;
    }

    /// <summary>
    /// Gets the type for a column expression in the result schema
    /// </summary>
    /// <param name="expression">The column expression</param>
    /// <param name="inputSchema">The input schema</param>
    /// <returns>The column type in the result</returns>
    static Type GetColumnType(ColumnExpression expression, Schema inputSchema)
    {
        // For simple column references, get the type from the schema
        if (expression is ColumnReference columnRef)
            return inputSchema.GetColumnType(columnRef.ColumnName);

        // For other expressions, use the expression's result type
        return expression.ResultType;
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var columnNames = GroupByColumns.Select(c => c.Name);
        return $"GroupBy({string.Join(", ", columnNames)})";
    }
}
