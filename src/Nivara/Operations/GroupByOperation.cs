using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Expressions;
using Nivara.Helpers;
using Nivara.Query;
using System.Buffers;

namespace Nivara;

/// <summary>
/// Represents grouped data with efficient access to groups and their indices
/// </summary>
internal sealed class GroupedData
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
/// Represents a composite key for grouping operations with proper equality and hashing.
/// Hot paths construct it once per distinct group from typed <see cref="IGroupKeyReader"/>s over
/// the key columns, so no per-row boxed key objects are allocated; <see cref="FromValues"/> boxes
/// for non-hot callers (tests, aggregation fixtures).
/// </summary>
internal sealed class GroupKey : IEquatable<GroupKey>
{
    readonly IReadOnlyList<IGroupKeyReader>? readers;
    readonly int rowIndex;
    readonly IReadOnlyList<object?>? boxedValues;
    readonly int hashCode;
    object?[]? materializedValues;

    /// <summary>
    /// Constructs a typed key from the key-column readers and the row holding the key values,
    /// with a precomputed row hash (avoids re-hashing in bulk grouping).
    /// </summary>
    internal GroupKey(IReadOnlyList<IGroupKeyReader> readers, int rowIndex, int hashCode)
    {
        this.readers = readers;
        this.rowIndex = rowIndex;
        this.hashCode = hashCode;
    }

    /// <summary>
    /// Constructs a typed key from the key-column readers and the row holding the key values.
    /// </summary>
    internal GroupKey(IReadOnlyList<IGroupKeyReader> readers, int rowIndex)
        : this(readers, rowIndex, TypedGroupHash.ComputeRowHash(readers, rowIndex))
    {
    }

    GroupKey(IReadOnlyList<object?> values)
    {
        boxedValues = values;
        int hash = 17;
        foreach (var value in values)
            hash = hash * 31 + (value?.GetHashCode() ?? 0);
        hashCode = hash;
    }

    /// <summary>
    /// Builds a boxed key from literal values for non-hot construction paths.
    /// </summary>
    public static GroupKey FromValues(IReadOnlyList<object?> values)
        => new(values ?? throw new ArgumentNullException(nameof(values)));

    /// <summary>
    /// Gets the number of key columns.
    /// </summary>
    public int KeyCount => readers?.Count ?? boxedValues!.Count;

    /// <summary>
    /// Gets the key value at the given column index (boxed).
    /// </summary>
    public object? GetValue(int index)
        => readers != null ? readers[index].GetValue(rowIndex) : boxedValues![index];

    /// <summary>
    /// Gets the key values (boxed; materialized lazily for typed keys).
    /// </summary>
    public IReadOnlyList<object?> Values
    {
        get
        {
            if (boxedValues != null)
                return boxedValues;

            if (materializedValues == null)
            {
                var values = new object?[readers!.Count];
                for (int i = 0; i < values.Length; i++)
                    values[i] = readers[i].GetValue(rowIndex);
                materializedValues = values;
            }

            return materializedValues;
        }
    }

    /// <inheritdoc />
    public bool Equals(GroupKey? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (KeyCount != other.KeyCount) return false;

        if (readers != null && other.readers != null)
        {
            for (int i = 0; i < readers.Count; i++)
                if (!readers[i].ValuesEqual(rowIndex, other.readers[i], other.rowIndex))
                    return false;
            return true;
        }

        for (int i = 0; i < KeyCount; i++)
            if (!Equals(GetValue(i), other.GetValue(i)))
                return false;
        return true;
    }

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as GroupKey);

    /// <inheritdoc />
    public override int GetHashCode() => hashCode;

    /// <inheritdoc />
    public override string ToString()
    {
        var valueStrings = new string[KeyCount];
        for (int i = 0; i < KeyCount; i++)
            valueStrings[i] = GetValue(i)?.ToString() ?? "null";
        return $"({string.Join(", ", valueStrings)})";
    }
}

/// <summary>
/// Describes a single aggregation applied to grouped rows: the source expression to aggregate,
/// the aggregation function, and the name of the result column in the grouped output.
/// </summary>
internal sealed record GroupedAggregation(string ResultColumnName, ColumnExpression Source, AggregationFunction Function);

/// <summary>
/// Represents a group by operation that groups rows by specified columns with hash-based grouping
/// </summary>
internal sealed class GroupByOperation : IQueryOperation, IParallelGroupByOperation
{
    readonly ColumnExpression[] groupByColumns;
    readonly string[]? keyOutputNames;
    readonly IReadOnlyList<GroupedAggregation>? aggregations;

    /// <summary>
    /// Initializes a new instance of GroupByOperation
    /// </summary>
    /// <param name="groupByColumns">The column expressions to group by</param>
    /// <exception cref="ArgumentNullException">Thrown when groupByColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified</exception>
    public GroupByOperation(ColumnExpression[] groupByColumns)
        : this(groupByColumns, null, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of GroupByOperation with optional aggregation definitions and
    /// explicit key output names. When <paramref name="keyOutputNames"/> is null, key columns are
    /// named after their source expressions (existing behavior). When aggregations are present they
    /// are computed per group and appended to the key columns.
    /// </summary>
    /// <param name="groupByColumns">The column expressions to group by</param>
    /// <param name="keyOutputNames">Optional explicit names for the key result columns</param>
    /// <param name="aggregations">Optional per-group aggregations to compute</param>
    /// <exception cref="ArgumentNullException">Thrown when groupByColumns is null</exception>
    /// <exception cref="ArgumentException">Thrown when no columns are specified, key output names do
    /// not match the key column count, or result column names collide</exception>
    public GroupByOperation(ColumnExpression[] groupByColumns, string[]? keyOutputNames, IReadOnlyList<GroupedAggregation>? aggregations)
    {
        this.groupByColumns = groupByColumns ?? throw new ArgumentNullException(nameof(groupByColumns));

        if (groupByColumns.Length == 0)
            throw new ArgumentException("Must specify at least one column expression for grouping", nameof(groupByColumns));

        if (keyOutputNames != null)
        {
            if (keyOutputNames.Length != groupByColumns.Length)
                throw new ArgumentException("Key output names must match the group-by column count", nameof(keyOutputNames));

            if (keyOutputNames.Any(string.IsNullOrWhiteSpace))
                throw new ArgumentException("Key output names cannot be null or whitespace", nameof(keyOutputNames));

            this.keyOutputNames = keyOutputNames.ToArray();
        }

        if (aggregations is { Count: > 0 })
        {
            var resultNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var keyNames = this.keyOutputNames ?? groupByColumns.Select(c => c.Name).ToArray();

            foreach (var keyName in keyNames)
                resultNames.Add(keyName);

            foreach (var aggregation in aggregations)
            {
                if (string.IsNullOrWhiteSpace(aggregation.ResultColumnName))
                    throw new ArgumentException("Aggregation result column name cannot be null or whitespace", nameof(aggregations));

                if (aggregation.Source is null)
                    throw new ArgumentException($"Aggregation '{aggregation.ResultColumnName}' has a null source expression", nameof(aggregations));

                if (aggregation.Function is null)
                    throw new ArgumentException($"Aggregation '{aggregation.ResultColumnName}' has a null aggregation function", nameof(aggregations));

                if (!resultNames.Add(aggregation.ResultColumnName))
                    throw new ArgumentException($"Duplicate result column name '{aggregation.ResultColumnName}' in group-by aggregations", nameof(aggregations));
            }

            this.aggregations = aggregations.ToList();
        }
    }

    /// <summary>
    /// Gets the column expressions to group by
    /// </summary>
    public IReadOnlyList<ColumnExpression> GroupByColumns => groupByColumns;

    /// <summary>
    /// Gets the explicit key result column names, or null when keys are named after their sources
    /// </summary>
    public IReadOnlyList<string>? KeyOutputNames => keyOutputNames;

    /// <summary>
    /// Gets the per-group aggregations to compute, or null when none are defined
    /// </summary>
    public IReadOnlyList<GroupedAggregation>? Aggregations => aggregations;

    /// <summary>
    /// Gets a value indicating whether this operation computes per-group aggregations
    /// </summary>
    public bool HasAggregations => aggregations is { Count: > 0 };

    public string OperationType => Query.OperationType.GroupBy;

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

        var groupedColumns = new List<(string Name, Type Type)>();

        for (int i = 0; i < GroupByColumns.Count; i++)
        {
            var column = GroupByColumns[i];
            var columnName = GetKeyOutputName(i, column, inputSchema);
            var columnType = GetColumnType(column, inputSchema);
            groupedColumns.Add((columnName, columnType));
        }

        if (aggregations != null)
        {
            foreach (var aggregation in aggregations)
            {
                aggregation.Source.Validate(inputSchema);
                var sourceType = GetColumnType(aggregation.Source, inputSchema);
                groupedColumns.Add((aggregation.ResultColumnName, aggregation.Function.GetResultType(sourceType)));
            }
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

            for (int i = 0; i < keyColumnNames.Length; i++)
            {
                var keyColumnName = keyColumnNames[i];
                var outputName = GetKeyOutputName(i, GroupByColumns[i], input);
                var sourceColumn = input[keyColumnName];
                var distinctValues = ExtractDistinctKeyValues(groupedData, keyColumnName, sourceColumn);
                resultColumns[outputName] = distinctValues;
            }

            if (aggregations != null)
            {
                foreach (var aggregation in aggregations)
                {
                    var sourceName = GetColumnName(aggregation.Source, input);
                    var sourceColumn = input[sourceName];
                    resultColumns[aggregation.ResultColumnName] =
                        aggregation.Function.ApplyToGroups(sourceColumn, groupedData.GetAllGroups());
                }
            }

            return resultColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"GroupBy operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Creates grouped data using typed multi-column key hashing with no per-row boxing.
    /// A <see cref="GroupKey"/> is constructed once per distinct group from its representative row;
    /// rows are bucketed by a typed composite hash and disambiguated with typed equality.
    /// </summary>
    /// <param name="input">The input columns</param>
    /// <param name="keyColumns">The key column names</param>
    /// <returns>The grouped data</returns>
    internal static GroupedData CreateGroupsInternal(IReadOnlyDictionary<string, IColumn> input, string[] keyColumns, int offset = 0)
    {
        var firstColumn = input.Values.First();
        var rowCount = firstColumn.Length;
        var readers = keyColumns.Select(name => GroupKeyReaderFactory.Create(input[name])).ToArray();
        var groups = new Dictionary<GroupKey, List<int>>();
        var hashBuckets = new Dictionary<int, List<int>>();
        var repToKey = new Dictionary<int, GroupKey>();

        var pooled = rowCount > 1024;
        var hashes = pooled ? ArrayPool<int>.Shared.Rent(rowCount) : new int[rowCount];
        try
        {
            TypedGroupHash.ComputeRowHashes(readers, rowCount, hashes);

            for (int rowIndex = 0; rowIndex < rowCount; rowIndex++)
            {
                int hash = hashes[rowIndex];

                if (!hashBuckets.TryGetValue(hash, out var reps))
                {
                    reps = new List<int>(1);
                    hashBuckets[hash] = reps;
                    var groupKey = new GroupKey(readers, rowIndex, hash);
                    repToKey[rowIndex] = groupKey;
                    reps.Add(rowIndex);
                    groups[groupKey] = new List<int>(1) { rowIndex + offset };
                    continue;
                }

                bool joined = false;
                foreach (var rep in reps)
                {
                    if (!TypedGroupHash.RowsEqual(readers, rep, rowIndex))
                        continue;

                    groups[repToKey[rep]].Add(rowIndex + offset);
                    joined = true;
                    break;
                }

                if (!joined)
                {
                    reps.Add(rowIndex);
                    var groupKey = new GroupKey(readers, rowIndex, hash);
                    repToKey[rowIndex] = groupKey;
                    groups[groupKey] = new List<int>(1) { rowIndex + offset };
                }
            }
        }
        finally
        {
            if (pooled)
                ArrayPool<int>.Shared.Return(hashes);
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
            .Select(key => key.GetValue(keyColumnIndex))
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
        return ColumnFactory.Create(elementType, values);
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

    /// <summary>
    /// Gets the output name for a key column at the given index, honoring explicit key output
    /// names when provided and falling back to the expression's source name otherwise.
    /// </summary>
    /// <param name="index">The key column index</param>
    /// <param name="expression">The key column expression</param>
    /// <param name="inputSchema">The input schema (for validation)</param>
    /// <returns>The key result column name</returns>
    string GetKeyOutputName(int index, ColumnExpression expression, Schema inputSchema)
    {
        return keyOutputNames?[index] ?? GetColumnName(expression, inputSchema);
    }

    /// <summary>
    /// Gets the output name for a key column at the given index (runtime version)
    /// </summary>
    /// <param name="index">The key column index</param>
    /// <param name="expression">The key column expression</param>
    /// <param name="input">The input columns</param>
    /// <returns>The key result column name</returns>
    string GetKeyOutputName(int index, ColumnExpression expression, IReadOnlyDictionary<string, IColumn> input)
    {
        return keyOutputNames?[index] ?? GetColumnName(expression, input);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        var columnNames = GroupByColumns.Select(c => c.Name);
        return $"GroupBy({string.Join(", ", columnNames)})";
    }
}
