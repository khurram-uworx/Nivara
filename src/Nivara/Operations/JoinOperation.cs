using Nivara.Exceptions;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Represents the type of join operation to perform
/// </summary>
public enum JoinType
{
    /// <summary>
    /// Inner join - returns only rows where join keys exist in both DataFrames
    /// </summary>
    Inner,

    /// <summary>
    /// Left join - returns all rows from the left DataFrame with matching rows from the right
    /// </summary>
    Left,

    /// <summary>
    /// Right join - returns all rows from the right DataFrame with matching rows from the left
    /// </summary>
    Right,

    /// <summary>
    /// Full outer join - returns all rows from both DataFrames with nulls for non-matching rows
    /// </summary>
    FullOuter
}

/// <summary>
/// Represents a join key mapping between columns in two DataFrames
/// </summary>
public sealed class JoinKey
{
    /// <summary>
    /// Initializes a new instance of JoinKey
    /// </summary>
    /// <param name="leftColumn">The column name in the left DataFrame</param>
    /// <param name="rightColumn">The column name in the right DataFrame</param>
    /// <exception cref="ArgumentException">Thrown when column names are null or whitespace</exception>
    public JoinKey(string leftColumn, string rightColumn)
    {
        if (string.IsNullOrWhiteSpace(leftColumn))
            throw new ArgumentException("Left column name cannot be null or whitespace", nameof(leftColumn));
        if (string.IsNullOrWhiteSpace(rightColumn))
            throw new ArgumentException("Right column name cannot be null or whitespace", nameof(rightColumn));

        LeftColumn = leftColumn;
        RightColumn = rightColumn;
    }

    /// <summary>
    /// Initializes a new instance of JoinKey with the same column name for both sides
    /// </summary>
    /// <param name="columnName">The column name to use for both left and right DataFrames</param>
    /// <exception cref="ArgumentException">Thrown when column name is null or whitespace</exception>
    public JoinKey(string columnName) : this(columnName, columnName)
    {
    }

    /// <summary>
    /// Gets the column name in the left DataFrame
    /// </summary>
    public string LeftColumn { get; }

    /// <summary>
    /// Gets the column name in the right DataFrame
    /// </summary>
    public string RightColumn { get; }

    /// <summary>
    /// Returns a string representation of the join key
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        return LeftColumn == RightColumn ? LeftColumn : $"{LeftColumn}={RightColumn}";
    }
}

/// <summary>
/// Represents a strategy for handling column name conflicts during joins
/// </summary>
public enum ColumnDisambiguationStrategy
{
    /// <summary>
    /// Add prefixes to conflicting columns (left_column, right_column)
    /// </summary>
    Prefix,

    /// <summary>
    /// Add suffixes to conflicting columns (column_left, column_right)
    /// </summary>
    Suffix,

    /// <summary>
    /// Throw an exception when column name conflicts occur
    /// </summary>
    Error
}

/// <summary>
/// Represents the result of computing join indices
/// </summary>
internal sealed class JoinIndices
{
    /// <summary>
    /// Initializes a new instance of JoinIndices
    /// </summary>
    /// <param name="leftIndices">Indices from the left DataFrame</param>
    /// <param name="rightIndices">Indices from the right DataFrame</param>
    public JoinIndices(int[] leftIndices, int[] rightIndices)
    {
        LeftIndices = leftIndices ?? throw new ArgumentNullException(nameof(leftIndices));
        RightIndices = rightIndices ?? throw new ArgumentNullException(nameof(rightIndices));

        if (leftIndices.Length != rightIndices.Length)
            throw new ArgumentException("Left and right indices must have the same length");
    }

    /// <summary>
    /// Gets the indices from the left DataFrame
    /// </summary>
    public int[] LeftIndices { get; }

    /// <summary>
    /// Gets the indices from the right DataFrame
    /// </summary>
    public int[] RightIndices { get; }

    /// <summary>
    /// Gets the number of result rows
    /// </summary>
    public int Count => LeftIndices.Length;
}

/// <summary>
/// Represents a join operation between two DataFrames
/// </summary>
internal sealed class JoinOperation : IQueryOperation
{
    readonly IReadOnlyDictionary<string, IColumn> leftColumns;
    readonly IReadOnlyDictionary<string, IColumn> rightColumns;
    readonly JoinType joinType;
    readonly JoinKey[] joinKeys;
    readonly ColumnDisambiguationStrategy disambiguationStrategy;
    readonly string leftPrefix;
    readonly string rightPrefix;

    /// <summary>
    /// Initializes a new instance of JoinOperation
    /// </summary>
    /// <param name="leftColumns">The columns from the left DataFrame</param>
    /// <param name="rightColumns">The columns from the right DataFrame</param>
    /// <param name="joinType">The type of join to perform</param>
    /// <param name="joinKeys">The join keys</param>
    /// <param name="disambiguationStrategy">The strategy for handling column name conflicts</param>
    /// <param name="leftPrefix">The prefix for left columns (when using prefix strategy)</param>
    /// <param name="rightPrefix">The prefix for right columns (when using prefix strategy)</param>
    public JoinOperation(
        IReadOnlyDictionary<string, IColumn> leftColumns,
        IReadOnlyDictionary<string, IColumn> rightColumns,
        JoinType joinType,
        JoinKey[] joinKeys,
        ColumnDisambiguationStrategy disambiguationStrategy = ColumnDisambiguationStrategy.Suffix,
        string leftPrefix = "left",
        string rightPrefix = "right")
    {
        this.leftColumns = leftColumns ?? throw new ArgumentNullException(nameof(leftColumns));
        this.rightColumns = rightColumns ?? throw new ArgumentNullException(nameof(rightColumns));
        this.joinType = joinType;
        this.joinKeys = joinKeys ?? throw new ArgumentNullException(nameof(joinKeys));
        this.disambiguationStrategy = disambiguationStrategy;
        this.leftPrefix = leftPrefix ?? "left";
        this.rightPrefix = rightPrefix ?? "right";

        if (joinKeys.Length == 0)
            throw new ArgumentException("Must specify at least one join key", nameof(joinKeys));
    }

    /// <inheritdoc />
    public string OperationType => "Join";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        // For join operations, we need to validate both left and right schemas
        // This is a simplified implementation - in practice, we'd need both schemas
        var leftSchema = new Schema(leftColumns.Select(kvp => (kvp.Key, kvp.Value.ElementType)));
        var rightSchema = new Schema(rightColumns.Select(kvp => (kvp.Key, kvp.Value.ElementType)));

        // Validate join keys exist in both schemas
        foreach (var joinKey in joinKeys)
        {
            if (!leftSchema.HasColumn(joinKey.LeftColumn))
            {
                throw new SchemaValidationException(
                    $"Left join key column '{joinKey.LeftColumn}' not found in left schema. Available columns: {string.Join(", ", leftSchema.ColumnNames)}");
            }

            if (!rightSchema.HasColumn(joinKey.RightColumn))
            {
                throw new SchemaValidationException(
                    $"Right join key column '{joinKey.RightColumn}' not found in right schema. Available columns: {string.Join(", ", rightSchema.ColumnNames)}");
            }

            // Validate join key types are compatible
            var leftType = leftSchema.GetColumnType(joinKey.LeftColumn);
            var rightType = rightSchema.GetColumnType(joinKey.RightColumn);

            if (!AreTypesCompatibleForJoin(leftType, rightType))
            {
                throw new SchemaValidationException(
                    $"Join key types are incompatible: left column '{joinKey.LeftColumn}' is {leftType.Name}, right column '{joinKey.RightColumn}' is {rightType.Name}");
            }
        }

        // Build result schema with column disambiguation
        var resultColumns = new List<(string Name, Type Type)>();

        // First, identify all potential column names to detect conflicts
        var leftColumnNames = leftColumns.Keys.ToList();
        var rightColumnNames = rightColumns.Keys.Where(name =>
            !joinKeys.Any(jk => string.Equals(jk.RightColumn, name, StringComparison.OrdinalIgnoreCase)))
            .ToList();

        // Add left columns
        foreach (var kvp in leftColumns)
        {
            var columnName = ResolveColumnName(kvp.Key, true, leftColumnNames, rightColumnNames);
            resultColumns.Add((columnName, kvp.Value.ElementType));
        }

        // Add right columns (excluding join keys that are already included from left)
        foreach (var kvp in rightColumns)
        {
            // Skip right join key columns - they are always excluded since left join keys are included
            bool isJoinKey = joinKeys.Any(jk =>
                string.Equals(jk.RightColumn, kvp.Key, StringComparison.OrdinalIgnoreCase));

            if (!isJoinKey)
            {
                var columnName = ResolveColumnName(kvp.Key, false, leftColumnNames, rightColumnNames);
                resultColumns.Add((columnName, kvp.Value.ElementType));
            }
        }

        return new Schema(resultColumns);
    }

    /// <inheritdoc />
    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        // Note: For join operations, the input parameter is not used as we work with two separate DataFrames
        // This is a limitation of the current IQueryOperation interface design

        try
        {
            // Validate join keys exist in both DataFrames and check type compatibility
            foreach (var joinKey in joinKeys)
            {
                if (!leftColumns.ContainsKey(joinKey.LeftColumn))
                {
                    throw new SchemaValidationException(
                        $"Left join key column '{joinKey.LeftColumn}' not found in left DataFrame. Available columns: {string.Join(", ", leftColumns.Keys)}");
                }

                if (!rightColumns.ContainsKey(joinKey.RightColumn))
                {
                    throw new SchemaValidationException(
                        $"Right join key column '{joinKey.RightColumn}' not found in right DataFrame. Available columns: {string.Join(", ", rightColumns.Keys)}");
                }

                // Validate join key types are compatible
                var leftType = leftColumns[joinKey.LeftColumn].ElementType;
                var rightType = rightColumns[joinKey.RightColumn].ElementType;

                if (!AreTypesCompatibleForJoin(leftType, rightType))
                {
                    throw new SchemaValidationException(
                        $"Join key types are incompatible: left column '{joinKey.LeftColumn}' is {leftType.Name}, right column '{joinKey.RightColumn}' is {rightType.Name}");
                }
            }

            // Compute join indices
            var joinIndices = ComputeJoinIndices();

            // Materialize the join result
            return MaterializeJoinResult(joinIndices);
        }
        catch (Exception ex) when (ex is not QueryExecutionException and not SchemaValidationException)
        {
            throw new QueryExecutionException($"Join operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Computes the join indices based on the join type and keys
    /// </summary>
    /// <returns>The computed join indices</returns>
    private JoinIndices ComputeJoinIndices()
    {
        // Get row counts
        var leftRowCount = leftColumns.Values.FirstOrDefault()?.Length ?? 0;
        var rightRowCount = rightColumns.Values.FirstOrDefault()?.Length ?? 0;

        if (leftRowCount == 0 || rightRowCount == 0)
        {
            // Handle empty DataFrames
            return HandleEmptyDataFrames(leftRowCount, rightRowCount);
        }

        // Build hash map for right DataFrame
        var rightHashMap = BuildHashMap(rightColumns, joinKeys.Select(jk => jk.RightColumn).ToArray(), rightRowCount);

        // Perform join based on type
        return joinType switch
        {
            JoinType.Inner => ComputeInnerJoinIndices(leftRowCount, rightHashMap),
            JoinType.Left => ComputeLeftJoinIndices(leftRowCount, rightHashMap),
            JoinType.Right => ComputeRightJoinIndices(leftRowCount, rightRowCount, rightHashMap),
            JoinType.FullOuter => ComputeFullOuterJoinIndices(leftRowCount, rightRowCount, rightHashMap),
            _ => throw new ArgumentException($"Unsupported join type: {joinType}")
        };
    }

    /// <summary>
    /// Handles join operations when one or both DataFrames are empty
    /// </summary>
    private JoinIndices HandleEmptyDataFrames(int leftRowCount, int rightRowCount)
    {
        return joinType switch
        {
            JoinType.Inner => new JoinIndices(Array.Empty<int>(), Array.Empty<int>()),
            JoinType.Left when leftRowCount > 0 => CreateLeftOnlyIndices(leftRowCount),
            JoinType.Right when rightRowCount > 0 => CreateRightOnlyIndices(rightRowCount),
            JoinType.FullOuter => CreateFullOuterEmptyIndices(leftRowCount, rightRowCount),
            _ => new JoinIndices(Array.Empty<int>(), Array.Empty<int>())
        };
    }

    /// <summary>
    /// Creates indices for left-only rows (used in left and full outer joins)
    /// </summary>
    private static JoinIndices CreateLeftOnlyIndices(int leftRowCount)
    {
        var leftIndices = Enumerable.Range(0, leftRowCount).ToArray();
        var rightIndices = new int[leftRowCount];
        Array.Fill(rightIndices, -1); // -1 indicates no match (null)
        return new JoinIndices(leftIndices, rightIndices);
    }

    /// <summary>
    /// Creates indices for right-only rows (used in right and full outer joins)
    /// </summary>
    private static JoinIndices CreateRightOnlyIndices(int rightRowCount)
    {
        var leftIndices = new int[rightRowCount];
        Array.Fill(leftIndices, -1); // -1 indicates no match (null)
        var rightIndices = Enumerable.Range(0, rightRowCount).ToArray();
        return new JoinIndices(leftIndices, rightIndices);
    }

    /// <summary>
    /// Creates indices for full outer join with empty DataFrames
    /// </summary>
    private static JoinIndices CreateFullOuterEmptyIndices(int leftRowCount, int rightRowCount)
    {
        var totalRows = leftRowCount + rightRowCount;
        var leftIndices = new int[totalRows];
        var rightIndices = new int[totalRows];

        // Fill left part
        for (int i = 0; i < leftRowCount; i++)
        {
            leftIndices[i] = i;
            rightIndices[i] = -1;
        }

        // Fill right part
        for (int i = 0; i < rightRowCount; i++)
        {
            leftIndices[leftRowCount + i] = -1;
            rightIndices[leftRowCount + i] = i;
        }

        return new JoinIndices(leftIndices, rightIndices);
    }

    /// <summary>
    /// Builds a hash map for efficient join key lookup
    /// </summary>
    private Dictionary<CompositeKey, List<int>> BuildHashMap(
        IReadOnlyDictionary<string, IColumn> columns,
        string[] keyColumns,
        int rowCount)
    {
        var hashMap = new Dictionary<CompositeKey, List<int>>();

        for (int i = 0; i < rowCount; i++)
        {
            var keyValues = new object?[keyColumns.Length];
            for (int j = 0; j < keyColumns.Length; j++)
            {
                keyValues[j] = columns[keyColumns[j]].GetValue(i);
            }

            var compositeKey = new CompositeKey(keyValues);

            if (!hashMap.TryGetValue(compositeKey, out var indices))
            {
                indices = new List<int>();
                hashMap[compositeKey] = indices;
            }

            indices.Add(i);
        }

        return hashMap;
    }

    /// <summary>
    /// Computes indices for inner join
    /// </summary>
    private JoinIndices ComputeInnerJoinIndices(int leftRowCount, Dictionary<CompositeKey, List<int>> rightHashMap)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        var leftKeyColumns = joinKeys.Select(jk => jk.LeftColumn).ToArray();

        for (int leftIndex = 0; leftIndex < leftRowCount; leftIndex++)
        {
            var leftKeyValues = new object?[leftKeyColumns.Length];
            for (int j = 0; j < leftKeyColumns.Length; j++)
            {
                leftKeyValues[j] = leftColumns[leftKeyColumns[j]].GetValue(leftIndex);
            }

            var leftKey = new CompositeKey(leftKeyValues);

            if (rightHashMap.TryGetValue(leftKey, out var matchingRightIndices))
            {
                foreach (var rightIndex in matchingRightIndices)
                {
                    leftIndices.Add(leftIndex);
                    rightIndices.Add(rightIndex);
                }
            }
        }

        return new JoinIndices(leftIndices.ToArray(), rightIndices.ToArray());
    }

    /// <summary>
    /// Computes indices for left join
    /// </summary>
    private JoinIndices ComputeLeftJoinIndices(int leftRowCount, Dictionary<CompositeKey, List<int>> rightHashMap)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        var leftKeyColumns = joinKeys.Select(jk => jk.LeftColumn).ToArray();

        for (int leftIndex = 0; leftIndex < leftRowCount; leftIndex++)
        {
            var leftKeyValues = new object?[leftKeyColumns.Length];
            for (int j = 0; j < leftKeyColumns.Length; j++)
            {
                leftKeyValues[j] = leftColumns[leftKeyColumns[j]].GetValue(leftIndex);
            }

            var leftKey = new CompositeKey(leftKeyValues);

            if (rightHashMap.TryGetValue(leftKey, out var matchingRightIndices))
            {
                foreach (var rightIndex in matchingRightIndices)
                {
                    leftIndices.Add(leftIndex);
                    rightIndices.Add(rightIndex);
                }
            }
            else
            {
                // No match found, include left row with null for right
                leftIndices.Add(leftIndex);
                rightIndices.Add(-1);
            }
        }

        return new JoinIndices(leftIndices.ToArray(), rightIndices.ToArray());
    }

    /// <summary>
    /// Computes indices for right join
    /// </summary>
    private JoinIndices ComputeRightJoinIndices(int leftRowCount, int rightRowCount, Dictionary<CompositeKey, List<int>> rightHashMap)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        // First, find all matches (same as inner join)
        var matchedRightIndices = new HashSet<int>();
        var leftKeyColumns = joinKeys.Select(jk => jk.LeftColumn).ToArray();

        for (int leftIndex = 0; leftIndex < leftRowCount; leftIndex++)
        {
            var leftKeyValues = new object?[leftKeyColumns.Length];
            for (int j = 0; j < leftKeyColumns.Length; j++)
            {
                leftKeyValues[j] = leftColumns[leftKeyColumns[j]].GetValue(leftIndex);
            }

            var leftKey = new CompositeKey(leftKeyValues);

            if (rightHashMap.TryGetValue(leftKey, out var matchingRightIndices))
            {
                foreach (var rightIndex in matchingRightIndices)
                {
                    leftIndices.Add(leftIndex);
                    rightIndices.Add(rightIndex);
                    matchedRightIndices.Add(rightIndex);
                }
            }
        }

        // Add unmatched right rows
        for (int rightIndex = 0; rightIndex < rightRowCount; rightIndex++)
        {
            if (!matchedRightIndices.Contains(rightIndex))
            {
                leftIndices.Add(-1);
                rightIndices.Add(rightIndex);
            }
        }

        return new JoinIndices(leftIndices.ToArray(), rightIndices.ToArray());
    }

    /// <summary>
    /// Computes indices for full outer join
    /// </summary>
    private JoinIndices ComputeFullOuterJoinIndices(int leftRowCount, int rightRowCount, Dictionary<CompositeKey, List<int>> rightHashMap)
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();

        // First, find all matches and track matched right indices
        var matchedRightIndices = new HashSet<int>();
        var leftKeyColumns = joinKeys.Select(jk => jk.LeftColumn).ToArray();

        for (int leftIndex = 0; leftIndex < leftRowCount; leftIndex++)
        {
            var leftKeyValues = new object?[leftKeyColumns.Length];
            for (int j = 0; j < leftKeyColumns.Length; j++)
            {
                leftKeyValues[j] = leftColumns[leftKeyColumns[j]].GetValue(leftIndex);
            }

            var leftKey = new CompositeKey(leftKeyValues);

            if (rightHashMap.TryGetValue(leftKey, out var matchingRightIndices))
            {
                foreach (var rightIndex in matchingRightIndices)
                {
                    leftIndices.Add(leftIndex);
                    rightIndices.Add(rightIndex);
                    matchedRightIndices.Add(rightIndex);
                }
            }
            else
            {
                // No match found, include left row with null for right
                leftIndices.Add(leftIndex);
                rightIndices.Add(-1);
            }
        }

        // Add unmatched right rows
        for (int rightIndex = 0; rightIndex < rightRowCount; rightIndex++)
        {
            if (!matchedRightIndices.Contains(rightIndex))
            {
                leftIndices.Add(-1);
                rightIndices.Add(rightIndex);
            }
        }

        return new JoinIndices(leftIndices.ToArray(), rightIndices.ToArray());
    }

    /// <summary>
    /// Materializes the join result using the computed indices
    /// </summary>
    private IReadOnlyDictionary<string, IColumn> MaterializeJoinResult(JoinIndices joinIndices)
    {
        var resultColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
        var usedNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        // First, identify all potential column names to detect conflicts
        var leftColumnNames = leftColumns.Keys.ToList();
        var rightColumnNames = rightColumns.Keys.Where(name =>
            !joinKeys.Any(jk => string.Equals(jk.RightColumn, name, StringComparison.OrdinalIgnoreCase)))
            .ToList();

        // Add left columns (with special handling for join keys in outer joins)
        foreach (var kvp in leftColumns)
        {
            var columnName = ResolveColumnName(kvp.Key, true, leftColumnNames, rightColumnNames);

            // Check if this is a join key column that needs special handling for outer joins
            var joinKey = joinKeys.FirstOrDefault(jk =>
                string.Equals(jk.LeftColumn, kvp.Key, StringComparison.OrdinalIgnoreCase));

            IColumn resultColumn;
            if (joinKey != null && (joinType == JoinType.Right || joinType == JoinType.FullOuter))
            {
                // For right and full outer joins, coalesce join key values
                resultColumn = CreateCoalescedJoinKeyColumn(kvp.Value, rightColumns[joinKey.RightColumn], joinIndices);
            }
            else
            {
                resultColumn = GatherColumn(kvp.Value, joinIndices.LeftIndices);
            }

            resultColumns[columnName] = resultColumn;
            usedNames.Add(columnName);
        }

        // Add right columns (excluding join keys that are already included from left)
        foreach (var kvp in rightColumns)
        {
            // Skip right join key columns - they are always excluded since left join keys are included
            bool isJoinKey = joinKeys.Any(jk =>
                string.Equals(jk.RightColumn, kvp.Key, StringComparison.OrdinalIgnoreCase));

            if (!isJoinKey)
            {
                var columnName = ResolveColumnName(kvp.Key, false, leftColumnNames, rightColumnNames);
                var resultColumn = GatherColumn(kvp.Value, joinIndices.RightIndices);
                resultColumns[columnName] = resultColumn;
                usedNames.Add(columnName);
            }
        }

        return resultColumns;
    }

    /// <summary>
    /// Creates a coalesced join key column that uses left values when available, otherwise right values
    /// </summary>
    private IColumn CreateCoalescedJoinKeyColumn(IColumn leftColumn, IColumn rightColumn, JoinIndices joinIndices)
    {
        var elementType = leftColumn.ElementType;

        // Use dynamic dispatch to create the appropriate column type
        return elementType switch
        {
            Type t when t == typeof(int) => CreateCoalescedJoinKeyColumnTyped<int>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(double) => CreateCoalescedJoinKeyColumnTyped<double>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(float) => CreateCoalescedJoinKeyColumnTyped<float>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(long) => CreateCoalescedJoinKeyColumnTyped<long>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(string) => CreateCoalescedJoinKeyColumnTyped<string>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(bool) => CreateCoalescedJoinKeyColumnTyped<bool>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(decimal) => CreateCoalescedJoinKeyColumnTyped<decimal>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(byte) => CreateCoalescedJoinKeyColumnTyped<byte>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(short) => CreateCoalescedJoinKeyColumnTyped<short>(leftColumn, rightColumn, joinIndices),
            Type t when t == typeof(DateTime) => CreateCoalescedJoinKeyColumnTyped<DateTime>(leftColumn, rightColumn, joinIndices),
            _ => CreateCoalescedJoinKeyColumnGeneric(leftColumn, rightColumn, joinIndices)
        };
    }

    /// <summary>
    /// Creates a coalesced join key column for a specific type
    /// </summary>
    static IColumn CreateCoalescedJoinKeyColumnTyped<T>(IColumn leftColumn, IColumn rightColumn, JoinIndices joinIndices)
    {
        // Check if T is a value type to determine which creation method to use
        if (typeof(T).IsValueType)
        {
            // For value types, create nullable array and use CreateFromNullable
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var coalescedArray = System.Array.CreateInstance(nullableType, joinIndices.Count);

            for (int i = 0; i < joinIndices.Count; i++)
            {
                object? value = null;

                // Use left value if available, otherwise use right value
                if (joinIndices.LeftIndices[i] >= 0)
                {
                    value = leftColumn.GetValue(joinIndices.LeftIndices[i]);
                }
                else if (joinIndices.RightIndices[i] >= 0)
                {
                    value = rightColumn.GetValue(joinIndices.RightIndices[i]);
                }

                if (value != null)
                {
                    var nullableInstance = Activator.CreateInstance(nullableType, value);
                    coalescedArray.SetValue(nullableInstance, i);
                }
                // null values remain null in the array
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { coalescedArray })!;
        }
        else
        {
            // For reference types, create regular array and use CreateForReferenceType
            var coalescedArray = new T[joinIndices.Count];

            for (int i = 0; i < joinIndices.Count; i++)
            {
                // Use left value if available, otherwise use right value
                if (joinIndices.LeftIndices[i] >= 0)
                {
                    var value = leftColumn.GetValue(joinIndices.LeftIndices[i]);
                    coalescedArray[i] = (T)value!;
                }
                else if (joinIndices.RightIndices[i] >= 0)
                {
                    var value = rightColumn.GetValue(joinIndices.RightIndices[i]);
                    coalescedArray[i] = (T)value!;
                }
                // null values remain null in the array (default for reference types)
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { coalescedArray })!;
        }
    }

    /// <summary>
    /// Creates a coalesced join key column for unknown types using object column
    /// </summary>
    static IColumn CreateCoalescedJoinKeyColumnGeneric(IColumn leftColumn, IColumn rightColumn, JoinIndices joinIndices)
    {
        var coalescedArray = new object[joinIndices.Count];

        for (int i = 0; i < joinIndices.Count; i++)
        {
            // Use left value if available, otherwise use right value
            if (joinIndices.LeftIndices[i] >= 0)
            {
                coalescedArray[i] = leftColumn.GetValue(joinIndices.LeftIndices[i])!;
            }
            else if (joinIndices.RightIndices[i] >= 0)
            {
                coalescedArray[i] = rightColumn.GetValue(joinIndices.RightIndices[i])!;
            }
            // null values remain null in the array
        }

        return NivaraColumn<object>.Create(coalescedArray);
    }

    /// <summary>
    /// Resolves column name conflicts using the disambiguation strategy
    /// </summary>
    private string ResolveColumnName(string originalName, bool isLeft, List<string> leftColumnNames, List<string> rightColumnNames)
    {
        // Check if there's a conflict between left and right column names
        bool hasConflict = leftColumnNames.Any(name => string.Equals(name, originalName, StringComparison.OrdinalIgnoreCase)) &&
                          rightColumnNames.Any(name => string.Equals(name, originalName, StringComparison.OrdinalIgnoreCase));

        if (!hasConflict)
        {
            return originalName;
        }

        return disambiguationStrategy switch
        {
            ColumnDisambiguationStrategy.Prefix => isLeft ? $"{leftPrefix}_{originalName}" : $"{rightPrefix}_{originalName}",
            ColumnDisambiguationStrategy.Suffix => isLeft ? $"{originalName}_{leftPrefix}" : $"{originalName}_{rightPrefix}",
            ColumnDisambiguationStrategy.Error => throw new SchemaValidationException(
                $"Column name conflict: '{originalName}' exists in both left and right DataFrames. Use a different disambiguation strategy or rename columns."),
            _ => throw new ArgumentException($"Unknown disambiguation strategy: {disambiguationStrategy}")
        };
    }

    /// <summary>
    /// Gathers values from a column using the specified indices
    /// </summary>
    private static IColumn GatherColumn(IColumn sourceColumn, int[] indices)
    {
        var elementType = sourceColumn.ElementType;

        // Use dynamic dispatch to create the appropriate column type
        return elementType switch
        {
            Type t when t == typeof(int) => GatherColumnTyped<int>(sourceColumn, indices),
            Type t when t == typeof(double) => GatherColumnTyped<double>(sourceColumn, indices),
            Type t when t == typeof(float) => GatherColumnTyped<float>(sourceColumn, indices),
            Type t when t == typeof(long) => GatherColumnTyped<long>(sourceColumn, indices),
            Type t when t == typeof(string) => GatherColumnTyped<string>(sourceColumn, indices),
            Type t when t == typeof(bool) => GatherColumnTyped<bool>(sourceColumn, indices),
            Type t when t == typeof(decimal) => GatherColumnTyped<decimal>(sourceColumn, indices),
            Type t when t == typeof(byte) => GatherColumnTyped<byte>(sourceColumn, indices),
            Type t when t == typeof(short) => GatherColumnTyped<short>(sourceColumn, indices),
            Type t when t == typeof(DateTime) => GatherColumnTyped<DateTime>(sourceColumn, indices),
            _ => GatherColumnGeneric(sourceColumn, indices)
        };
    }

    /// <summary>
    /// Gathers values from a column for a specific type
    /// </summary>
    static IColumn GatherColumnTyped<T>(IColumn sourceColumn, int[] indices)
    {
        // Check if T is a value type to determine which creation method to use
        if (typeof(T).IsValueType)
        {
            // For value types, create nullable array and use CreateFromNullable
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var gatheredArray = System.Array.CreateInstance(nullableType, indices.Length);

            for (int i = 0; i < indices.Length; i++)
            {
                if (indices[i] >= 0) // -1 indicates null (no match in join)
                {
                    var value = sourceColumn.GetValue(indices[i]);
                    if (value != null)
                    {
                        var nullableInstance = Activator.CreateInstance(nullableType, value);
                        gatheredArray.SetValue(nullableInstance, i);
                    }
                }
                // null values remain null in the array
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { gatheredArray })!;
        }
        else
        {
            // For reference types, create regular array and use CreateForReferenceType
            var gatheredArray = new T[indices.Length];

            for (int i = 0; i < indices.Length; i++)
            {
                if (indices[i] >= 0) // -1 indicates null (no match in join)
                {
                    var value = sourceColumn.GetValue(indices[i]);
                    gatheredArray[i] = (T)value!; // Reference types can be null
                }
                // null values remain null in the array (default for reference types)
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { gatheredArray })!;
        }
    }

    /// <summary>
    /// Gathers values from a column for unknown types using object column
    /// </summary>
    static IColumn GatherColumnGeneric(IColumn sourceColumn, int[] indices)
    {
        var gatheredArray = new object[indices.Length];

        for (int i = 0; i < indices.Length; i++)
        {
            if (indices[i] >= 0) // -1 indicates null (no match in join)
            {
                gatheredArray[i] = sourceColumn.GetValue(indices[i])!;
            }
            // null values remain null in the array
        }

        return NivaraColumn<object>.Create(gatheredArray);
    }

    /// <summary>
    /// Checks if two types are compatible for join operations
    /// </summary>
    private static bool AreTypesCompatibleForJoin(Type leftType, Type rightType)
    {
        // Handle nullable types
        var leftUnderlying = Nullable.GetUnderlyingType(leftType) ?? leftType;
        var rightUnderlying = Nullable.GetUnderlyingType(rightType) ?? rightType;

        // Types must be exactly the same for join operations
        return leftUnderlying == rightUnderlying;
    }

    /// <summary>
    /// Returns a string representation of the join operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var keysStr = string.Join(", ", joinKeys.Select(jk => jk.ToString()));
        return $"{joinType}Join({keysStr})";
    }
}

/// <summary>
/// Represents a composite key for join operations with proper equality and hashing
/// </summary>
internal sealed class CompositeKey : IEquatable<CompositeKey>
{
    readonly object?[] values;
    readonly int hashCode;

    /// <summary>
    /// Initializes a new instance of CompositeKey
    /// </summary>
    /// <param name="values">The key values</param>
    public CompositeKey(object?[] values)
    {
        this.values = values ?? throw new ArgumentNullException(nameof(values));

        // Pre-compute hash code for performance
        var hash = new HashCode();
        foreach (var value in values)
        {
            hash.Add(value);
        }
        hashCode = hash.ToHashCode();
    }

    /// <summary>
    /// Determines whether the specified CompositeKey is equal to the current CompositeKey
    /// </summary>
    /// <param name="other">The CompositeKey to compare with the current CompositeKey</param>
    /// <returns>true if the specified CompositeKey is equal to the current CompositeKey; otherwise, false</returns>
    public bool Equals(CompositeKey? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (values.Length != other.values.Length) return false;

        for (int i = 0; i < values.Length; i++)
        {
            // In join operations, null values should not match with anything, including other nulls
            if (values[i] == null || other.values[i] == null)
                return false;

            if (!Equals(values[i], other.values[i]))
                return false;
        }

        return true;
    }

    /// <summary>
    /// Determines whether the specified object is equal to the current CompositeKey
    /// </summary>
    /// <param name="obj">The object to compare with the current CompositeKey</param>
    /// <returns>true if the specified object is equal to the current CompositeKey; otherwise, false</returns>
    public override bool Equals(object? obj)
    {
        return Equals(obj as CompositeKey);
    }

    /// <summary>
    /// Returns the hash code for this CompositeKey
    /// </summary>
    /// <returns>A hash code for the current CompositeKey</returns>
    public override int GetHashCode()
    {
        return hashCode;
    }

    /// <summary>
    /// Returns a string representation of the CompositeKey
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        var valueStrings = values.Select(v => v?.ToString() ?? "null");
        return $"({string.Join(", ", valueStrings)})";
    }
}