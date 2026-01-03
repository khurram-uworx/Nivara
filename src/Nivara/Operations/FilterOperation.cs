using Nivara.Exceptions;
using Nivara.Expressions;

namespace Nivara;

/// <summary>
/// Represents a filter operation that applies a condition to filter rows
/// </summary>
internal sealed class FilterOperation : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of FilterOperation
    /// </summary>
    /// <param name="condition">The condition to filter by</param>
    /// <exception cref="ArgumentNullException">Thrown when condition is null</exception>
    public FilterOperation(ColumnExpression condition)
    {
        Condition = condition ?? throw new ArgumentNullException(nameof(condition));
    }

    /// <summary>
    /// Gets the filter condition
    /// </summary>
    public ColumnExpression Condition { get; }

    /// <inheritdoc />
    public string OperationType => "Filter";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Validate the condition against the schema
        try
        {
            Condition.Validate(inputSchema);
        }
        catch (SchemaValidationException ex)
        {
            throw new SchemaValidationException($"Filter condition validation failed: {ex.Message}");
        }

        // Filter doesn't change the schema structure, only the number of rows
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
            // Evaluate the condition to get a boolean mask
            var evaluator = new ExpressionEvaluator();
            var mask = evaluator.EvaluateBoolean(Condition, input);

            // Apply the mask to all columns
            var filteredColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);

            foreach (var kvp in input)
            {
                var filteredColumn = ApplyMask(kvp.Value, mask);
                filteredColumns[kvp.Key] = filteredColumn;
            }

            return filteredColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Filter operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Applies a boolean mask to a column to filter its values
    /// </summary>
    /// <param name="column">The column to filter</param>
    /// <param name="mask">The boolean mask indicating which rows to keep</param>
    /// <returns>A new column with filtered values</returns>
    static IColumn ApplyMask(IColumn column, NivaraColumn<bool> mask)
    {
        if (column.Length != mask.Length)
            throw new ArgumentException("Column and mask must have the same length");

        var filteredIndices = new List<int>();

        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i] == true) // Only include rows where mask is true
            {
                filteredIndices.Add(i);
            }
        }

        // Create a new column with only the filtered values
        return CreateFilteredColumn(column, filteredIndices);
    }

    /// <summary>
    /// Creates a new column containing only the values at the specified indices
    /// </summary>
    /// <param name="column">The source column</param>
    /// <param name="indices">The indices of values to include</param>
    /// <returns>A new column with filtered values</returns>
    static IColumn CreateFilteredColumn(IColumn column, List<int> indices)
    {
        var elementType = column.ElementType;

        // Use dynamic dispatch to create the appropriate column type
        return elementType switch
        {
            Type t when t == typeof(int) => CreateFilteredColumnTyped<int>(column, indices),
            Type t when t == typeof(double) => CreateFilteredColumnTyped<double>(column, indices),
            Type t when t == typeof(float) => CreateFilteredColumnTyped<float>(column, indices),
            Type t when t == typeof(long) => CreateFilteredColumnTyped<long>(column, indices),
            Type t when t == typeof(string) => CreateFilteredColumnTyped<string>(column, indices),
            Type t when t == typeof(bool) => CreateFilteredColumnTyped<bool>(column, indices),
            Type t when t == typeof(decimal) => CreateFilteredColumnTyped<decimal>(column, indices),
            Type t when t == typeof(byte) => CreateFilteredColumnTyped<byte>(column, indices),
            Type t when t == typeof(short) => CreateFilteredColumnTyped<short>(column, indices),
            Type t when t == typeof(DateTime) => CreateFilteredColumnTyped<DateTime>(column, indices),
            _ => CreateFilteredColumnGeneric(column, indices)
        };
    }

    /// <summary>
    /// Creates a filtered column for a specific type
    /// </summary>
    static IColumn CreateFilteredColumnTyped<T>(IColumn column, List<int> indices)
    {
        // Check if T is a value type to determine which creation method to use
        if (typeof(T).IsValueType)
        {
            // For value types, create nullable array and use CreateFromNullable
            var nullableType = typeof(Nullable<>).MakeGenericType(typeof(T));
            var filteredArray = System.Array.CreateInstance(nullableType, indices.Count);

            for (int i = 0; i < indices.Count; i++)
            {
                var value = column.GetValue(indices[i]);
                if (value != null)
                {
                    var nullableInstance = Activator.CreateInstance(nullableType, value);
                    filteredArray.SetValue(nullableInstance, i);
                }
                // null values remain null in the array
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<int>.CreateFromNullable), new[] { nullableType.MakeArrayType() })!
                .Invoke(null, new object[] { filteredArray })!;
        }
        else
        {
            // For reference types, create regular array and use CreateForReferenceType
            var filteredArray = new T[indices.Count];

            for (int i = 0; i < indices.Count; i++)
            {
                var value = column.GetValue(indices[i]);
                filteredArray[i] = (T)value!; // Reference types can be null
            }

            return (IColumn)typeof(NivaraColumn<>)
                .MakeGenericType(typeof(T))
                .GetMethod(nameof(NivaraColumn<string>.CreateForReferenceType), new[] { typeof(T[]) })!
                .Invoke(null, new object[] { filteredArray })!;
        }
    }

    /// <summary>
    /// Creates a filtered column for unknown types using object column
    /// </summary>
    static IColumn CreateFilteredColumnGeneric(IColumn column, List<int> indices)
    {
        var filteredArray = new object[indices.Count];

        for (int i = 0; i < indices.Count; i++)
        {
            filteredArray[i] = column.GetValue(indices[i])!;
        }

        return NivaraColumn<object>.Create(filteredArray);
    }

    /// <summary>
    /// Returns a string representation of the filter operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        return $"Filter({Condition})";
    }
}
