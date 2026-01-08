using Nivara.Exceptions;

namespace Nivara.Operations;

/// <summary>
/// Represents a slice operation that takes a subset of rows from a DataFrame
/// </summary>
internal sealed class SliceOperation : IQueryOperation
{
    /// <summary>
    /// Initializes a new instance of SliceOperation
    /// </summary>
    /// <param name="skip">The number of rows to skip from the beginning</param>
    /// <param name="take">The number of rows to take after skipping (null means take all remaining)</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when skip or take are negative</exception>
    public SliceOperation(int skip, int? take = null)
    {
        if (skip < 0)
            throw new ArgumentOutOfRangeException(nameof(skip), "Skip count cannot be negative");
        if (take.HasValue && take.Value < 0)
            throw new ArgumentOutOfRangeException(nameof(take), "Take count cannot be negative");

        Skip = skip;
        Take = take;
    }

    /// <summary>
    /// Gets the number of rows to skip
    /// </summary>
    public int Skip { get; }

    /// <summary>
    /// Gets the number of rows to take (null means take all remaining)
    /// </summary>
    public int? Take { get; }

    /// <inheritdoc />
    public string OperationType => "Slice";

    /// <inheritdoc />
    public Schema TransformSchema(Schema inputSchema)
    {
        if (inputSchema == null)
            throw new ArgumentNullException(nameof(inputSchema));

        // Slice doesn't change the schema structure, only the number of rows
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

            // Calculate the actual slice parameters
            var actualSkip = Math.Min(Skip, rowCount);
            var remainingRows = Math.Max(0, rowCount - actualSkip);
            var actualTake = Take.HasValue ? Math.Min(Take.Value, remainingRows) : remainingRows;

            // If no rows to return, create empty columns
            if (actualTake == 0)
            {
                var emptyColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
                foreach (var kvp in input)
                {
                    var emptyColumn = CreateEmptyColumn(kvp.Value.ElementType);
                    emptyColumns[kvp.Key] = emptyColumn;
                }
                return emptyColumns;
            }

            // Apply slice to all columns
            var slicedColumns = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
            {
                var slicedColumn = SliceColumn(kvp.Value, actualSkip, actualTake);
                slicedColumns[kvp.Key] = slicedColumn;
            }

            return slicedColumns;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Slice operation failed: {ex.Message}", ex);
        }
    }

    /// <summary>
    /// Slices a column using the column's built-in Slice method
    /// </summary>
    /// <param name="column">The column to slice</param>
    /// <param name="start">The starting index</param>
    /// <param name="length">The number of elements to include</param>
    /// <returns>A sliced column</returns>
    static IColumn SliceColumn(IColumn column, int start, int length)
    {
        // Use reflection to call the Slice method on the typed column
        var columnType = column.GetType();
        var sliceMethod = columnType.GetMethod("Slice", new[] { typeof(int), typeof(int) });
        
        if (sliceMethod != null)
        {
            return (IColumn)sliceMethod.Invoke(column, new object[] { start, length })!;
        }

        // Fallback: create filtered column using indices
        var indices = Enumerable.Range(start, length).ToList();
        return CreateFilteredColumn(column, indices);
    }

    /// <summary>
    /// Creates an empty column of the specified type
    /// </summary>
    /// <param name="elementType">The type of elements in the column</param>
    /// <returns>An empty column of the specified type</returns>
    static IColumn CreateEmptyColumn(Type elementType)
    {
        return elementType switch
        {
            Type t when t == typeof(int) => NivaraColumn<int>.Create(Array.Empty<int>()),
            Type t when t == typeof(double) => NivaraColumn<double>.Create(Array.Empty<double>()),
            Type t when t == typeof(float) => NivaraColumn<float>.Create(Array.Empty<float>()),
            Type t when t == typeof(long) => NivaraColumn<long>.Create(Array.Empty<long>()),
            Type t when t == typeof(string) => NivaraColumn<string>.CreateForReferenceType(Array.Empty<string>()),
            Type t when t == typeof(bool) => NivaraColumn<bool>.Create(Array.Empty<bool>()),
            Type t when t == typeof(decimal) => NivaraColumn<decimal>.Create(Array.Empty<decimal>()),
            Type t when t == typeof(byte) => NivaraColumn<byte>.Create(Array.Empty<byte>()),
            Type t when t == typeof(short) => NivaraColumn<short>.Create(Array.Empty<short>()),
            Type t when t == typeof(DateTime) => NivaraColumn<DateTime>.Create(Array.Empty<DateTime>()),
            _ => NivaraColumn<object>.Create(Array.Empty<object>())
        };
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
    /// Returns a string representation of the slice operation
    /// </summary>
    /// <returns>A string representation</returns>
    public override string ToString()
    {
        return Take.HasValue ? $"Slice(Skip: {Skip}, Take: {Take})" : $"Slice(Skip: {Skip})";
    }
}