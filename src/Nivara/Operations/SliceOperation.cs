using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

/// <summary>
/// Represents a slice operation that takes a subset of rows from a DataFrame
/// </summary>
sealed class SliceOperation : IQueryOperation
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

    public string OperationType => Query.OperationType.Slice;

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
        return ColumnFilterHelper.CreateFilteredColumn(column, indices);
    }

    static IColumn CreateEmptyColumn(Type elementType)
        => ColumnFilterHelper.CreateEmptyColumn(elementType);

    public override string ToString()
    {
        return Take.HasValue ? $"Slice(Skip: {Skip}, Take: {Take})" : $"Slice(Skip: {Skip})";
    }
}
