using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

sealed class SelectRowsOperation : IQueryOperation
{
    public SelectRowsOperation(int[] indices)
    {
        ArgumentNullException.ThrowIfNull(indices);
        if (indices.Length == 0)
            throw new ArgumentException("Indices must not be empty", nameof(indices));

        Indices = new int[indices.Length];
        Array.Copy(indices, Indices, indices.Length);
    }

    public int[] Indices { get; }

    public string OperationType => Query.OperationType.SelectRows;

    public Schema TransformSchema(Schema inputSchema)
    {
        ArgumentNullException.ThrowIfNull(inputSchema);
        return inputSchema;
    }

    public IReadOnlyDictionary<string, IColumn> Execute(IReadOnlyDictionary<string, IColumn> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input.Count == 0)
            return input;

        try
        {
            var rowCount = input.Values.First().Length;
            var validIndices = new List<int>(Indices.Length);
            foreach (var idx in Indices)
            {
                if (idx < 0 || idx >= rowCount)
                    throw new ArgumentOutOfRangeException("indices",
                        $"Index {idx} is out of range. Column has {rowCount} rows.");
                validIndices.Add(idx);
            }

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
                result[kvp.Key] = ColumnFilterHelper.CreateFilteredColumn(kvp.Value, validIndices);

            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"SelectRows operation failed: {ex.Message}", ex);
        }
    }

    public override string ToString()
    {
        var count = Indices.Length;
        var preview = count <= 5
            ? string.Join(", ", Indices)
            : string.Join(", ", Indices.Take(5)) + $", ... ({count} indices)";
        return $"SelectRows({preview})";
    }
}
