using Nivara.Exceptions;
using Nivara.Helpers;
using Nivara.Query;

namespace Nivara.Operations;

sealed class DistinctOperation : IQueryOperation
{
    readonly string[]? columnNames;

    public DistinctOperation(string[]? columnNames = null)
    {
        this.columnNames = columnNames;
    }

    public string OperationType => Query.OperationType.Distinct;

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
            var firstColumn = input.Values.First();
            var rowCount = firstColumn.Length;

            if (rowCount <= 1)
                return input;

            var seen = new HashSet<GroupKey>();
            var uniqueIndices = new List<int>();

            for (int i = 0; i < rowCount; i++)
            {
                var keyValues = columnNames != null
                    ? columnNames.Select(n => input[n].GetValue(i)).ToArray()
                    : input.Values.Select(c => c.GetValue(i)).ToArray();

                var key = new GroupKey(keyValues);
                if (seen.Add(key))
                    uniqueIndices.Add(i);
            }

            var result = new Dictionary<string, IColumn>(StringComparer.OrdinalIgnoreCase);
            foreach (var kvp in input)
                result[kvp.Key] = ColumnFilterHelper.CreateFilteredColumn(kvp.Value, uniqueIndices);

            return result;
        }
        catch (Exception ex) when (ex is not QueryExecutionException)
        {
            throw new QueryExecutionException($"Distinct operation failed: {ex.Message}", ex);
        }
    }

    public override string ToString()
    {
        return columnNames is { Length: > 0 }
            ? $"Distinct({string.Join(", ", columnNames)})"
            : "Distinct(*)";
    }
}
