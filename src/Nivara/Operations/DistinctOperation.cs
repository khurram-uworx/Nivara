using System.Buffers;
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

            var keyColumns = columnNames != null
                ? columnNames.Select(n => input[n]).ToArray()
                : input.Values.ToArray();
            var readers = keyColumns.Select(GroupKeyReaderFactory.Create).ToArray();
            var uniqueIndices = new List<int>();

            var pooled = rowCount > 1024;
            var hashes = pooled ? ArrayPool<int>.Shared.Rent(rowCount) : new int[rowCount];
            try
            {
                TypedGroupHash.ComputeRowHashes(readers, rowCount, hashes);
                var hashBuckets = new Dictionary<int, List<int>>();

                for (int i = 0; i < rowCount; i++)
                {
                    int hash = hashes[i];

                    if (!hashBuckets.TryGetValue(hash, out var reps))
                    {
                        hashBuckets[hash] = new List<int>(1) { i };
                        uniqueIndices.Add(i);
                        continue;
                    }

                    bool duplicate = false;
                    foreach (var rep in reps)
                    {
                        if (!TypedGroupHash.RowsEqual(readers, rep, i))
                            continue;

                        duplicate = true;
                        break;
                    }

                    if (!duplicate)
                    {
                        reps.Add(i);
                        uniqueIndices.Add(i);
                    }
                }
            }
            finally
            {
                if (pooled)
                    ArrayPool<int>.Shared.Return(hashes);
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
