using Nivara.Helpers;
using Nivara.Operations;

namespace Nivara.Tensors;

/// <summary>
/// Shared partitioned-window engine used by the eager <see cref="Nivara.NivaraFrameExtensions"/>
/// window methods and the lazy query-pipeline window operations when a
/// <see cref="WindowSpec"/> with partition and/or order keys is supplied.
/// <para>
/// The engine partitions rows by the spec's partition keys (reusing the hash-based grouping
/// from <see cref="GroupByOperation"/>), stable-sorts each partition by the order keys using
/// <see cref="MultiColumnComparer"/>, computes the window per partition via the supplied
/// delegate, concatenates the per-partition results, then scatters them back to the original
/// row order. Null order-key rows are ordered per the sort keys' <see cref="NullOrdering"/>
/// and participate in the window (SQL-faithful). An empty spec short-circuits to the raw
/// delegate so behavior is identical to the existing unpartitioned paths.
/// </para>
/// </summary>
/// <remarks>Added as part of issue #162 Over/WindowSpec builder delivery.</remarks>
internal static class PartitionedWindowEngine
{
    /// <summary>
    /// Computes a partitioned window over <paramref name="sourceColumn"/> and returns a result
    /// column aligned with the original row order.
    /// </summary>
    /// <param name="columns">All columns of the frame / input dict (partition and order keys are resolved here)</param>
    /// <param name="sourceColumn">The source column the window is computed over</param>
    /// <param name="spec">The window specification (partition keys + order keys)</param>
    /// <param name="partitionCompute">Delegate computing the window over a contiguous sorted partition</param>
    /// <returns>A result column in the original row order</returns>
    /// <exception cref="ArgumentException">Thrown when a partition/order column is missing or an order column is not comparable</exception>
    public static IColumn Compute(
        IReadOnlyDictionary<string, IColumn> columns,
        IColumn sourceColumn,
        WindowSpec spec,
        Func<IColumn, IColumn> partitionCompute)
    {
        ArgumentNullException.ThrowIfNull(columns);
        ArgumentNullException.ThrowIfNull(sourceColumn);
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(partitionCompute);

        if (spec.IsEmpty)
            return partitionCompute(sourceColumn);

        ValidateColumns(columns, spec);

        var rowCount = sourceColumn.Length;
        if (rowCount == 0)
            return partitionCompute(sourceColumn);

        var partitions = spec.PartitionColumns.Count == 0
            ? new[] { Enumerable.Range(0, rowCount).ToArray() }
            : GroupByOperation.CreateGroupsInternal(columns, spec.PartitionColumns.ToArray())
                .GetAllGroups()
                .Select(g => g.Indices.ToArray())
                .ToArray();

        var comparer = new MultiColumnComparer(columns, spec.OrderKeys);

        var sortedAll = new int[rowCount];
        int cursor = 0;
        foreach (var partition in partitions)
        {
            var sorted = partition.OrderBy(i => i, comparer).ToArray();
            sorted.CopyTo(sortedAll, cursor);
            cursor += sorted.Length;
        }

        var sortedSource = ColumnFilterHelper.ReorderColumn(sourceColumn, sortedAll);

        var computedParts = new List<IColumn>(partitions.Length);
        cursor = 0;
        foreach (var partition in partitions)
        {
            computedParts.Add(partitionCompute(sortedSource.Slice(cursor, partition.Length)));
            cursor += partition.Length;
        }

        var sortedResult = ColumnFilterHelper.ConcatenateColumns(computedParts);

        var inverse = new int[rowCount];
        for (int i = 0; i < sortedAll.Length; i++)
            inverse[sortedAll[i]] = i;

        return ColumnFilterHelper.ReorderColumn(sortedResult, inverse);
    }

    /// <summary>
    /// Validates that all partition columns exist and all order columns exist and are comparable.
    /// </summary>
    /// <param name="columns">The columns to validate against</param>
    /// <param name="spec">The window specification</param>
    /// <exception cref="ArgumentException">Thrown when a partition/order column is missing or an order column is not comparable</exception>
    public static void ValidateColumns(IReadOnlyDictionary<string, IColumn> columns, WindowSpec spec)
    {
        foreach (var name in spec.PartitionColumns)
            if (!columns.ContainsKey(name))
                throw new ArgumentException($"Partition column '{name}' not found", nameof(spec));

        foreach (var key in spec.OrderKeys)
        {
            if (!columns.ContainsKey(key.ColumnName))
                throw new ArgumentException($"Order column '{key.ColumnName}' not found", nameof(spec));

            if (!SortOperation.IsComparableType(columns[key.ColumnName].ElementType))
                throw new ArgumentException(
                    $"Order column '{key.ColumnName}' of type {columns[key.ColumnName].ElementType.Name} is not comparable",
                    nameof(spec));
        }
    }
}
