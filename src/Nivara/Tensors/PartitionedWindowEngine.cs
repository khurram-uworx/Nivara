using System.Buffers;
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
/// <see cref="MultiColumnComparer"/> with a row-index tiebreak, computes the window per
/// partition via the supplied delegate over contiguous sorted slices of a single gathered
/// source column, then scatters the per-partition results back to the original row order in
/// one pass (no concatenation or inverse-permutation step). Null order-key rows are ordered
/// per the sort keys' <see cref="NullOrdering"/> and participate in the window (SQL-faithful).
/// An empty spec short-circuits to the raw delegate so behavior is identical to the existing
/// unpartitioned paths.
/// </para>
/// </summary>
/// <remarks>Added as part of issue #162 Over/WindowSpec builder delivery; scatter refactor part of
/// issue #251 allocation reduction.</remarks>
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

        var singlePartition = spec.PartitionColumns.Count == 0;
        var partitions = singlePartition
            ? Array.Empty<int[]>()
            : GroupByOperation.CreateGroupsInternal(columns, spec.PartitionColumns.ToArray())
                .GetAllGroups()
                .Select(g => g.Indices.ToArray())
                .ToArray();

        var comparer = new MultiColumnComparer(columns, spec.OrderKeys);
        var tieBreakComparer = new RankTieBreakComparer(comparer);

        // positions[pos] = original row holding the sorted position `pos`.
        var positions = new int[rowCount];

        int[]? scratch = null;
        bool scratchPooled = false;
        if (singlePartition)
        {
            if (rowCount > 1024)
            {
                scratch = ArrayPool<int>.Shared.Rent(rowCount);
                scratchPooled = true;
            }
            else
            {
                scratch = new int[rowCount];
            }

            for (int i = 0; i < rowCount; i++)
                scratch[i] = i;
            if (spec.OrderKeys.Count > 0)
                Array.Sort(scratch, 0, rowCount, tieBreakComparer);
            scratch.CopyTo(positions, 0);
        }
        else
        {
            int cursor = 0;
            foreach (var partition in partitions)
            {
                if (spec.OrderKeys.Count > 0)
                    Array.Sort(partition, 0, partition.Length, tieBreakComparer);
                partition.CopyTo(positions, cursor);
                cursor += partition.Length;
            }
        }

        try
        {
            var sortedSource = ColumnFilterHelper.ReorderColumn(sourceColumn, positions);

            var computedParts = new List<IColumn>(singlePartition ? 1 : partitions.Length);
            int offset = 0;
            if (singlePartition)
            {
                computedParts.Add(partitionCompute(sortedSource));
            }
            else
            {
                foreach (var partition in partitions)
                {
                    computedParts.Add(partitionCompute(sortedSource.Slice(offset, partition.Length)));
                    offset += partition.Length;
                }
            }

            return ColumnFilterHelper.ScatterPartsColumn(computedParts, positions);
        }
        finally
        {
            if (scratchPooled)
                ArrayPool<int>.Shared.Return(scratch!);
        }
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
