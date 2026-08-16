using Nivara.Operations;
using System.Buffers;

namespace Nivara.Tensors;

/// <summary>
/// Identifies a rank-family window function.
/// </summary>
public enum RankKind
{
    /// <summary>
    /// Sequential number per partition: 1, 2, 3, ... with no ties.
    /// </summary>
    RowNumber,

    /// <summary>
    /// Standard rank with gaps on ties: 1, 1, 3, ...
    /// </summary>
    Rank,

    /// <summary>
    /// Rank without gaps on ties: 1, 1, 2, ...
    /// </summary>
    DenseRank,

    /// <summary>
    /// Relative rank: (rank - 1) / (partitionSize - 1), 0.0 for a single-row partition.
    /// </summary>
    PercentRank
}

/// <summary>
/// Rank-family window-function kernel: row_number / rank / dense_rank / percent_rank
/// over partition + order-by keys.
/// <para>
/// Partitions rows by the partition keys (reusing the hash-based grouping from
/// <see cref="GroupByOperation"/>), then sorts each partition by the order keys
/// using <see cref="MultiColumnComparer"/> with a row-index tiebreak (<see cref="RankTieBreakComparer"/>)
/// so in-place <see cref="Array.Sort(Array, int, int, IComparer?)"/> reproduces the stable ordering of
/// LINQ's <c>OrderBy</c>. For rank/dense_rank/percent_rank, rows with any null order key produce a
/// null output and are excluded from numbering and from the percent_rank denominator. Row_number
/// instead numbers every partition row in the sorted order (null-key rows placed per the order keys'
/// <see cref="NullOrdering"/>; ties preserve stable partition order), matching SQL semantics (issue #254).
/// </para>
/// </summary>
/// <remarks>Added as part of issue #156 rank family window functions delivery. Scratch arrays are rented
/// from <see cref="ArrayPool{T}"/> for partitions larger than 1024 rows.</remarks>
internal static class RankKernel
{
    /// <summary>
    /// Computes a rank-family column over the given columns.
    /// </summary>
    /// <param name="columns">The source columns</param>
    /// <param name="partitionBy">The partition key column names (empty = single partition)</param>
    /// <param name="orderBy">The order keys (empty = partition order, valid only for RowNumber)</param>
    /// <param name="kind">The rank function kind</param>
    /// <returns>A long column (row_number/rank/dense_rank) or double column (percent_rank)</returns>
    internal static IColumn Compute(
        IReadOnlyDictionary<string, IColumn> columns,
        string[] partitionBy,
        IReadOnlyList<SortKey> orderBy,
        RankKind kind)
    {
        ArgumentNullException.ThrowIfNull(columns);

        var rowCount = columns.Values.FirstOrDefault()?.Length ?? 0;
        if (rowCount == 0)
            return kind == RankKind.PercentRank
                ? NivaraColumn<double>.Create(Array.Empty<double>())
                : NivaraColumn<long>.Create(Array.Empty<long>());

        var rankResult = new long[rowCount];
        var percentResult = new double[rowCount];
        var mask = new bool[rowCount];
        var comparer = orderBy.Count > 0 ? new MultiColumnComparer(columns, orderBy) : null;
        var tieBreakComparer = comparer != null ? new RankTieBreakComparer(comparer) : null;

        var singlePartition = partitionBy.Length == 0;
        var partitions = singlePartition
            ? Array.Empty<int[]>()
            : GroupByOperation.CreateGroupsInternal(columns, partitionBy)
                .GetAllGroups()
                .Select(g => g.Indices.ToArray())
                .ToArray();

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
        }

        try
        {
            if (singlePartition)
                ProcessPartition(scratch!, rowCount);
            else
                foreach (var partition in partitions)
                    ProcessPartition(partition, partition.Length);
        }
        finally
        {
            if (scratchPooled)
                ArrayPool<int>.Shared.Return(scratch!);
        }

        return kind == RankKind.PercentRank
            ? NivaraColumn<double>.CreateFromSpans(percentResult, mask)
            : NivaraColumn<long>.CreateFromSpans(rankResult, mask);

        void ProcessPartition(int[] rows, int count)
        {
            if (kind == RankKind.RowNumber)
            {
                // RowNumber numbers every partition row, null-key rows included, ordered per the
                // order keys' NullOrdering (ties preserve stable partition order) (issue #254).
                if (tieBreakComparer != null)
                    Array.Sort(rows, 0, count, tieBreakComparer);
                for (int pos = 0; pos < count; pos++)
                    rankResult[rows[pos]] = pos + 1;
                return;
            }

            int validCount = 0;
            for (int i = 0; i < count; i++)
            {
                var row = rows[i];
                if (comparer != null && HasNullKey(columns, orderBy, row))
                    mask[row] = true;
                else
                    rows[validCount++] = row;
            }

            if (validCount == 0)
                return;

            if (tieBreakComparer != null)
                Array.Sort(rows, 0, validCount, tieBreakComparer);

            int gapRank = 0;
            int denseCount = 0;

            for (int pos = 0; pos < validCount; pos++)
            {
                bool newGroup = comparer == null || pos == 0 || comparer.Compare(rows[pos], rows[pos - 1]) != 0;
                if (newGroup)
                {
                    gapRank = pos + 1;
                    denseCount++;
                }

                var row = rows[pos];
                switch (kind)
                {
                    case RankKind.Rank:
                        rankResult[row] = gapRank;
                        break;
                    case RankKind.DenseRank:
                        rankResult[row] = denseCount;
                        break;
                    case RankKind.PercentRank:
                        percentResult[row] = validCount == 1 ? 0.0 : (double)(gapRank - 1) / (validCount - 1);
                        break;
                }
            }
        }
    }

    static bool HasNullKey(IReadOnlyDictionary<string, IColumn> columns, IReadOnlyList<SortKey> orderBy, int row)
    {
        for (int k = 0; k < orderBy.Count; k++)
            if (columns[orderBy[k].ColumnName].IsNull(row))
                return true;

        return false;
    }
}

/// <summary>
/// Reproduces the stable ordering of LINQ <c>OrderBy</c> for in-place <see cref="Array.Sort(Array, int, int, IComparer?)"/>
/// by breaking ties on the row index. Rows within a partition are stored in ascending row order, so the
/// row-index tiebreak matches OrderBy's stability (RowNumber tie order and null-key ordering, issue #254).
/// </summary>
internal sealed class RankTieBreakComparer : IComparer<int>
{
    readonly MultiColumnComparer comparer;

    public RankTieBreakComparer(MultiColumnComparer comparer) => this.comparer = comparer;

    public int Compare(int x, int y)
    {
        int result = comparer.Compare(x, y);
        return result != 0 ? result : x.CompareTo(y);
    }
}
