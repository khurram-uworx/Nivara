using Nivara.Operations;

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
/// <see cref="GroupByOperation"/>), then stable-sorts each partition by the order keys
/// using <see cref="MultiColumnComparer"/>. Rows with any null order key produce a null
/// output and are excluded from numbering and from the percent_rank denominator.
/// </para>
/// </summary>
/// <remarks>Added as part of issue #156 rank family window functions delivery.</remarks>
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

        var partitions = partitionBy.Length == 0
            ? new[] { Enumerable.Range(0, rowCount).ToArray() }
            : GroupByOperation.CreateGroupsInternal(columns, partitionBy)
                .GetAllGroups()
                .Select(g => g.Indices.ToArray())
                .ToArray();

        var rankResult = new long[rowCount];
        var percentResult = new double[rowCount];
        var mask = new bool[rowCount];
        var comparer = orderBy.Count > 0 ? new MultiColumnComparer(columns, orderBy) : null;

        foreach (var partition in partitions)
        {
            var valid = new List<int>(partition.Length);
            for (int i = 0; i < partition.Length; i++)
            {
                var row = partition[i];
                if (comparer != null && hasNullKey(columns, orderBy, row))
                    mask[row] = true;
                else
                    valid.Add(row);
            }

            if (valid.Count == 0)
                continue;

            var sorted = comparer == null
                ? valid.ToArray()
                : valid.OrderBy(i => i, comparer).ToArray();

            int gapRank = 0;
            int denseCount = 0;

            for (int pos = 0; pos < sorted.Length; pos++)
            {
                bool newGroup = comparer == null || pos == 0 || comparer.Compare(sorted[pos], sorted[pos - 1]) != 0;
                if (newGroup)
                {
                    gapRank = pos + 1;
                    denseCount++;
                }

                var row = sorted[pos];
                switch (kind)
                {
                    case RankKind.RowNumber:
                        rankResult[row] = pos + 1;
                        break;
                    case RankKind.Rank:
                        rankResult[row] = gapRank;
                        break;
                    case RankKind.DenseRank:
                        rankResult[row] = denseCount;
                        break;
                    case RankKind.PercentRank:
                        percentResult[row] = sorted.Length == 1 ? 0.0 : (double)(gapRank - 1) / (sorted.Length - 1);
                        break;
                }
            }
        }

        return kind == RankKind.PercentRank
            ? NivaraColumn<double>.CreateFromSpans(percentResult, mask)
            : NivaraColumn<long>.CreateFromSpans(rankResult, mask);
    }

    static bool hasNullKey(IReadOnlyDictionary<string, IColumn> columns, IReadOnlyList<SortKey> orderBy, int row)
    {
        for (int k = 0; k < orderBy.Count; k++)
            if (columns[orderBy[k].ColumnName].IsNull(row))
                return true;

        return false;
    }
}
