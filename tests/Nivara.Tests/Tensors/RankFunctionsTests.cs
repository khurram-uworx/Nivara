using Nivara.Operations;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Tests for the rank-family window kernel (row_number / rank / dense_rank / percent_rank).
/// </summary>
/// <remarks>Added as part of issue #156 rank family window functions delivery.</remarks>
[TestFixture]
public class RankFunctionsTests
{
    static IReadOnlyDictionary<string, IColumn> Columns(params (string Name, IColumn Column)[] columns)
        => columns.ToDictionary(c => c.Name, c => c.Column, StringComparer.OrdinalIgnoreCase);

    static NivaraColumn<T> Rank<T>(IColumn result, RankKind kind)
        where T : struct
    {
        var expected = kind == RankKind.PercentRank ? typeof(double) : typeof(long);
        Assert.That(result.ElementType, Is.EqualTo(expected), $"{kind} result type mismatch");
        Assert.That(result, Is.InstanceOf<NivaraColumn<T>>(), $"{kind} result column type mismatch");
        return (NivaraColumn<T>)result;
    }

    // ── RowNumber ──

    [Test]
    public void RowNumber_NoPartition_Sequential()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 10, 20, 30 })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [], RankKind.RowNumber), RankKind.RowNumber);

        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(2));
        Assert.That(result[2], Is.EqualTo(3));
    }

    [Test]
    public void RowNumber_WithPartition_ResetsPerGroup()
    {
        var columns = Columns(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "A" })),
            ("v", NivaraColumn<int>.Create(new[] { 10, 20, 30, 20 })));

        var result = Rank<long>(RankKernel.Compute(columns, ["g"], [], RankKind.RowNumber), RankKind.RowNumber);

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(2));
        Assert.That(result[2], Is.EqualTo(1));
        Assert.That(result[3], Is.EqualTo(3));
    }

    // ── Rank / DenseRank / PercentRank ties ──

    [Test]
    public void Rank_WithTies_LeavesGaps()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 10, 20, 20, 30 })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.Rank), RankKind.Rank);

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(2));
        Assert.That(result[2], Is.EqualTo(2));
        Assert.That(result[3], Is.EqualTo(4));
    }

    [Test]
    public void DenseRank_WithTies_NoGaps()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 10, 20, 20, 30, 20 })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.DenseRank), RankKind.DenseRank);

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(2));
        Assert.That(result[2], Is.EqualTo(2));
        Assert.That(result[3], Is.EqualTo(3));
        Assert.That(result[4], Is.EqualTo(2));
    }

    [Test]
    public void PercentRank_WithTies_RelativePosition()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 10, 20, 20, 30 })));

        var result = Rank<double>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.PercentRank), RankKind.PercentRank);

        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(0.0));
        Assert.That(result[1], Is.EqualTo(1.0 / 3.0).Within(1e-9));
        Assert.That(result[2], Is.EqualTo(1.0 / 3.0).Within(1e-9));
        Assert.That(result[3], Is.EqualTo(1.0));
    }

    [Test]
    public void PercentRank_SingleRowPartition_Zero()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 5 })));

        var result = Rank<double>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.PercentRank), RankKind.PercentRank);

        Assert.That(result[0], Is.EqualTo(0.0));
    }

    // ── Direction ──

    [Test]
    public void Rank_Descending_OrdersHighToLow()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(new[] { 30, 20, 20, 10 })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("v", SortDirection.Descending)], RankKind.Rank), RankKind.Rank);

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(2));
        Assert.That(result[2], Is.EqualTo(2));
        Assert.That(result[3], Is.EqualTo(4));
    }

    [Test]
    public void Rank_MultipleOrderKeys_TieBrokenBySecondKey()
    {
        var columns = Columns(
            ("a", NivaraColumn<int>.Create(new[] { 1, 1, 1, 2 })),
            ("b", NivaraColumn<int>.Create(new[] { 3, 1, 3, 1 })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("a"), new SortKey("b")], RankKind.Rank), RankKind.Rank);

        Assert.That(result[0], Is.EqualTo(2));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(2));
        Assert.That(result[3], Is.EqualTo(4));
    }

    // ── Null order keys ──

    [Test]
    public void RowNumber_NullOrderKey_NumberedLast()
    {
        var columns = Columns(("v", NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.RowNumber), RankKind.RowNumber);

        Assert.That(result.HasNulls, Is.False, "row_number numbers every row including null-key rows (issue #254)");
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(3));
        Assert.That(result[2], Is.EqualTo(2));
    }

    [Test]
    public void RowNumber_NullOrderKey_NullsFirstOrdering_NumberedFirst()
    {
        var columns = Columns(("v", NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false })));

        var result = Rank<long>(RankKernel.Compute(
            columns, [], [new SortKey("v", SortDirection.Ascending, NullOrdering.NullsFirst)], RankKind.RowNumber), RankKind.RowNumber);

        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(2));
        Assert.That(result[1], Is.EqualTo(1));
        Assert.That(result[2], Is.EqualTo(3));
    }

    [Test]
    public void RowNumber_NullOrderKey_MultipleKeys_OrderedPerSecondKey()
    {
        var columns = Columns(
            ("a", NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 0, 2 }, new[] { false, true, true, false })),
            ("b", NivaraColumn<int>.Create(new[] { 9, 5, 2, 1 })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("a"), new SortKey("b")], RankKind.RowNumber), RankKind.RowNumber);

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result[1], Is.EqualTo(4));
        Assert.That(result[2], Is.EqualTo(3));
        Assert.That(result[3], Is.EqualTo(2));
    }

    [Test]
    public void Rank_NullOrderKey_NullOutput_ExcludedFromNumbering()
    {
        var columns = Columns(("v", NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.Rank), RankKind.Rank);

        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(2));
    }

    [Test]
    public void PercentRank_NullOrderKey_ExcludedFromDenominator()
    {
        var columns = Columns(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "A" })),
            ("v", NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 2 }, new[] { false, true, false })));

        var result = Rank<double>(RankKernel.Compute(columns, ["g"], [new SortKey("v")], RankKind.PercentRank), RankKind.PercentRank);

        Assert.That(result[0], Is.EqualTo(0.0));
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(1.0));
    }

    [Test]
    public void Rank_PartitionWithAllNullKeys_AllNull()
    {
        var columns = Columns(("v", NivaraColumn<int>.CreateFromSpans(new[] { 0, 0 }, new[] { true, true })));

        var result = Rank<long>(RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.Rank), RankKind.Rank);

        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.IsNull(0), Is.True);
        Assert.That(result.IsNull(1), Is.True);
    }

    // ── Empty ──

    [Test]
    public void Rank_EmptyColumns_ReturnsEmpty()
    {
        var columns = Columns(("v", NivaraColumn<int>.Create(Array.Empty<int>())));

        var result = RankKernel.Compute(columns, [], [new SortKey("v")], RankKind.Rank);

        Assert.That(result.Length, Is.EqualTo(0));
        Assert.That(result.ElementType, Is.EqualTo(typeof(long)));
    }

    // ── Property-style: randomized comparison against naive references ──

    [Test]
    public void RankFamily_RandomPartitionsAndTies_MatchesNaive()
    {
        var random = new Random(156);
        int length = 400;
        int[] values = new int[length];
        int[] groups = new int[length];
        bool[] mask = new bool[length];
        for (int i = 0; i < length; i++)
        {
            values[i] = random.Next(0, 8);
            groups[i] = random.Next(0, 3);
            mask[i] = random.Next(6) == 0;
        }

        var columns = Columns(
            ("g", NivaraColumn<int>.Create(groups)),
            ("v", NivaraColumn<int>.CreateFromSpans(values, mask)));

        var rowNumber = Rank<long>(RankKernel.Compute(columns, ["g"], [new SortKey("v")], RankKind.RowNumber), RankKind.RowNumber);
        var rank = Rank<long>(RankKernel.Compute(columns, ["g"], [new SortKey("v")], RankKind.Rank), RankKind.Rank);
        var denseRank = Rank<long>(RankKernel.Compute(columns, ["g"], [new SortKey("v")], RankKind.DenseRank), RankKind.DenseRank);
        var percentRank = Rank<double>(RankKernel.Compute(columns, ["g"], [new SortKey("v")], RankKind.PercentRank), RankKind.PercentRank);

        foreach (var group in new[] { 0, 1, 2 })
        {
            var validRows = new List<int>();
            var nullRows = new List<int>();
            for (int i = 0; i < length; i++)
            {
                if (groups[i] != group)
                    continue;
                if (mask[i])
                    nullRows.Add(i);
                else
                    validRows.Add(i);
            }

            var distinctValues = validRows.Select(i => values[i]).Distinct().OrderBy(x => x).ToList();
            var denseBase = new Dictionary<int, long>();
            for (int d = 0; d < distinctValues.Count; d++)
                denseBase[distinctValues[d]] = d + 1;

            foreach (var row in validRows)
            {
                int lessCount = validRows.Count(j => values[j] < values[row]);
                int equalBefore = validRows.Count(j => j < row && values[j] == values[row]);

                Assert.That(rowNumber[row], Is.EqualTo((long)lessCount + equalBefore + 1), $"rowNumber mismatch at {row}");
                Assert.That(rank[row], Is.EqualTo((long)lessCount + 1), $"rank mismatch at {row}");
                Assert.That(denseRank[row], Is.EqualTo(denseBase[values[row]]), $"denseRank mismatch at {row}");
                var expectedPercent = validRows.Count == 1 ? 0.0 : (double)lessCount / (validRows.Count - 1);
                Assert.That(percentRank[row], Is.EqualTo(expectedPercent).Within(1e-9), $"percentRank mismatch at {row}");
            }

            for (int n = 0; n < nullRows.Count; n++)
            {
                var row = nullRows[n];
                Assert.That(rowNumber[row], Is.EqualTo((long)validRows.Count + n + 1), $"rowNumber null-key row mismatch at {row}");
                Assert.That(rowNumber.IsNull(row), Is.False, $"rowNumber must not be null at {row}");
                Assert.That(rank.IsNull(row), Is.True, $"rank null mismatch at {row}");
                Assert.That(denseRank.IsNull(row), Is.True, $"denseRank null mismatch at {row}");
                Assert.That(percentRank.IsNull(row), Is.True, $"percentRank null mismatch at {row}");
            }
        }
    }
}
