using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.Operations;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for window operations over computed source/key expressions in the lazy query
/// pipeline (#159). Expression-based results must match the name-based operations over an
/// equivalent pre-computed column.
/// </summary>
[TestFixture]
public class WindowExpressionOperationTests
{
    static NivaraColumn<int> IntColumn(params int[] values)
        => NivaraColumn<int>.Create(values);

    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns)
        => new NivaraFrame(columns);

    [Test]
    public void RollingSum_OverComputedSource_MatchesNameBased()
    {
        using var frame = FrameWith(
            ("A", IntColumn(1, 2, 3, 4)),
            ("A2", IntColumn(2, 4, 6, 8)));
        using var nameBased = frame.AsQueryFrame().RollingSum("A2", "r_name", 2).Collect();
        using var expressionBased = frame.AsQueryFrame().RollingSum(ColumnExpressions.Col("A") * 2, "r_expr", 2).Collect();

        var expected = nameBased.GetColumn<int>("r_name");
        var actual = expressionBased.GetColumn<int>("r_expr");
        Assert.That(expressionBased.HasColumn("A"), Is.True, "input columns must be preserved");
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
            Assert.That(actual.IsNull(i), Is.EqualTo(expected.IsNull(i)), $"null mask at {i}");
        Assert.That(actual, Is.EqualTo(expected), "expression rolling sum must equal name-based over the computed column");
    }

    [Test]
    public void RollingSum_OverComputedSource_ValuesAreCorrect()
    {
        using var frame = FrameWith(("A", IntColumn(1, 2, 3, 4)));
        using var result = frame.AsQueryFrame().RollingSum(ColumnExpressions.Col("A") * 2, "r", 2).Collect();

        var r = result.GetColumn<int>("r");
        Assert.That(r.IsNull(0), Is.True);
        Assert.That(r[1], Is.EqualTo(6));
        Assert.That(r[2], Is.EqualTo(10));
        Assert.That(r[3], Is.EqualTo(14));
    }

    [Test]
    public void RollingMean_OverComputedSource_PromotesToDouble()
    {
        using var frame = FrameWith(("A", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().RollingMean(ColumnExpressions.Col("A") * 2, "m", 3).Collect();

        var m = result.GetColumn<double>("m");
        Assert.That(m[2], Is.EqualTo(4.0).Within(1e-9));
    }

    [Test]
    public void CumulativeSum_OverComputedSource_MatchesNameBased()
    {
        using var frame = FrameWith(
            ("A", IntColumn(1, 2, 3, 4)),
            ("A2", IntColumn(2, 4, 6, 8)));
        using var nameBased = frame.AsQueryFrame().CumulativeSum("A2", "c_name").Collect();
        using var expressionBased = frame.AsQueryFrame().CumulativeSum(ColumnExpressions.Col("A") * 2, "c_expr").Collect();

        Assert.That(
            expressionBased.GetColumn<int>("c_expr"),
            Is.EqualTo(nameBased.GetColumn<int>("c_name")));
    }

    [Test]
    public void CumulativeCount_OverComputedSource_IsLongColumn()
    {
        using var frame = FrameWith(("A", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame().CumulativeCount(ColumnExpressions.Col("A") * 2, "cnt").Collect();

        var cnt = result.GetColumn<long>("cnt");
        Assert.That(cnt[0], Is.EqualTo(1));
        Assert.That(cnt[1], Is.EqualTo(2));
        Assert.That(cnt[2], Is.EqualTo(3));
    }

    [Test]
    public void Shift_OverComputedSource_AppliesFillAtBoundary()
    {
        using var frame = FrameWith(("A", IntColumn(1, 2, 3, 4)));
        using var result = frame.AsQueryFrame().Shift(ColumnExpressions.Col("A") * 2, "s", 1, -1).Collect();

        var s = result.GetColumn<int>("s");
        Assert.That(s[0], Is.EqualTo(-1));
        Assert.That(s[1], Is.EqualTo(2));
        Assert.That(s[2], Is.EqualTo(4));
        Assert.That(s[3], Is.EqualTo(6));
    }

    [Test]
    public void Lead_OverComputedSource_MovesForwardWithNulls()
    {
        using var frame = FrameWith(("A", IntColumn(1, 2, 3, 4)));
        using var result = frame.AsQueryFrame().Lead(ColumnExpressions.Col("A") * 2, "l", 1).Collect();

        var l = result.GetColumn<int>("l");
        Assert.That(l[0], Is.EqualTo(4));
        Assert.That(l[1], Is.EqualTo(6));
        Assert.That(l[2], Is.EqualTo(8));
        Assert.That(l.IsNull(3), Is.True);
    }

    [Test]
    public void Rank_OverExpressionKeys_MatchesNameBased()
    {
        using var frame = FrameWith(
            ("Dept", NivaraColumn<string>.CreateForReferenceType(new[] { "X", "Y", "X", "Z" })),
            ("Score", IntColumn(1, 2, 3, 4)));
        using var nameBased = frame.AsQueryFrame().Rank("rn", new[] { new SortKey("Score") }, "Dept").Collect();
        using var expressionBased = frame.AsQueryFrame().Rank(
            "re",
            new[] { new SortExpressionKey(ColumnExpressions.Col("Score")) },
            ColumnExpressions.Col("Dept")).Collect();

        Assert.That(expressionBased.GetColumn<long>("re"), Is.EqualTo(nameBased.GetColumn<long>("rn")));
    }

    [Test]
    public void Rank_OverComputedOrderExpression_ValuesAreCorrect()
    {
        using var frame = FrameWith(
            ("Dept", NivaraColumn<string>.CreateForReferenceType(new[] { "X", "Y", "X", "Z" })),
            ("Score", IntColumn(1, 2, 3, 4)));
        using var result = frame.AsQueryFrame().Rank(
            "r",
            new[] { new SortExpressionKey(ColumnExpressions.Col("Score") * 10, SortDirection.Descending) },
            ColumnExpressions.Col("Dept")).Collect();

        var r = result.GetColumn<long>("r");
        Assert.That(r[0], Is.EqualTo(2), "descending order flips the X partition ranking");
        Assert.That(r[1], Is.EqualTo(1));
        Assert.That(r[2], Is.EqualTo(1));
        Assert.That(r[3], Is.EqualTo(1));
    }

    [Test]
    public void MissingExpressionColumn_Throws()
    {
        using var frame = FrameWith(("A", IntColumn(1, 2, 3)));

        var ex = Assert.Throws<QueryExecutionException>(() =>
            frame.AsQueryFrame().RollingSum(ColumnExpressions.Col("Missing"), "r", 2).Collect());
        Assert.That(ex!.Message, Does.Contain("Missing"));
    }
}
