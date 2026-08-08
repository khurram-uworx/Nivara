using Nivara.Exceptions;
using Nivara.Operations;
using Nivara.Query;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for the rank-family window operation in the lazy query pipeline.
/// </summary>
/// <remarks>Added as part of issue #156 rank family window functions delivery.</remarks>
[TestFixture]
public class RankOperationTests
{
    static Schema SchemaOf(params (string Name, Type Type)[] columns)
        => new(columns);

    static IReadOnlyDictionary<string, IColumn> Columns(params (string Name, IColumn Column)[] columns)
        => columns.ToDictionary(c => c.Name, c => c.Column, StringComparer.OrdinalIgnoreCase);

    static NivaraColumn<int> IntColumn(params int[] values)
        => NivaraColumn<int>.Create(values);

    // ── Execute ──

    [Test]
    public void Rank_AddsResultColumn_KeepsSourceColumns()
    {
        var input = Columns(("v", IntColumn(10, 20, 20, 30)));
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("v") });

        var result = op.Execute(input);

        Assert.That(result.Count, Is.EqualTo(2));
        Assert.That(result.ContainsKey("v"), Is.True);
        var rnk = (NivaraColumn<long>)result["rnk"];
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk[1], Is.EqualTo(2));
        Assert.That(rnk[2], Is.EqualTo(2));
        Assert.That(rnk[3], Is.EqualTo(4));
    }

    [Test]
    public void PercentRank_AddsDoubleColumn()
    {
        var input = Columns(("v", IntColumn(10, 20, 30)));
        var op = new RankOperation("pct", RankKind.PercentRank, new[] { new SortKey("v") });

        var result = op.Execute(input);

        Assert.That(result["pct"].ElementType, Is.EqualTo(typeof(double)));
        var pct = (NivaraColumn<double>)result["pct"];
        Assert.That(pct[0], Is.EqualTo(0.0));
        Assert.That(pct[1], Is.EqualTo(0.5));
        Assert.That(pct[2], Is.EqualTo(1.0));
    }

    [Test]
    public void RowNumber_WithPartition_ResetsPerGroup()
    {
        var input = Columns(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B" })),
            ("v", IntColumn(10, 20, 30)));
        var op = new RankOperation("rn", RankKind.RowNumber, Array.Empty<SortKey>(), new[] { "g" });

        var result = op.Execute(input);

        var rn = (NivaraColumn<long>)result["rn"];
        Assert.That(rn[0], Is.EqualTo(1));
        Assert.That(rn[1], Is.EqualTo(2));
        Assert.That(rn[2], Is.EqualTo(1));
    }

    // ── TransformSchema ──

    [Test]
    public void Rank_TransformSchema_AppendsLongColumn()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("v") });

        var schema = op.TransformSchema(SchemaOf(("v", typeof(int))));

        Assert.That(schema.HasColumn("rnk"), Is.True);
        Assert.That(schema.GetColumnType("rnk"), Is.EqualTo(typeof(long)));
        Assert.That(schema.ColumnNames.Count, Is.EqualTo(2));
    }

    [Test]
    public void PercentRank_TransformSchema_AppendsDoubleColumn()
    {
        var op = new RankOperation("pct", RankKind.PercentRank, new[] { new SortKey("v") });

        var schema = op.TransformSchema(SchemaOf(("v", typeof(int))));

        Assert.That(schema.GetColumnType("pct"), Is.EqualTo(typeof(double)));
    }

    // ── Validation ──

    [Test]
    public void Rank_ResultColumnCollision_Throws()
    {
        var op = new RankOperation("v", RankKind.Rank, new[] { new SortKey("v") });

        Assert.Throws<ArgumentException>(() => op.TransformSchema(SchemaOf(("v", typeof(int)))));
    }

    [Test]
    public void Rank_MissingPartitionColumn_Throws()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("v") }, new[] { "missing" });

        Assert.Throws<SchemaValidationException>(() => op.TransformSchema(SchemaOf(("v", typeof(int)))));
    }

    [Test]
    public void Rank_MissingOrderColumn_Throws()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("missing") });

        Assert.Throws<SchemaValidationException>(() => op.TransformSchema(SchemaOf(("v", typeof(int)))));
    }

    [Test]
    public void Rank_NonComparableOrderColumn_Throws()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("name") });

        Assert.Throws<SchemaValidationException>(() => op.TransformSchema(SchemaOf(("name", typeof(object)))));
    }

    [Test]
    public void Rank_NoOrderKeys_Throws()
    {
        Assert.Throws<ArgumentException>(() => new RankOperation("rnk", RankKind.Rank, Array.Empty<SortKey>()));
    }

    [Test]
    public void RowNumber_NoOrderKeys_Allowed()
    {
        Assert.DoesNotThrow(() => new RankOperation("rn", RankKind.RowNumber, Array.Empty<SortKey>()));
    }

    [Test]
    public void Rank_NullOrderKey_ProducesNullResult()
    {
        var input = Columns(("v", NivaraColumn<int>.CreateFromSpans(new[] { 1, 0, 3 }, new[] { false, true, false })));
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("v") });

        var result = op.Execute(input);

        var rnk = (NivaraColumn<long>)result["rnk"];
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk.IsNull(1), Is.True);
        Assert.That(rnk[2], Is.EqualTo(2));
    }

    [Test]
    public void Rank_MissingSourceAtExecution_ThrowsQueryExecutionException()
    {
        var input = Columns(("v", IntColumn(1, 2, 3)));
        var op = new RankOperation("rnk", RankKind.Rank, new[] { new SortKey("missing") });

        Assert.Throws<QueryExecutionException>(() => op.Execute(input));
    }
}
