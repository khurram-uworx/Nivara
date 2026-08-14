using Nivara.Exceptions;
using Nivara.Operations;
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

    // ── QueryFrame pipeline ──

    static NivaraFrame FrameWith(params (string Name, IColumn Column)[] columns)
        => new(columns);

    [Test]
    public void QueryFrame_Rank_Collect_AddsResultColumn()
    {
        using var frame = FrameWith(("v", IntColumn(10, 20, 20, 30)));
        using var result = frame.AsQueryFrame().Rank("rnk", new[] { new SortKey("v") }).Collect();

        Assert.That(result.HasColumn("v"), Is.True);
        Assert.That(result.HasColumn("rnk"), Is.True);
        var rnk = result.GetColumn<long>("rnk");
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk[1], Is.EqualTo(2));
        Assert.That(rnk[2], Is.EqualTo(2));
        Assert.That(rnk[3], Is.EqualTo(4));
    }

    [Test]
    public void QueryFrame_DenseRankAndPercentRank_AddColumns()
    {
        using var frame = FrameWith(("v", IntColumn(10, 20, 20, 30)));
        using var result = frame.AsQueryFrame()
            .DenseRank("dense", new[] { new SortKey("v") })
            .PercentRank("pct", new[] { new SortKey("v") })
            .Collect();

        var dense = result.GetColumn<long>("dense");
        var pct = result.GetColumn<double>("pct");
        Assert.That(dense[0], Is.EqualTo(1));
        Assert.That(dense[1], Is.EqualTo(2));
        Assert.That(dense[2], Is.EqualTo(2));
        Assert.That(dense[3], Is.EqualTo(3));
        Assert.That(pct[0], Is.EqualTo(0.0));
        Assert.That(pct[3], Is.EqualTo(1.0));
    }

    [Test]
    public void QueryFrame_RowNumber_NoOrderBy_Sequential()
    {
        using var frame = FrameWith(("v", IntColumn(5, 1, 3)));
        using var result = frame.AsQueryFrame().RowNumber("rn").Collect();

        var rn = result.GetColumn<long>("rn");
        Assert.That(rn[0], Is.EqualTo(1));
        Assert.That(rn[1], Is.EqualTo(2));
        Assert.That(rn[2], Is.EqualTo(3));
    }

    [Test]
    public void QueryFrame_RowNumber_WithPartitionAndOrder_ResetsPerGroup()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "A" })),
            ("v", IntColumn(30, 10, 20, 20)));
        using var result = frame.AsQueryFrame()
            .RowNumber("rn", new[] { "g" }, new[] { new SortKey("v", SortDirection.Descending) })
            .Collect();

        var rn = result.GetColumn<long>("rn");
        Assert.That(rn[0], Is.EqualTo(1));
        Assert.That(rn[1], Is.EqualTo(3));
        Assert.That(rn[2], Is.EqualTo(1));
        Assert.That(rn[3], Is.EqualTo(2));
    }

    [Test]
    public void QueryFrame_Rank_SchemaReflectsColumnTypes()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        var queryFrame = frame.AsQueryFrame()
            .Rank("rnk", new[] { new SortKey("v") })
            .PercentRank("pct", new[] { new SortKey("v") });

        var schema = queryFrame.Schema;
        Assert.That(schema.HasColumn("rnk"), Is.True);
        Assert.That(schema.GetColumnType("rnk"), Is.EqualTo(typeof(long)));
        Assert.That(schema.HasColumn("pct"), Is.True);
        Assert.That(schema.GetColumnType("pct"), Is.EqualTo(typeof(double)));
    }

    [Test]
    public void QueryFrame_Rank_ComposesWithOtherWindowOps()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        using var result = frame.AsQueryFrame()
            .CumulativeCount("v", "count")
            .Rank("rnk", new[] { new SortKey("v") })
            .Collect();

        Assert.That(result.HasColumn("count"), Is.True);
        Assert.That(result.HasColumn("rnk"), Is.True);
        Assert.That(result.GetColumn<long>("count")[2], Is.EqualTo(3));
        Assert.That(result.GetColumn<long>("rnk")[0], Is.EqualTo(1));
    }

    [Test]
    public void QueryFrame_Rank_MissingPartitionColumn_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().Rank("rnk", new[] { new SortKey("v") }, "missing").Collect());
    }

    [Test]
    public void QueryFrame_Rank_NoOrderKeys_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));

        Assert.Throws<ArgumentException>(() => frame.AsQueryFrame().Rank("rnk", Array.Empty<SortKey>()));
    }

    // ── WindowSpec ──

    [Test]
    public void Rank_WithSpec_MatchesNamedColumns()
    {
        var input = Columns(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "B" })),
            ("v", IntColumn(10, 20, 20, 30)));
        var spec = new WindowSpec().PartitionBy("g").OrderBy("v");
        var op = new RankOperation("rnk", RankKind.Rank, spec);

        var result = op.Execute(input);

        var rnk = (NivaraColumn<long>)result["rnk"];
        Assert.That(rnk[0], Is.EqualTo(1));
        Assert.That(rnk[1], Is.EqualTo(2));
        Assert.That(rnk[2], Is.EqualTo(1));
        Assert.That(rnk[3], Is.EqualTo(2));
    }

    [Test]
    public void Rank_WithSpec_TransformSchemaAppendsLongColumn()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new WindowSpec().OrderBy("v"));

        var schema = op.TransformSchema(SchemaOf(("v", typeof(int))));

        Assert.That(schema.HasColumn("rnk"), Is.True);
        Assert.That(schema.GetColumnType("rnk"), Is.EqualTo(typeof(long)));
    }

    [Test]
    public void Rank_WithSpec_RequiresOrderKeys()
    {
        Assert.Throws<ArgumentException>(() => new RankOperation("rnk", RankKind.Rank, new WindowSpec()));
    }

    [Test]
    public void Rank_WithSpec_MissingPartitionColumn_Throws()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new WindowSpec().PartitionBy("missing").OrderBy("v"));

        Assert.Throws<SchemaValidationException>(() => op.TransformSchema(SchemaOf(("v", typeof(int)))));
    }

    [Test]
    public void Rank_WithSpec_MissingOrderColumn_Throws()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new WindowSpec().OrderBy("missing"));

        Assert.Throws<SchemaValidationException>(() => op.TransformSchema(SchemaOf(("v", typeof(int)))));
    }

    [Test]
    public void Rank_WithSpec_NonComparableOrderColumn_Throws()
    {
        var op = new RankOperation("rnk", RankKind.Rank, new WindowSpec().OrderBy("name"));

        Assert.Throws<SchemaValidationException>(() => op.TransformSchema(SchemaOf(("name", typeof(object)))));
    }

    [Test]
    public void QueryFrame_Rank_WithSpec_Collect_AddsResultColumn()
    {
        using var frame = FrameWith(
            ("g", NivaraColumn<string>.CreateForReferenceType(new[] { "A", "A", "B", "B" })),
            ("t", IntColumn(2, 1, 2, 1)),
            ("v", IntColumn(10, 20, 30, 40)));
        var spec = frame.AsQueryFrame().Over().PartitionBy("g").OrderBy("t");

        using var result = frame.AsQueryFrame().Rank("rnk", spec).Collect();

        Assert.That(result.HasColumn("rnk"), Is.True);
        var rnk = result.GetColumn<long>("rnk");
        Assert.That(rnk[0], Is.EqualTo(2));
        Assert.That(rnk[1], Is.EqualTo(1));
        Assert.That(rnk[2], Is.EqualTo(2));
        Assert.That(rnk[3], Is.EqualTo(1));
    }

    [Test]
    public void QueryFrame_Rank_WithSpec_MissingPartitionColumn_Throws()
    {
        using var frame = FrameWith(("v", IntColumn(1, 2, 3)));
        var spec = new WindowSpec().PartitionBy("missing").OrderBy("v");

        Assert.Throws<QueryExecutionException>(() => frame.AsQueryFrame().Rank("rnk", spec).Collect());
    }
}
