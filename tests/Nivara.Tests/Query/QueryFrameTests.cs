using Nivara.Exceptions;
using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class QueryFrameTests
{
    static NivaraFrame CreateTestFrame()
    {
        var intColumn = NivaraColumn<int>.Create(new[] { 10, 20, 30, 40, 50 });
        var strColumn = NivaraColumn<string>.CreateForReferenceType(new[] { "a", "b", "c", "d", "e" });
        return NivaraFrame.Create(
            ("X", intColumn),
            ("Y", strColumn)
        );
    }

    static NivaraFrame CreateTestFrameWithNulls()
    {
        var intColumn = NivaraColumn<int>.CreateFromNullable(new int?[] { 1, null, 3, null, 5 });
        var strColumn = NivaraColumn<string>.CreateForReferenceType(new string[] { "a", null!, "c", null!, "e" });
        return NivaraFrame.Create(
            ("X", intColumn),
            ("Y", strColumn)
        );
    }

    // ── SelectRows ──

    [Test]
    public void SelectRows_ContiguousIndices_ExtractsCorrectRows()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().SelectRows(1, 2, 3);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.ColumnCount, Is.EqualTo(2));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(20));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(30));
        Assert.That(result.GetColumn<int>("X")[2], Is.EqualTo(40));
        Assert.That(result.GetColumn<string>("Y")[0], Is.EqualTo("b"));
        Assert.That(result.GetColumn<string>("Y")[2], Is.EqualTo("d"));
    }

    [Test]
    public void SelectRows_SparseIndices_ExtractsCorrectRows()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().SelectRows(0, 2, 4);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(10));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(30));
        Assert.That(result.GetColumn<int>("X")[2], Is.EqualTo(50));
    }

    [Test]
    public void SelectRows_ReversedIndices_ExtractsCorrectRows()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().SelectRows(4, 3, 2, 1, 0);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(5));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(50));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(40));
        Assert.That(result.GetColumn<int>("X")[4], Is.EqualTo(10));
    }

    [Test]
    public void SelectRows_NullPropagation_NullsPreserved()
    {
        using var frame = CreateTestFrameWithNulls();
        var queryFrame = frame.AsQueryFrame().SelectRows(0, 1, 2, 3, 4);
        using var result = queryFrame.Collect();

        var xCol = result.GetColumn<int>("X");
        var yCol = result.GetColumn<string>("Y");
        Assert.That(xCol.IsNull(0), Is.False);
        Assert.That(xCol.IsNull(1), Is.True);
        Assert.That(xCol.IsNull(2), Is.False);
        Assert.That(xCol.IsNull(3), Is.True);
        Assert.That(xCol.IsNull(4), Is.False);
        Assert.That(yCol.IsNull(1), Is.True);
        Assert.That(yCol.IsNull(3), Is.True);
    }

    [Test]
    public void SelectRows_InvalidIndex_ThrowsArgumentOutOfRangeException()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().SelectRows(0, 99);

        var ex = Assert.Throws<QueryExecutionException>(() => queryFrame.Collect());
        Assert.That(ex!.Message, Does.Contain("out of range"));
    }

    [Test]
    public void SelectRows_EmptyIndices_ThrowsArgumentException()
    {
        using var frame = CreateTestFrame();
        Assert.Throws<ArgumentException>(() => frame.AsQueryFrame().SelectRows());
    }

    // ── Skip / Take / Slice ──

    [Test]
    public void Skip_ValidCount_ReturnsRemainingRows()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Skip(2);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(30));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(40));
        Assert.That(result.GetColumn<int>("X")[2], Is.EqualTo(50));
    }

    [Test]
    public void Take_ValidCount_ReturnsFirstNRows()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Take(3);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(10));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(20));
        Assert.That(result.GetColumn<int>("X")[2], Is.EqualTo(30));
    }

    [Test]
    public void Slice_Combined_CorrectSlice()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Slice(1, 3);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(3));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(20));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(30));
        Assert.That(result.GetColumn<int>("X")[2], Is.EqualTo(40));
    }

    [Test]
    public void SkipThenTake_Chained_CorrectResult()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Skip(1).Take(2);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(2));
        Assert.That(result.GetColumn<int>("X")[0], Is.EqualTo(20));
        Assert.That(result.GetColumn<int>("X")[1], Is.EqualTo(30));
    }

    [Test]
    public void Skip_Zero_ReturnsAllRows()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Skip(0);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(5));
    }

    [Test]
    public void Take_Zero_ReturnsEmptyFrame()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Take(0);
        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(0));
        Assert.That(result.ColumnCount, Is.EqualTo(2));
    }

    [Test]
    public void Skip_Negative_ThrowsArgumentOutOfRangeException()
    {
        using var frame = CreateTestFrame();
        Assert.Throws<ArgumentOutOfRangeException>(() => frame.AsQueryFrame().Skip(-1));
    }

    [Test]
    public void Take_Negative_ThrowsArgumentOutOfRangeException()
    {
        using var frame = CreateTestFrame();
        Assert.Throws<ArgumentOutOfRangeException>(() => frame.AsQueryFrame().Take(-1));
    }

    // ── Parallel execution ──

    [Test]
    public void SelectRows_Parallel_RoundTripsIdentical()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().SelectRows(4, 2, 0);

        var engine = new ExecutionEngine();
        using var serialResult = engine.Execute(queryFrame.ToQueryPlan(), new NivaraExecutionContext(ExecutionStrategy.Eager));
        using var parallelResult = engine.Execute(queryFrame.ToQueryPlan(), new NivaraExecutionContext(ExecutionStrategy.Parallel));

        Assert.That(parallelResult.RowCount, Is.EqualTo(serialResult.RowCount));
        Assert.That(parallelResult.GetColumn<int>("X")[0], Is.EqualTo(serialResult.GetColumn<int>("X")[0]));
        Assert.That(parallelResult.GetColumn<int>("X")[1], Is.EqualTo(serialResult.GetColumn<int>("X")[1]));
        Assert.That(parallelResult.GetColumn<int>("X")[2], Is.EqualTo(serialResult.GetColumn<int>("X")[2]));
    }

    [Test]
    public void Slice_Parallel_RoundTripsIdentical()
    {
        using var frame = CreateTestFrame();
        var queryFrame = frame.AsQueryFrame().Skip(1).Take(2);

        var engine = new ExecutionEngine();
        using var serialResult = engine.Execute(queryFrame.ToQueryPlan(), new NivaraExecutionContext(ExecutionStrategy.Eager));
        using var parallelResult = engine.Execute(queryFrame.ToQueryPlan(), new NivaraExecutionContext(ExecutionStrategy.Parallel));

        Assert.That(parallelResult.RowCount, Is.EqualTo(serialResult.RowCount));
        Assert.That(parallelResult.GetColumn<int>("X")[0], Is.EqualTo(serialResult.GetColumn<int>("X")[0]));
        Assert.That(parallelResult.GetColumn<int>("X")[1], Is.EqualTo(serialResult.GetColumn<int>("X")[1]));
    }
}
