using Nivara.Diagnostics;
using Nivara.Execution;
using Nivara.Expressions;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class QueryFrameDiagnosticsTests
{
    static NivaraFrame CreateTestFrame(int rowCount = 10)
    {
        var intData = new int[rowCount];
        var strData = new string[rowCount];
        for (int i = 0; i < rowCount; i++)
        {
            intData[i] = i * 10;
            strData[i] = $"val{i}";
        }
        return NivaraFrame.Create(
            ("X", NivaraColumn<int>.Create(intData)),
            ("Y", NivaraColumn<string>.Create(strData)));
    }

    // ── Collect / CollectAsync ──

    [Test]
    public void Collect_PopulatesDiagnostics_WithRowCounters()
    {
        using var frame = CreateTestFrame(10);
        var queryFrame = frame.AsQueryFrame().Filter(ColumnExpressions.Col("X") > 50);

        using var result = queryFrame.Collect();

        Assert.That(result.RowCount, Is.EqualTo(4));

        var diagnostics = queryFrame.GetExecutionDiagnostics();
        Assert.That(diagnostics, Is.Not.Null);
        Assert.That(diagnostics!.ExecutionStrategy, Is.EqualTo(ExecutionStrategy.Lazy));
        Assert.That(diagnostics.RowsRead, Is.EqualTo(10));
        Assert.That(diagnostics.RowsReturned, Is.EqualTo(4));
        Assert.That(diagnostics.MaterializedColumns, Is.EqualTo(2));
        Assert.That(diagnostics.TotalExecutionTime, Is.GreaterThan(TimeSpan.Zero));
        Assert.That(diagnostics.OperationTimings, Is.Not.Empty);
    }

    [Test]
    public async Task CollectAsync_PopulatesDiagnostics_WithRowCounters()
    {
        using var frame = CreateTestFrame(10);
        var queryFrame = frame.AsQueryFrame().Filter(ColumnExpressions.Col("X") > 50);

        using var result = await queryFrame.CollectAsync();

        var diagnostics = queryFrame.GetExecutionDiagnostics();
        Assert.That(diagnostics, Is.Not.Null);
        Assert.That(diagnostics!.RowsRead, Is.EqualTo(10));
        Assert.That(diagnostics.RowsReturned, Is.EqualTo(4));
        Assert.That(diagnostics.MaterializedColumns, Is.EqualTo(2));
    }

    [Test]
    public void LastExecutionDiagnostics_BeforeExecution_IsNull()
    {
        using var frame = CreateTestFrame(3);
        var queryFrame = frame.AsQueryFrame();

        Assert.That(queryFrame.GetExecutionDiagnostics(), Is.Null);
        Assert.That(queryFrame.LastExecutionDiagnostics, Is.Null);
    }

    [Test]
    public void GetExecutionDiagnostics_And_LastExecutionDiagnostics_ReturnSameInstance()
    {
        using var frame = CreateTestFrame(3);
        var queryFrame = frame.AsQueryFrame();

        using var _ = queryFrame.Collect();

        Assert.That(queryFrame.GetExecutionDiagnostics(), Is.SameAs(queryFrame.LastExecutionDiagnostics));
    }

    [Test]
    public void Collect_RepeatedExecution_RefreshesDiagnostics()
    {
        using var frame = CreateTestFrame(10);
        var queryFrame = frame.AsQueryFrame().Filter(ColumnExpressions.Col("X") > 50);

        using (queryFrame.Collect())
        {
            Assert.That(queryFrame.GetExecutionDiagnostics()!.RowsReturned, Is.EqualTo(4));
        }

        var firstDiagnostics = queryFrame.GetExecutionDiagnostics();

        var filtered = queryFrame.Filter(ColumnExpressions.Col("X") > 70);
        using (filtered.Collect())
        {
            Assert.That(filtered.GetExecutionDiagnostics()!.RowsReturned, Is.EqualTo(2));
        }

        Assert.That(queryFrame.GetExecutionDiagnostics(), Is.SameAs(firstDiagnostics));
    }

    [Test]
    public void Collect_Diagnostics_GetSummary_IncludesRowCounters()
    {
        using var frame = CreateTestFrame(10);
        var queryFrame = frame.AsQueryFrame().Filter(ColumnExpressions.Col("X") > 50);

        using var _ = queryFrame.Collect();

        var summary = queryFrame.GetExecutionDiagnostics()!.GetSummary();
        Assert.That(summary.RowsRead, Is.EqualTo(10));
        Assert.That(summary.RowsReturned, Is.EqualTo(4));
        Assert.That(summary.MaterializedColumns, Is.EqualTo(2));
        Assert.That(summary.OperationCount, Is.GreaterThan(0));
    }

    [Test]
    public void Collect_Diagnostics_GenerateReport_IncludesRowCounters()
    {
        using var frame = CreateTestFrame(10);
        var queryFrame = frame.AsQueryFrame().Filter(ColumnExpressions.Col("X") > 50);

        using var _ = queryFrame.Collect();

        var report = queryFrame.GetExecutionDiagnostics()!.GenerateReport();
        Assert.That(report, Does.Contain("Rows: 10 read, 4 returned, 2 columns"));
    }

    // ── AsStream ──

    [Test]
    public async Task AsStream_AfterFullEnumeration_PopulatesDiagnostics()
    {
        using var frame = CreateTestFrame(10);
        var queryFrame = frame.AsQueryFrame().Filter(ColumnExpressions.Col("X") > 50);

        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in queryFrame.AsStream(chunkSize: 3))
        {
            chunks.Add(chunk);
        }

        try
        {
            Assert.That(chunks, Is.Not.Empty);
            Assert.That(chunks.Sum(c => c.RowCount), Is.EqualTo(4));
        }
        finally
        {
            foreach (var chunk in chunks)
                chunk.Dispose();
        }

        var diagnostics = queryFrame.GetExecutionDiagnostics();
        Assert.That(diagnostics, Is.Not.Null);
        Assert.That(diagnostics!.RowsRead, Is.EqualTo(10));
        Assert.That(diagnostics.RowsReturned, Is.EqualTo(4));
        Assert.That(diagnostics.MaterializedColumns, Is.EqualTo(2));
    }

    [Test]
    public async Task AsStream_ChunkSource_AccumulatesRowsReadAcrossChunks()
    {
        var source = new ChunkedTestSource(totalRows: 25, chunkSize: 10);
        using var queryFrame = new QueryFrame(source)
            .Filter(ColumnExpressions.Col("V") > 5);

        var chunks = new List<NivaraFrame>();
        int totalRows;
        try
        {
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 10))
                chunks.Add(chunk);

            Assert.That(chunks, Is.Not.Empty);
            totalRows = chunks.Sum(c => c.RowCount);
        }
        finally
        {
            foreach (var chunk in chunks)
                chunk.Dispose();
        }

        var diagnostics = queryFrame.GetExecutionDiagnostics();
        Assert.That(diagnostics, Is.Not.Null);
        Assert.That(diagnostics!.RowsRead, Is.EqualTo(25));
        Assert.That(diagnostics.RowsReturned, Is.EqualTo(totalRows));
        Assert.That(diagnostics.MaterializedColumns, Is.EqualTo(1));
    }

    // ── ExecutionDiagnostics row-counter defaults ──

    [Test]
    public void ExecutionDiagnostics_NewInstance_RowCountersDefaultToZero()
    {
        var diagnostics = new ExecutionDiagnostics();

        Assert.That(diagnostics.RowsRead, Is.EqualTo(0));
        Assert.That(diagnostics.RowsReturned, Is.EqualTo(0));
        Assert.That(diagnostics.MaterializedColumns, Is.EqualTo(0));
    }

    sealed class ChunkedTestSource : IQuerySource
    {
        readonly int totalRows;
        readonly int chunkSize;

        public ChunkedTestSource(int totalRows, int chunkSize)
        {
            this.totalRows = totalRows;
            this.chunkSize = chunkSize;
        }

        public Schema Schema => new(new[] { ("V", typeof(int)) });
        public bool IsLazy => false;
        public bool CanReadInChunks => true;
        public int? EstimatedRowCount => totalRows;

        public IReadOnlyDictionary<string, IColumn> Execute()
            => new Dictionary<string, IColumn> { ["V"] = NivaraColumn<int>.Create(BuildData(0, totalRows)) };

        public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
        {
            var start = chunkIndex * this.chunkSize;
            var length = Math.Min(this.chunkSize, totalRows - start);
            if (length <= 0)
                return new Dictionary<string, IColumn>(0);
            return new Dictionary<string, IColumn> { ["V"] = NivaraColumn<int>.Create(BuildData(start, length)) };
        }

        public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
            int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            return ReadChunk(chunkIndex, chunkSize);
        }

        public void Dispose()
        {
        }

        static int[] BuildData(int start, int count)
        {
            var data = new int[count];
            for (int i = 0; i < count; i++)
                data[i] = start + i;
            return data;
        }
    }
}
