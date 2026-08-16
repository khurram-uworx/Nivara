using Nivara.Execution;
using Nivara.Expressions;
using Nivara.IO;
using Nivara.Linq;
using Nivara.Operations;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class AsyncStreamingTests
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

    sealed class Person
    {
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }
        public double Salary { get; set; }
    }

    sealed class TestRow
    {
        public int X { get; set; }
        public string Y { get; set; } = string.Empty;
    }

    sealed class DisposalRecordingSource : IQuerySource
    {
        public Schema Schema { get; } = new(new[] { ("A", typeof(int)) });
        public bool IsLazy => true;
        public bool Disposed { get; private set; }

        public IReadOnlyDictionary<string, IColumn> Execute() =>
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(new[] { 1, 2, 3 }) };

        public void Dispose() => Disposed = true;
    }

    sealed class CancellationChunkSource : IQuerySource
    {
        readonly int totalRowCount;
        int chunksRead;
        int cancelTarget = -1;
        CancellationTokenSource? cancelCts;

        public CancellationChunkSource(int totalRows, int cancelAfterChunks)
        {
            totalRowCount = totalRows;
            cancelTarget = cancelAfterChunks;
        }

        public Schema Schema => new(new[] { ("A", typeof(int)) });
        public bool IsLazy => false;
        public bool CanReadInChunks => true;
        public int? EstimatedRowCount => totalRowCount;
        public int ChunksRead => chunksRead;

        public void CancelWhenChunkCountReaches(CancellationTokenSource cts, int targetChunk)
        {
            cancelCts = cts;
            cancelTarget = targetChunk;
        }

        public IReadOnlyDictionary<string, IColumn> Execute() =>
            new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(BuildData(0, totalRowCount)) };

        public IReadOnlyDictionary<string, IColumn> ReadChunk(int chunkIndex, int chunkSize)
        {
            var start = chunkIndex * chunkSize;
            var length = Math.Min(chunkSize, totalRowCount - start);
            if (length <= 0)
                return new Dictionary<string, IColumn>(0);
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(BuildData(start, length)) };
        }

        public async ValueTask<IReadOnlyDictionary<string, IColumn>> ReadChunkAsync(
            int chunkIndex, int chunkSize, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var n = Interlocked.Increment(ref chunksRead);
            if (cancelCts != null && n >= cancelTarget)
                cancelCts.Cancel();

            await Task.Yield();

            var start = chunkIndex * chunkSize;
            var length = Math.Min(chunkSize, totalRowCount - start);
            if (length <= 0)
                return new Dictionary<string, IColumn>(0);
            return new Dictionary<string, IColumn> { ["A"] = NivaraColumn<int>.Create(BuildData(start, length)) };
        }

        static int[] BuildData(int start, int count)
        {
            var data = new int[count];
            for (int i = 0; i < count; i++)
                data[i] = start + i;
            return data;
        }

        public void Dispose()
        {
        }
    }

    [Test]
    public async Task CollectAsync_ParityWithCollect_ProducesIdenticalResults()
    {
        var frame = CreateTestFrame(20);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 50);

            using var syncResult = queryFrame.Collect();
            using var asyncResult = await queryFrame.CollectAsync();

            Assert.That(asyncResult.RowCount, Is.EqualTo(syncResult.RowCount));
            Assert.That(asyncResult.ColumnNames, Is.EquivalentTo(syncResult.ColumnNames));

            for (int i = 0; i < syncResult.RowCount; i++)
            {
                Assert.That(asyncResult.GetColumn<int>("X")[i], Is.EqualTo(syncResult.GetColumn<int>("X")[i]));
                Assert.That(asyncResult.GetColumn<string>("Y")[i], Is.EqualTo(syncResult.GetColumn<string>("Y")[i]));
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public void CollectAsync_Cancellation_ThrowsOperationCanceledException()
    {
        var frame = CreateTestFrame(5);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 0);

            using var ct = new CancellationTokenSource();
            ct.Cancel();

            Assert.ThrowsAsync<OperationCanceledException>(() => queryFrame.CollectAsync(ct.Token));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task CollectAsync_WithSelect_ParityWithCollect()
    {
        var frame = CreateTestFrame(20);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 50)
                .Select(ColumnExpressions.Col("X") * 2);

            using var syncResult = queryFrame.Collect();
            using var asyncResult = await queryFrame.CollectAsync();

            Assert.That(asyncResult.RowCount, Is.EqualTo(syncResult.RowCount));

            for (int i = 0; i < syncResult.RowCount; i++)
                Assert.That(asyncResult.GetColumn<int>(asyncResult.ColumnNames[0])[i],
                    Is.EqualTo(syncResult.GetColumn<int>(syncResult.ColumnNames[0])[i]));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task AsStream_YieldsChunks_MatchesCollect()
    {
        var frame = CreateTestFrame(25);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 20);

            using var syncResult = queryFrame.Collect();

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 5))
                chunks.Add(chunk);

            int totalRows = chunks.Sum(c => c.RowCount);
            Assert.That(totalRows, Is.EqualTo(syncResult.RowCount));

            foreach (var chunk in chunks)
                chunk.Dispose();
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task AsStream_SingleNonChunkedSource_ReturnsSingleChunk()
    {
        var frame = CreateTestFrame(5);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 10);

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 5))
                chunks.Add(chunk);

            Assert.That(chunks.Count, Is.EqualTo(1));
            Assert.That(chunks[0].RowCount, Is.EqualTo(3));

            foreach (var chunk in chunks)
                chunk.Dispose();
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task AsStream_HonorsRequestedChunkSize()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 10000);
            var queryFrame = Nivara.IO.Csv.ScanFrame(csvPath);

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 2000))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Count, Is.EqualTo(5));
                foreach (var chunk in chunks)
                    Assert.That(chunk.RowCount, Is.EqualTo(2000));
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task AsStream_SmallChunkSize_NotClamped()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 100);
            var queryFrame = Nivara.IO.Csv.ScanFrame(csvPath);

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 5))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Count, Is.EqualTo(20));
                foreach (var chunk in chunks)
                    Assert.That(chunk.RowCount, Is.LessThanOrEqualTo(5));
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void ParquetLazySource_MultipleRowGroups_ExecutesCorrectly()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 2500).Select(i => i * 2).ToArray();
            var frame = NivaraFrame.Create(("Index", NivaraColumn<int>.Create(values)));

            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            var source = new ParquetLazySource(tempFile);
            Assert.That(source.RowGroupCount, Is.EqualTo(3));
            Assert.That(source.EstimatedRowCount, Is.EqualTo(2500));
            Assert.That(source.CanReadInChunks, Is.True);
            Assert.That(source.IsLazy, Is.True);

            var result = source.Execute();
            var column = result["Index"];

            Assert.That(column.Length, Is.EqualTo(2500));
            Assert.That(column.GetValue(0), Is.EqualTo(0));
            Assert.That(column.GetValue(2499), Is.EqualTo(4998));

            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task ParquetLazySource_ReadChunkAsync_ChunksAtRowGroupBoundaries()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 2500).Select(i => i * 2).ToArray();
            var frame = NivaraFrame.Create(("Index", NivaraColumn<int>.Create(values)));

            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            var source = new ParquetLazySource(tempFile);

            var chunk0 = source.ReadChunk(0, 10000);
            var chunk1 = source.ReadChunk(1, 10000);
            var chunk2 = source.ReadChunk(2, 10000);
            var chunk3 = await source.ReadChunkAsync(3, 10000);

            Assert.That(chunk0["Index"].Length, Is.EqualTo(1000));
            Assert.That(chunk1["Index"].Length, Is.EqualTo(1000));
            Assert.That(chunk2["Index"].Length, Is.EqualTo(500));
            Assert.That(chunk3.Count, Is.EqualTo(0));

            Assert.That(chunk0["Index"].GetValue(0), Is.EqualTo(0));
            Assert.That(chunk1["Index"].GetValue(0), Is.EqualTo(2000));
            Assert.That(chunk2["Index"].GetValue(499), Is.EqualTo(4998));

            source.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task ParquetLazySource_ExecuteAsync_WithFilterAndSelect()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 2500).Select(i => i * 2).ToArray();
            var frame = NivaraFrame.Create(("Index", NivaraColumn<int>.Create(values)));

            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1000));
            frame.Dispose();

            var source = new ParquetLazySource(tempFile);
            using var queryFrame = new QueryFrame(source)
                .Filter(ColumnExpressions.Col("Index") > 1000)
                .Select(ColumnExpressions.Col("Index") * 2);

            using var result = await queryFrame.CollectAsync();

            // Filter Index > 1000 over 0,2,4,...,4998 (2500 rows) keeps 1002..4998 = 1999 rows
            Assert.That(result.RowCount, Is.EqualTo(1999));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ParquetLazySource_Execute_MatchesReadParquet()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 500).Select(i => i + 1).ToArray();
            var strings = Enumerable.Range(0, 500).Select(i => $"row-{i}").ToArray();
            var frame = NivaraFrame.Create(
                ("Num", NivaraColumn<int>.Create(values)),
                ("Str", NivaraColumn<string>.Create(strings)));

            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 200));
            frame.Dispose();

            using var eagerResult = NivaraParquetReader.ReadParquet(tempFile);
            var lazySource = new ParquetLazySource(tempFile);
            var lazyData = lazySource.Execute();

            Assert.That(lazyData["Num"].Length, Is.EqualTo(eagerResult.RowCount));
            Assert.That(lazyData["Str"].Length, Is.EqualTo(eagerResult.RowCount));
            Assert.That(lazyData["Num"].GetValue(499), Is.EqualTo(500));
            Assert.That(lazyData["Str"].GetValue(499), Is.EqualTo("row-499"));

            lazySource.Dispose();
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public void ParquetLazySource_NonExistentFile_ThrowsOnSchemaAccess()
    {
        var missingPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString() + ".parquet");
        var source = new ParquetLazySource(missingPath);

        Assert.Throws<Nivara.Exceptions.DataSourceException>(() =>
        {
            var schema = source.Schema;
        });
    }

    [Test]
    public async Task ParquetLazySource_ScanQuery_PersonTypedRows()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var names = new[] { "Alice", "Bob", "Charlie" };
            var ages = new[] { 30, 25, 35 };
            var salaries = new[] { 75000.0, 65000.0, 85000.0 };
            var frame = NivaraFrame.Create(
                ("Name", NivaraColumn<string>.Create(names)),
                ("Age", NivaraColumn<int>.Create(ages)),
                ("Salary", NivaraColumn<double>.Create(salaries)));

            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 1));
            frame.Dispose();

            var query = NivaraParquetReader.ScanQuery<Person>(tempFile);
            var rows = await query.ToListAsync();

            Assert.That(rows.Count, Is.EqualTo(3));
            Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "Alice", "Bob", "Charlie" }));
            Assert.That(rows.Select(r => r.Age), Is.EqualTo(new[] { 30, 25, 35 }));
            Assert.That(rows.Select(r => r.Salary), Is.EqualTo(new[] { 75000.0, 65000.0, 85000.0 }));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task CollectAsync_EmptyDataFrame_ReturnsEmpty()
    {
        var frame = NivaraFrame.Create(("Empty", NivaraColumn<int>.Create(Array.Empty<int>()))
);
        try
        {
            var queryFrame = frame.AsQueryFrame();

            using var result = await queryFrame.CollectAsync();

            Assert.That(result.RowCount, Is.EqualTo(0));
            Assert.That(result.ColumnCount, Is.EqualTo(1));
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task QueryFrame_DisposeAsync_ReleasesResources()
    {
        var frame = CreateTestFrame(3);
        try
        {
            var queryFrame = frame.AsQueryFrame()
                .Filter(ColumnExpressions.Col("X") > 5);

            await queryFrame.DisposeAsync();

            Assert.Throws<ObjectDisposedException>(() => queryFrame.Collect());
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public void QueryFrame_Dispose_ReleasesSourceResources()
    {
        var source = new DisposalRecordingSource();
        var queryFrame = new QueryFrame(source);

        queryFrame.Dispose();

        Assert.That(source.Disposed, Is.True,
            "Sync Dispose must release the underlying IQuerySource to align with DisposeAsync");
        Assert.Throws<ObjectDisposedException>(() => queryFrame.Collect());
    }

    [Test]
    public async Task QueryFrame_DisposeAsync_ReleasesSourceResources()
    {
        var source = new DisposalRecordingSource();
        var queryFrame = new QueryFrame(source);

        await queryFrame.DisposeAsync();

        Assert.That(source.Disposed, Is.True,
            "DisposeAsync must release the underlying IQuerySource");
        Assert.Throws<ObjectDisposedException>(() => queryFrame.Collect());
    }

    [Test]
    public async Task CollectAsync_CsvLazySource_ParityWithCollect()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        var csvPath = Path.Combine(tempDir, "data.csv");

        try
        {
            var lines = new List<string> { "Name,Age,Salary" };
            for (int i = 0; i < 100; i++)
                lines.Add($"Person{i},20 + i,50000 + i");
            File.WriteAllText(csvPath, string.Join("\n", lines));

            var queryFrame = Nivara.IO.Csv.ScanFrame(csvPath);

            using var syncResult = queryFrame.Collect();
            using var asyncResult = await queryFrame.CollectAsync();

            Assert.That(asyncResult.RowCount, Is.EqualTo(syncResult.RowCount));
            Assert.That(asyncResult.ColumnNames, Is.EquivalentTo(syncResult.ColumnNames));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task CollectAsync_JsonLazySource_ParityWithCollect()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        var jsonPath = Path.Combine(tempDir, "data.json");

        try
        {
            var records = Enumerable.Range(0, 100)
                .Select(i => $"{{\"Name\":\"Person{i}\",\"Age\":{20 + i},\"Salary\":{50000 + i}}}");
            File.WriteAllText(jsonPath, "[" + string.Join(",", records) + "]");

            var queryFrame = Nivara.IO.Json.ScanFrame(jsonPath);

            using var syncResult = queryFrame.Collect();
            using var asyncResult = await queryFrame.CollectAsync();

            Assert.That(asyncResult.RowCount, Is.EqualTo(syncResult.RowCount));
            Assert.That(asyncResult.ColumnNames, Is.EquivalentTo(syncResult.ColumnNames));

            for (int i = 0; i < syncResult.RowCount; i++)
            {
                Assert.That(asyncResult.GetColumn("Name").GetValue(i), Is.EqualTo(syncResult.GetColumn("Name").GetValue(i)));
                Assert.That(asyncResult.GetColumn("Age").GetValue(i), Is.EqualTo(syncResult.GetColumn("Age").GetValue(i)));
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void CsvLazySource_ReadChunkAsync_CancelledToken_ThrowsOperationCanceledException()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 100);
            using var source = new CsvLazySource(csvPath, CsvOptions.Default);
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            Assert.ThrowsAsync<OperationCanceledException>(() => source.ReadChunkAsync(0, 10, cts.Token).AsTask());
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonLazySource_ReadChunkAsync_CancelledToken_ThrowsOperationCanceledException()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var jsonPath = Path.Combine(tempDir, "data.json");
            var records = Enumerable.Range(0, 100)
                .Select(i => $"{{\"Name\":\"Person{i}\",\"Age\":{20 + i},\"Salary\":{50000 + i}}}");
            File.WriteAllText(jsonPath, "[" + string.Join(",", records) + "]");

            using var source = new JsonLazySource(jsonPath, JsonOptions.Default);
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            Assert.ThrowsAsync<OperationCanceledException>(() => source.ReadChunkAsync(0, 10, cts.Token).AsTask());
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task StreamingStrategy_ChannelPipeline_ParityWithLazy()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 10000);
            var source = new CsvLazySource(csvPath, CsvOptions.Default);
            var plan = new QueryPlan(source, new List<IQueryOperation>
            {
                new FilterOperation(ColumnExpressions.Col("Age") > 30),
                new SelectOperation(new[] { ColumnExpressions.Col("Salary") * 2 })
            });

            var engine = new ExecutionEngine();

            using var streamingResult = await engine.ExecuteAsync(
                plan, new NivaraExecutionContext(ExecutionStrategy.Streaming) { MemoryBudget = 2_000_000 });
            using var lazyResult = await engine.ExecuteAsync(
                plan, new NivaraExecutionContext(ExecutionStrategy.Lazy));

            AssertFrameValuesEqual(streamingResult, lazyResult);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task StreamingStrategy_BoundaryOperation_FlushesAndResumes()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 10000);
            var source = new CsvLazySource(csvPath, CsvOptions.Default);
            var plan = new QueryPlan(source, new List<IQueryOperation>
            {
                new FilterOperation(ColumnExpressions.Col("Age") > 30),
                new SortOperation(new List<SortKey> { new SortKey("Age") })
            });

            var engine = new ExecutionEngine();

            using var streamingResult = await engine.ExecuteAsync(
                plan, new NivaraExecutionContext(ExecutionStrategy.Streaming) { MemoryBudget = 2_000_000 });
            using var lazyResult = await engine.ExecuteAsync(
                plan, new NivaraExecutionContext(ExecutionStrategy.Lazy));

            AssertFrameValuesEqual(streamingResult, lazyResult);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void StreamingStrategy_CancellationMidStream_ThrowsOperationCanceledException()
    {
        var source = new CancellationChunkSource(totalRows: 200_000, cancelAfterChunks: 3);
        var plan = new QueryPlan(source, Array.Empty<IQueryOperation>());
        var engine = new ExecutionEngine();
        using var cts = new CancellationTokenSource();
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
        {
            CancellationToken = cts.Token,
            ChunkSize = 10_000,
        };
        source.CancelWhenChunkCountReaches(cts, 3);

        Assert.ThrowsAsync<OperationCanceledException>(() => engine.ExecuteAsync(plan, context));

        Assert.That(source.ChunksRead, Is.GreaterThan(0), "cancellation must fire mid-stream, not pre-cancelled");
        Assert.That(source.ChunksRead, Is.LessThan(20), "run must not complete the full source");
    }

    [Test]
    public void CsvLazySource_ReadChunk_ReconstructsFullData()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 2500);
            var source = new CsvLazySource(csvPath, CsvOptions.Default);

            var expectedAge = source.Execute()["Age"];

            var chunk0 = source.ReadChunk(0, 1000);
            var chunk1 = source.ReadChunk(1, 1000);
            var chunk2 = source.ReadChunk(2, 1000);

            Assert.That(chunk0["Age"].Length, Is.EqualTo(1000));
            Assert.That(chunk1["Age"].Length, Is.EqualTo(1000));
            Assert.That(chunk2["Age"].Length, Is.EqualTo(500));

            Assert.That(chunk0["Age"].GetValue(0), Is.EqualTo(expectedAge.GetValue(0)));
            Assert.That(chunk1["Age"].GetValue(0), Is.EqualTo(expectedAge.GetValue(1000)));
            Assert.That(chunk2["Age"].GetValue(499), Is.EqualTo(expectedAge.GetValue(2499)));

            // Backward re-read after the reader has reached EOF must reopen and stay correct
            var chunk0Again = source.ReadChunk(0, 1000);
            Assert.That(chunk0Again["Age"].GetValue(999), Is.EqualTo(expectedAge.GetValue(999)));

            source.Dispose();
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task AsStream_NonStreamablePlan_ReturnsSingleMergedFrame()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 1000);
            using var queryFrame = Nivara.IO.Csv.ScanAsQueryFrame(csvPath).Sort("Age");

            using var expected = queryFrame.Collect();

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 100))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Count, Is.EqualTo(1));
                AssertFrameValuesEqual(chunks[0], expected);
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task AsStream_NonChunkCapableSource_ReturnsSingleFrame()
    {
        var frame = CreateTestFrame(25);
        try
        {
            using var queryFrame = frame.AsQueryFrame();
            using var expected = queryFrame.Collect();

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 5))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Count, Is.EqualTo(1));
                AssertFrameValuesEqual(chunks[0], expected);
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    [Test]
    public async Task ScanAsQueryFrame_CsvPublicEntryPoint_StreamsChunks()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var csvPath = CreateCsvFile(tempDir, 10000);
            using var queryFrame = Nivara.IO.Csv.ScanAsQueryFrame(csvPath);

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 2000))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Count, Is.EqualTo(5));
                Assert.That(chunks.Sum(c => c.RowCount), Is.EqualTo(10000));
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task ScanAsQueryFrame_JsonPublicEntryPoint_MatchesCollect()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraAsyncTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        try
        {
            var jsonPath = Path.Combine(tempDir, "data.json");
            var records = Enumerable.Range(0, 100)
                .Select(i => $"{{\"Name\":\"Person{i}\",\"Age\":{20 + i},\"Salary\":{50000 + i}}}");
            File.WriteAllText(jsonPath, "[" + string.Join(",", records) + "]");

            using var queryFrame = Nivara.IO.Json.ScanAsQueryFrame(jsonPath);
            using var expected = queryFrame.Collect();

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 50))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Sum(c => c.RowCount), Is.EqualTo(expected.RowCount));
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void ScanAsQueryFrame_ParquetPublicEntryPoint_CollectsCorrectRows()
    {
        var tempFile = Path.GetTempFileName();
        try
        {
            var values = Enumerable.Range(0, 500).Select(i => i + 1).ToArray();
            var frame = NivaraFrame.Create(("Num", NivaraColumn<int>.Create(values)));
            NivaraParquetWriter.WriteParquet(frame, tempFile,
                ParquetWriteOptions.Default.With(rowGroupSize: 200));
            frame.Dispose();

            using var queryFrame = NivaraParquetReader.ScanAsQueryFrame(tempFile);
            using var result = queryFrame.Collect();

            Assert.That(result.RowCount, Is.EqualTo(500));
            Assert.That(result.GetColumn<int>("Num").GetValue(499), Is.EqualTo(500));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Test]
    public async Task NivaraQuery_T_AsStream_Passthrough()
    {
        var frame = CreateTestFrame(25);
        try
        {
            var query = frame.Query<TestRow>();

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in query.AsStream(chunkSize: 5))
                chunks.Add(chunk);

            try
            {
                Assert.That(chunks.Sum(c => c.RowCount), Is.EqualTo(25));
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
            }
        }
        finally
        {
            frame.Dispose();
        }
    }

    static string CreateCsvFile(string tempDir, int rowCount)
    {
        var csvPath = Path.Combine(tempDir, "data.csv");
        var lines = new List<string> { "Name,Age,Salary" };
        for (int i = 0; i < rowCount; i++)
            lines.Add($"P{i},{i % 100},{50000 + i}");
        File.WriteAllText(csvPath, string.Join("\n", lines));
        return csvPath;
    }

    static void AssertFrameValuesEqual(NivaraFrame actual, NivaraFrame expected)
    {
        Assert.That(actual.RowCount, Is.EqualTo(expected.RowCount));
        Assert.That(actual.ColumnNames, Is.EquivalentTo(expected.ColumnNames));
        foreach (var name in expected.ColumnNames)
        {
            var actualColumn = actual.GetColumn(name);
            var expectedColumn = expected.GetColumn(name);
            for (int i = 0; i < expected.RowCount; i++)
                Assert.That(actualColumn.GetValue(i), Is.EqualTo(expectedColumn.GetValue(i)));
        }
    }
}
