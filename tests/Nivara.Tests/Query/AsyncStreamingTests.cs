using Nivara.Exceptions;
using Nivara.Expressions;
using Nivara.IO;
using Nivara.Linq;
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

            Assert.Throws<OperationCanceledException>(() => queryFrame.CollectAsync(ct.Token));
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
            Assert.That(chunk3["Index"].Length, Is.EqualTo(0));

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

            Assert.That(result.RowCount, Is.EqualTo(1000));
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
}
