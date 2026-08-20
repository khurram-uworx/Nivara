using Nivara.Expressions;
using Nivara.IO;
using Nivara.Streamix;
using NUnit.Framework;
using Streamix;

namespace Nivara.Tests.Streamix;

[TestFixture]
public class StreamixBridgeIntegrationTests
{
    string tempDir = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "NivaraStreamixIntegrationTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public async Task ToFluxCsvSource_ProducesMultipleChunks()
    {
        var csvPath = CreateCsvFile(tempDir, 50);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);

        var flux = queryFrame.ToFlux(chunkSize: 10);
        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in flux)
            chunks.Add(chunk);

        try
        {
            Assert.That(chunks.Count, Is.GreaterThan(1), "Should produce multiple chunks from CSV source");

            int totalRows = chunks.Sum(c => c.RowCount);
            Assert.That(totalRows, Is.EqualTo(50));

            for (int i = 0; i < 50; i++)
            {
                int chunkIndex = i / 10;
                int offset = i % 10;
                Assert.That(chunks[chunkIndex].GetColumn<int>("Id")[offset], Is.EqualTo(i));
                Assert.That(chunks[chunkIndex].GetColumn<string>("Name")[offset], Is.EqualTo($"item{i}"));
                Assert.That(chunks[chunkIndex].GetColumn<double>("Value")[offset], Is.EqualTo(i * 1.5).Within(0.001));
            }
        }
        finally
        {
            foreach (var chunk in chunks)
                chunk.Dispose();
        }
    }

    [Test]
    public async Task ToFluxParquetSource_ProducesRowGroupAlignedChunks()
    {
        var parquetPath = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);
        using var queryFrame = NivaraParquetReader.ScanAsQueryFrame(parquetPath);

        var flux = queryFrame.ToFlux(chunkSize: 100);
        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in flux)
            chunks.Add(chunk);

        try
        {
            Assert.That(chunks.Count, Is.EqualTo(3));
            Assert.That(chunks.Select(c => c.RowCount).ToArray(), Is.EqualTo(new[] { 1000, 1000, 500 }));
            Assert.That(chunks.Sum(c => c.RowCount), Is.EqualTo(2500));

            Assert.That(chunks[0].GetColumn<int>("Id")[0], Is.EqualTo(0));
            Assert.That(chunks[1].GetColumn<int>("Id")[0], Is.EqualTo(1000));
            Assert.That(chunks[2].GetColumn<int>("Id")[0], Is.EqualTo(2000));
        }
        finally
        {
            foreach (var chunk in chunks)
                chunk.Dispose();
        }
    }

    [Test]
    public async Task ToFluxCsvSource_WithFilter_MatchesCollect()
    {
        var csvPath = CreateCsvFile(tempDir, 30);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);

        var filtered = queryFrame.Filter(ColumnExpressions.Col("Id") > 10);
        using var expected = filtered.Collect();

        var flux = filtered.ToFlux(chunkSize: 5);
        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in flux)
            chunks.Add(chunk);

        try
        {
            int totalRows = chunks.Sum(c => c.RowCount);
            Assert.That(totalRows, Is.EqualTo(expected.RowCount));

            int offset = 0;
            foreach (var chunk in chunks)
            {
                for (int i = 0; i < chunk.RowCount; i++)
                {
                    Assert.That(chunk.GetColumn<int>("Id")[i], Is.EqualTo(expected.GetColumn<int>("Id")[offset]));
                    Assert.That(chunk.GetColumn<double>("Value")[i], Is.EqualTo(expected.GetColumn<double>("Value")[offset]).Within(0.001));
                    offset++;
                }
            }
        }
        finally
        {
            foreach (var chunk in chunks)
                chunk.Dispose();
        }
    }

    [Test]
    public async Task BackpressureWaitMode_RealCsvSource_CompletesEndToEnd()
    {
        var csvPath = CreateCsvFile(tempDir, 25);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);

        var flux = queryFrame.ToFlux(
            chunkSize: 5,
            backpressureMode: ChannelBackpressureMode.Wait,
            channelCapacity: 2);

        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in flux)
            chunks.Add(chunk);

        try
        {
            Assert.That(chunks.Count, Is.EqualTo(5));
            int totalRows = chunks.Sum(c => c.RowCount);
            Assert.That(totalRows, Is.EqualTo(25));

            int globalIndex = 0;
            foreach (var chunk in chunks)
            {
                for (int i = 0; i < chunk.RowCount; i++)
                {
                    Assert.That(chunk.GetColumn<int>("Id")[i], Is.EqualTo(globalIndex));
                    globalIndex++;
                }
            }
        }
        finally
        {
            foreach (var chunk in chunks)
                chunk.Dispose();
        }
    }

    [Test]
    public async Task Cancellation_PropagatesThroughRealCsvSource()
    {
        var csvPath = CreateCsvFile(tempDir, 500);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);

        using var cts = new CancellationTokenSource();
        var fluxRows = queryFrame.ToFluxRows(chunkSize: 10, cts.Token);

        int rowsConsumed = 0;
        OperationCanceledException? caught = null;
        try
        {
            await foreach (var row in fluxRows)
            {
                rowsConsumed++;
                if (rowsConsumed >= 5)
                    cts.Cancel();
            }
        }
        catch (OperationCanceledException ex)
        {
            caught = ex;
        }

        Assert.That(caught, Is.Not.Null, "OperationCanceledException should propagate through bridge with real source");
        Assert.That(rowsConsumed, Is.LessThanOrEqualTo(15),
            "Iteration should stop shortly after cancellation");
    }

    [Test]
    public async Task ToFluxRowsCsvSource_YieldsAllRows()
    {
        var csvPath = CreateCsvFile(tempDir, 20);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);

        var fluxRows = queryFrame.ToFluxRows(chunkSize: 5);

        int globalIndex = 0;
        await foreach (var row in fluxRows)
        {
            Assert.That(row.GetValue<int>("Id"), Is.EqualTo(globalIndex));
            Assert.That(row.GetValue<string>("Name"), Is.EqualTo($"item{globalIndex}"));
            Assert.That(row.GetValue<double>("Value"), Is.EqualTo(globalIndex * 1.5).Within(0.001));
            globalIndex++;
        }

        Assert.That(globalIndex, Is.EqualTo(20));
    }

    [Test]
    public async Task ToFluxCsvSource_ToNivaraFrameAsync_RoundTrips()
    {
        var csvPath = CreateCsvFile(tempDir, 15);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);
        using var expected = queryFrame.Collect();

        using var queryFrame2 = Csv.ScanAsQueryFrame(csvPath);
        var flux = queryFrame2.ToFlux(chunkSize: 5);
        using var result = await flux.ToNivaraFrameAsync();

        Assert.That(result.RowCount, Is.EqualTo(expected.RowCount));
        Assert.That(result.ColumnNames, Is.EquivalentTo(expected.ColumnNames));

        for (int i = 0; i < expected.RowCount; i++)
        {
            Assert.That(result.GetColumn<int>("Id")[i], Is.EqualTo(expected.GetColumn<int>("Id")[i]));
            Assert.That(result.GetColumn<string>("Name")[i], Is.EqualTo(expected.GetColumn<string>("Name")[i]));
            Assert.That(result.GetColumn<double>("Value")[i], Is.EqualTo(expected.GetColumn<double>("Value")[i]).Within(0.001));
        }
    }

    static string CreateCsvFile(string dir, int rowCount)
    {
        var path = Path.Combine(dir, $"data_{rowCount}.csv");
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("Id,Name,Value");
        for (int i = 0; i < rowCount; i++)
            sb.AppendLine($"{i},item{i},{i * 1.5}");
        File.WriteAllText(path, sb.ToString());
        return path;
    }

    static string CreateParquetFile(string dir, int rowCount, int rowGroupSize)
    {
        var path = Path.Combine(dir, $"data_{rowCount}.parquet");
        var ids = Enumerable.Range(0, rowCount).ToArray();
        var names = Enumerable.Range(0, rowCount).Select(i => $"item{i}").ToArray();
        var values = Enumerable.Range(0, rowCount).Select(i => i * 1.5).ToArray();
        var frame = NivaraFrame.Create(
            ("Id", NivaraColumn<int>.Create(ids)),
            ("Name", NivaraColumn<string>.Create(names)),
            ("Value", NivaraColumn<double>.Create(values)));
        try
        {
            NivaraParquetWriter.WriteParquet(frame, path, ParquetWriteOptions.Default.With(rowGroupSize: rowGroupSize));
        }
        finally
        {
            frame.Dispose();
        }
        return path;
    }

    [Test]
    public async Task ToFluxRows_ToNivaraFrameAsync_CsvRoundTrips()
    {
        var csvPath = CreateCsvFile(tempDir, 20);
        using var expected = Csv.ScanAsQueryFrame(csvPath).Collect();

        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);
        var fluxRows = queryFrame.ToFluxRows(chunkSize: 5);
        using var result = await fluxRows.ToNivaraFrameAsync();

        Assert.That(result.RowCount, Is.EqualTo(expected.RowCount));
        Assert.That(result.ColumnNames, Is.EquivalentTo(expected.ColumnNames));

        for (int i = 0; i < expected.RowCount; i++)
        {
            Assert.That(result.GetColumn<int>("Id")[i], Is.EqualTo(expected.GetColumn<int>("Id")[i]));
            Assert.That(result.GetColumn<string>("Name")[i], Is.EqualTo(expected.GetColumn<string>("Name")[i]));
            Assert.That(result.GetColumn<double>("Value")[i], Is.EqualTo(expected.GetColumn<double>("Value")[i]).Within(0.001));
        }
    }

    [Test]
    public async Task BufferFrames_CsvSource_ProducesCorrectBatches()
    {
        var csvPath = CreateCsvFile(tempDir, 25);
        using var queryFrame = Csv.ScanAsQueryFrame(csvPath);

        var fluxRows = queryFrame.ToFluxRows(chunkSize: 5);
        var batched = fluxRows.BufferFrames(batchSize: 10);

        var frames = new List<NivaraFrame>();
        await foreach (var f in batched)
            frames.Add(f);

        try
        {
            Assert.That(frames, Has.Count.EqualTo(3));
            Assert.That(frames[0].RowCount, Is.EqualTo(10));
            Assert.That(frames[1].RowCount, Is.EqualTo(10));
            Assert.That(frames[2].RowCount, Is.EqualTo(5));

            int totalRows = frames.Sum(f => f.RowCount);
            Assert.That(totalRows, Is.EqualTo(25));
        }
        finally
        {
            foreach (var f in frames)
                f.Dispose();
        }
    }
}
