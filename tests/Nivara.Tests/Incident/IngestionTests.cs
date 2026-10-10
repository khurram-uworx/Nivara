using Nivara.Samples.Incident;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

[TestFixture]
public class IngestionTests
{
    string tempDir = null!;

    // GenerateFromRecordCount fills exactly the requested record count, distributing the 10,000 / 30
    // remainder across the leading minutes, so a 10,000-record request yields a 10,000-row file with
    // no trailing defaults. The generator's own row-count and distribution guarantees are pinned by
    // DatasetGeneratorTests.
    const int TotalRows = 10_000;
    const int RowGroupSize = 100;

    // Parquet chunks align to native row groups and the chunkSize argument is advisory, so the
    // chunk count is the row-group count: ceil(10,000 / 100) = 100.
    const int ExpectedChunks = (TotalRows + RowGroupSize - 1) / RowGroupSize;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-ingestion-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        DatasetGenerator.GenerateFromRecordCount(tempDir, "A", TotalRows, RowGroupSize);
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (tempDir is not null && Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public void LoadParquet_ReturnsCorrectRowCount()
    {
        using var qf = Ingestion.LoadParquet(Path.Combine(tempDir, "requests.parquet"));
        using var frame = qf.Collect();
        Assert.That(frame.RowCount, Is.EqualTo(TotalRows));
    }

    [Test]
    public void LoadCsv_ReturnsIdenticalData()
    {
        using var pqQf = Ingestion.LoadParquet(Path.Combine(tempDir, "requests.parquet"));
        using var pqFrame = pqQf.Collect();

        using var csvQf = Ingestion.LoadCsv(Path.Combine(tempDir, "requests.csv"));
        using var csvFrame = csvQf.Collect();

        Assert.That(csvFrame.RowCount, Is.EqualTo(pqFrame.RowCount));

        var pqServiceCol = pqFrame.GetColumn<string>("Service");
        var csvServiceCol = csvFrame.GetColumn<string>("Service");
        var pqStatusCol = pqFrame.GetColumn<int>("StatusCode");
        var csvStatusCol = csvFrame.GetColumn<int>("StatusCode");

        for (int i = 0; i < pqFrame.RowCount; i++)
        {
            Assert.That(csvServiceCol.GetValue(i), Is.EqualTo(pqServiceCol.GetValue(i)),
                $"Service mismatch at row {i}");
            Assert.That(csvStatusCol.GetValue(i), Is.EqualTo(pqStatusCol.GetValue(i)),
                $"StatusCode mismatch at row {i}");
        }
    }

    [Test]
    public async Task StreamChunks_YieldsRowGroupAlignedChunks()
    {
        int chunkCount = 0;
        long rowsStreamed = 0;
        await foreach (var chunk in Ingestion.StreamChunks(
            Path.Combine(tempDir, "requests.parquet"), RowGroupSize))
        {
            chunkCount++;
            rowsStreamed += chunk.RowCount;
            chunk.Dispose();
        }

        Assert.Multiple(() =>
        {
            Assert.That(chunkCount, Is.EqualTo(ExpectedChunks),
                "chunks align to parquet row groups, so the count is ceil(totalRows / rowGroupSize)");
            Assert.That(rowsStreamed, Is.EqualTo(TotalRows),
                "every request row should appear in exactly one chunk");
        });
    }

    [Test]
    public async Task StreamChunks_DisposesResources()
    {
        var chunks = new List<NivaraFrame>();
        await foreach (var chunk in Ingestion.StreamChunks(
            Path.Combine(tempDir, "requests.parquet"), RowGroupSize))
        {
            chunks.Add(chunk);
            if (chunks.Count >= 3) break;
        }

        foreach (var chunk in chunks)
            chunk.Dispose();

        Assert.That(chunks.Count, Is.EqualTo(3));
    }

    [Test]
    public async Task StreamChunks_CancellationStopsStream()
    {
        using var cts = new CancellationTokenSource();
        int chunkCount = 0;
        const int cancelAfter = 3;

        try
        {
            await foreach (var chunk in Ingestion.StreamChunks(
                Path.Combine(tempDir, "requests.parquet"), RowGroupSize, cts.Token))
            {
                chunkCount++;
                chunk.Dispose();
                if (chunkCount >= cancelAfter)
                    cts.Cancel();
            }
        }
        catch (OperationCanceledException)
        {
        }

        Assert.That(chunkCount, Is.EqualTo(cancelAfter));
    }
}
