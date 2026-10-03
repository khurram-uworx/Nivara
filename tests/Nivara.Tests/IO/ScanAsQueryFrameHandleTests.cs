using Nivara.IO;
using Nivara.Linq;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Gates that every <c>ScanFrame</c>-backed public entry point releases the file handle it
/// opens, on the CI platform.
/// <para>
/// Issue #496 fixed six <c>Analysis</c> entry points and gated them, but the gate lived in
/// <c>tests/Nivara.Tests/Incident/AnalysisResourceTests.cs</c> and covered only the Incident
/// sample. Issue #498 promotes that probe to <see cref="FileHandleProbe"/> and extends it to
/// every other consumer.
/// </para>
/// <para>
/// <b>Why the per-source shapes differ.</b> The three sources do not release on the same terms.
/// <c>ParquetLazySource</c> holds its reused reader — and so the file stream — until
/// <c>Dispose()</c>, so a full <c>Collect()</c> leaves the handle open and disposal is
/// load-bearing. <c>CsvLazySource</c> and <c>JsonLazySource</c> close their chunk reader
/// unprompted once a read reaches EOF, so a full <c>Collect()</c> releases the handle whether
/// or not the consumer disposed anything — a case written that way passes even with disposal
/// deleted outright. Those cases therefore break out of <c>AsStream</c> part-way through the
/// file and assert the handle is <em>still open</em> before releasing it. Without that
/// assertion they would look green while testing nothing.
/// </para>
/// <para>
/// <b>What a green run does and does not prove.</b> It proves the probe can see a leaked handle
/// on the platform it ran on (<see cref="Probe_DetectsDeliberatelyLeakedHandle"/> fails loudly
/// otherwise) and that these six entry points release. It does not prove the Windows
/// <c>Directory.Delete</c> teardown symptom #496 reported: CI is <c>ubuntu-latest</c>
/// (`.github/workflows/ci.yml`), and Unix unlinks open files unconditionally, so that symptom
/// cannot be reproduced here. Detection is covered; reproduction is not.
/// </para>
/// </summary>
[TestFixture]
public class ScanAsQueryFrameHandleTests
{
    // Large enough that one chunk is nowhere near EOF, so a partial read is genuinely partial.
    const int RowCount = 10_000;
    const int PartialChunkSize = 100;

    string tempDir = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "NivaraScanHandleTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    // ── Negative control ──

    /// <summary>
    /// Establishes that <see cref="FileHandleProbe"/> can actually observe a leaked handle on
    /// the platform this test is running on.
    /// <para>
    /// Deliberately fails rather than skipping. A probe that cannot see a leak makes every other
    /// case in this fixture pass for the wrong reason, and reading that as coverage is worse than
    /// having no gate: the Incident fixture's <c>File.Delete</c>-based assertions already had that
    /// property on CI, where unlinking an open file succeeds.
    /// </para>
    /// </summary>
    [Test]
    public void Probe_DetectsDeliberatelyLeakedHandle()
    {
        var path = Path.Combine(tempDir, "control.txt");
        File.WriteAllText(path, "negative control for the handle probe");

        FileHandleProbe.AssertUnlocked(path, "nothing holds this file yet");

        // The exact shape every lazy source opens its backing file with.
        using (new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            FileHandleProbe.AssertLocked(path, "a FileShare.Read holder is open");
        }

        FileHandleProbe.AssertUnlocked(path, "the holder was disposed");
    }

    // ── Parquet: disposal is load-bearing ──

    [Test]
    public void ParquetScanAsQueryFrame_ReleasesHandleOnDispose()
    {
        var path = CreateParquetFile(RowCount, rowGroupSize: 1000);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before NivaraParquetReader.ScanAsQueryFrame");

        var frame = NivaraParquetReader.ScanAsQueryFrame(path);
        using (var collected = frame.Collect())
        {
            Assert.That(collected.RowCount, Is.EqualTo(RowCount));
        }

        // ParquetLazySource keeps its reused reader — and so the file stream — until Dispose, so
        // unlike the row-oriented sources the handle is still open here and disposal is what
        // releases it.
        FileHandleProbe.AssertLocked(path,
            "ParquetLazySource holds its reader until Dispose, so a full Collect must leave the handle open");

        frame.Dispose();

        FileHandleProbe.AssertUnlocked(path, "NivaraParquetReader.ScanAsQueryFrame left the file handle open");
    }

    [Test]
    public void ParquetScanQuery_ReleasesHandleOnAsQueryFrameDispose()
    {
        var path = CreateParquetFile(RowCount, rowGroupSize: 1000);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before NivaraParquetReader.ScanQuery");

        var query = NivaraParquetReader.ScanQuery<IndexRow>(path);
        using (query.Collect())
        {
        }

        // NivaraQuery<T> does not implement IDisposable (issue #501), so AsQueryFrame() is the
        // only release path. Named here so a future red is not "fixed" by adding a `using` to a
        // type that cannot carry one.
        query.AsQueryFrame().Dispose();

        FileHandleProbe.AssertUnlocked(path, "NivaraParquetReader.ScanQuery left the file handle open");
    }

    // ── CSV: partial read, so disposal is the only thing that can release ──

    [Test]
    public async Task CsvScanAsQueryFrame_ReleasesHandleOnDisposeAfterPartialRead()
    {
        var path = CreateCsvFile(RowCount);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before Csv.ScanAsQueryFrame");

        var frame = Csv.ScanAsQueryFrame(path);
        await ReadOneChunkThenStopAsync(frame);

        FileHandleProbe.AssertLocked(path,
            "the read stopped short of EOF, so CsvLazySource's chunk reader must still be open");

        frame.Dispose();

        FileHandleProbe.AssertUnlocked(path, "Csv.ScanAsQueryFrame left the file handle open");
    }

    [Test]
    public async Task CsvScanQuery_ReleasesHandleOnAsQueryFrameDisposeAfterPartialRead()
    {
        var path = CreateCsvFile(RowCount);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before Csv.ScanQuery");

        var query = Csv.ScanQuery<Person>(path);
        await ReadOneRowBatchThenStopAsync(query);

        FileHandleProbe.AssertLocked(path,
            "the read stopped short of EOF, so CsvLazySource's chunk reader must still be open");

        // See ParquetScanQuery: NivaraQuery<T> is not IDisposable (issue #501).
        query.AsQueryFrame().Dispose();

        FileHandleProbe.AssertUnlocked(path, "Csv.ScanQuery left the file handle open");
    }

    // ── JSON: partial read, so disposal is the only thing that can release ──

    [Test]
    public async Task JsonScanAsQueryFrame_ReleasesHandleOnDisposeAfterPartialRead()
    {
        var path = CreateJsonFile(RowCount);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before Json.ScanAsQueryFrame");

        var frame = Json.ScanAsQueryFrame(path);
        await ReadOneChunkThenStopAsync(frame);

        FileHandleProbe.AssertLocked(path,
            "the read stopped short of EOF, so JsonLazySource's chunk reader must still be open");

        frame.Dispose();

        FileHandleProbe.AssertUnlocked(path, "Json.ScanAsQueryFrame left the file handle open");
    }

    [Test]
    public async Task JsonScanQuery_ReleasesHandleOnAsQueryFrameDisposeAfterPartialRead()
    {
        var path = CreateJsonFile(RowCount);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before Json.ScanQuery");

        var query = Json.ScanQuery<JsonPerson>(path);
        await ReadOneRowBatchThenStopAsync(query);

        FileHandleProbe.AssertLocked(path,
            "the read stopped short of EOF, so JsonLazySource's chunk reader must still be open");

        // See ParquetScanQuery: NivaraQuery<T> is not IDisposable (issue #501).
        query.AsQueryFrame().Dispose();

        FileHandleProbe.AssertUnlocked(path, "Json.ScanQuery left the file handle open");
    }

    // ── Helpers ──

    /// <summary>
    /// Pulls exactly one chunk and abandons the stream. The source is left part-way through the
    /// file, so the chunk reader — and its file handle — is still open when this returns.
    /// </summary>
    static async Task ReadOneChunkThenStopAsync(QueryFrame frame)
    {
        await foreach (var chunk in frame.AsStream(PartialChunkSize))
        {
            Assert.That(chunk.RowCount, Is.EqualTo(PartialChunkSize),
                "expected a full first chunk, so the stop point is unambiguously mid-file");
            chunk.Dispose();
            break;
        }
    }

    static async Task ReadOneRowBatchThenStopAsync<T>(NivaraQuery<T> query) where T : class, new()
    {
        int seen = 0;
        await foreach (var _ in query.ToObjectsAsync(PartialChunkSize))
        {
            if (++seen == PartialChunkSize)
                break;
        }

        Assert.That(seen, Is.EqualTo(PartialChunkSize), "expected a full first batch of rows");
    }

    sealed class IndexRow
    {
        public int Index { get; set; }
    }

    sealed class Person
    {
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }
        public int Salary { get; set; }
    }

    /// <summary>
    /// JSON schema inference widens every integer to <c>double</c> (the conservative
    /// int → double → string ladder), so the JSON entries need their own row type rather than
    /// reusing <see cref="Person"/>, whose int members fail row-type validation against it.
    /// </summary>
    sealed class JsonPerson
    {
        public string Name { get; set; } = string.Empty;
        public double Age { get; set; }
        public double Salary { get; set; }
    }

    string CreateCsvFile(int rowCount)
    {
        var path = Path.Combine(tempDir, "people.csv");
        var lines = new List<string>(rowCount + 1) { "Name,Age,Salary" };
        for (int i = 0; i < rowCount; i++)
            lines.Add($"P{i},{i % 100},{50000 + i}");
        File.WriteAllText(path, string.Join("\n", lines));
        return path;
    }

    string CreateJsonFile(int rowCount)
    {
        var path = Path.Combine(tempDir, "people.json");
        var records = new List<string>(rowCount);
        for (int i = 0; i < rowCount; i++)
            records.Add($"{{\"Name\": \"P{i}\", \"Age\": {i % 100}, \"Salary\": {50000 + i}}}");
        File.WriteAllText(path, "[" + string.Join(",", records) + "]");
        return path;
    }

    string CreateParquetFile(int rowCount, int rowGroupSize)
    {
        var path = Path.Combine(tempDir, "people.parquet");
        var indices = Enumerable.Range(0, rowCount).ToArray();
        var names = Enumerable.Range(0, rowCount).Select(i => $"P{i}").ToArray();
        var ages = Enumerable.Range(0, rowCount).Select(i => i % 100).ToArray();
        var salaries = Enumerable.Range(0, rowCount).Select(i => 50000 + i).ToArray();

        var frame = NivaraFrame.Create(
            ("Index", NivaraColumn<int>.Create(indices)),
            ("Name", NivaraColumn<string>.Create(names)),
            ("Age", NivaraColumn<int>.Create(ages)),
            ("Salary", NivaraColumn<int>.Create(salaries)));
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
}