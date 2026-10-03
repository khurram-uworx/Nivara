using Nivara.IO;
using Nivara.Linq;
using Nivara.Tests.IO;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Gates that the typed LINQ surface (<see cref="NivaraQuery{T}"/> and
/// <see cref="NivaraGroupedQuery{TKey, T}"/>) is disposable and that disposing it releases the file
/// handle a lazy source opened — issue #501.
/// <para>
/// Before #501 neither type implemented <see cref="IDisposable"/>, so a <c>ScanQuery&lt;T&gt;</c>
/// consumer had no <c>using</c> and had to reach through
/// <c>query.AsQueryFrame().Dispose()</c>. <see cref="NivaraGroupedQuery{TKey, T}"/> was worse: it
/// had no <c>AsQueryFrame()</c> at all, so <c>ScanQuery&lt;T&gt;(parquet).GroupBy(...)</c> had no
/// caller-side release path whatsoever.
/// </para>
/// <para>
/// The file-handle cases deliberately avoid a full <c>Collect()</c> on CSV and JSON:
/// <c>CsvLazySource</c> and <c>JsonLazySource</c> close their chunk reader unprompted at EOF, so a
/// full read releases the handle whether or not anything was disposed and such a case would pass
/// vacuously. Each case asserts the handle is <em>still open</em> at its midpoint before releasing
/// it. Parquet is the exception — <c>ParquetLazySource</c> holds its reused reader until
/// <c>Dispose</c>, so there a full <c>Collect()</c> is itself the load-bearing shape.
/// </para>
/// <para>
/// Handle <em>detection</em> is gated by
/// <see cref="ScanAsQueryFrameHandleTests.Probe_DetectsDeliberatelyLeakedHandle"/>; it cannot
/// reproduce #496's Windows <c>Directory.Delete</c> symptom, because Unix unlinks open files
/// unconditionally. See <see cref="FileHandleProbe"/>.
/// </para>
/// </summary>
[TestFixture]
public class NivaraQueryDisposalTests
{
    // Large enough that one chunk is nowhere near EOF, so a partial read is genuinely partial.
    const int RowCount = 10_000;
    const int PartialChunkSize = 100;

    string tempDir = null!;

    [SetUp]
    public void SetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), "NivaraQueryDisposalTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
    }

    [TearDown]
    public void TearDown()
    {
        if (Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public void NivaraQuery_AndGroupedQuery_ImplementDisposableAndAsyncDisposable()
    {
        // The type-shape gate: this is what makes "I forgot to dispose" visible at compile time.
        // Both interfaces are needed - a type implementing only IAsyncDisposable cannot be used with
        // `using` (CS8418), and one implementing only IDisposable cannot be used with `await using`
        // (CS8417), so `ScanQuery<T>` consumers need both forms to compile.
        Assert.Multiple(() =>
        {
            Assert.That(typeof(NivaraQuery<Person>).IsAssignableTo(typeof(IDisposable)), Is.True,
                "NivaraQuery<T> must be IDisposable so ScanQuery<T> can be used with `using`");
            Assert.That(typeof(NivaraQuery<Person>).IsAssignableTo(typeof(IAsyncDisposable)), Is.True,
                "NivaraQuery<T> must be IAsyncDisposable so ScanQuery<T> can be used with `await using`");
            Assert.That(typeof(NivaraGroupedQuery<string, Person>).IsAssignableTo(typeof(IDisposable)), Is.True,
                "NivaraGroupedQuery<TKey, T> must be IDisposable; it had no release path at all");
            Assert.That(typeof(NivaraGroupedQuery<string, Person>).IsAssignableTo(typeof(IAsyncDisposable)), Is.True,
                "NivaraGroupedQuery<TKey, T> must be IAsyncDisposable");
        });
    }

    [Test]
    public async Task CsvScanQuery_UsingDeclaration_ReleasesFileHandleAfterPartialRead()
    {
        var path = CreateCsvFile(RowCount);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before Csv.ScanQuery");

        using (var query = Csv.ScanQuery<Person>(path))
        {
            await ReadOneRowBatchThenStopAsync(query);

            FileHandleProbe.AssertLocked(path,
                "the read stopped short of EOF, so CsvLazySource's chunk reader must still be open");
        }

        FileHandleProbe.AssertUnlocked(path,
            "the `using` declaration on NivaraQuery<T> must release the file handle at scope exit");
    }

    [Test]
    public void ParquetGroupedQuery_Dispose_ReleasesFileHandle()
    {
        var path = CreateParquetFile(RowCount, rowGroupSize: 1000);

        FileHandleProbe.AssertUnlocked(path, "a handle was already open before NivaraParquetReader.ScanQuery");

        var grouped = NivaraParquetReader.ScanQuery<IndexRow>(path).GroupBy(row => row.Group);

        // Disposed in a finally so a genuine failure reports exactly once. Releasing only on the
        // happy path would leave the handle open, and the TearDown's Directory.Delete would then
        // fail on Windows - burying the real assertion under the misleading "being used by another
        // process" teardown error that #496 and #498 were about.
        try
        {
            using (var collected = grouped.Collect())
            {
                Assert.That(collected.RowCount, Is.EqualTo(GroupCount), "one row per distinct group key");
            }

            FileHandleProbe.AssertLocked(path,
                "ParquetLazySource holds its reader until Dispose, so even a full Collect leaves the handle open");
        }
        finally
        {
            grouped.Dispose();
        }

        FileHandleProbe.AssertUnlocked(path,
            "NivaraGroupedQuery<TKey, T> had no release path at all before #501, so a Parquet-backed "
            + "grouped query could never be released by the caller");
    }

    [Test]
    public async Task ParquetGroupedQuery_DisposeAsync_ReleasesFileHandle()
    {
        var path = CreateParquetFile(RowCount, rowGroupSize: 1000);

        var grouped = NivaraParquetReader.ScanQuery<IndexRow>(path).GroupBy(row => row.Group);

        try
        {
            using (var collected = grouped.Collect())
            {
                Assert.That(collected.RowCount, Is.EqualTo(GroupCount));
            }

            FileHandleProbe.AssertLocked(path, "ParquetLazySource holds its reader until Dispose");
        }
        finally
        {
            await grouped.DisposeAsync();
        }

        FileHandleProbe.AssertUnlocked(path, "DisposeAsync on NivaraGroupedQuery must release the handle too");
    }

    [Test]
    public void NivaraQuery_Dispose_DoesNotDisposeSourceFrameColumns()
    {
        var frame = CreatePersonFrame();

        var query = frame.Query<Person>();
        query.Dispose();

        Assert.Multiple(() =>
        {
            Assert.That(frame.RowCount, Is.EqualTo(3));
            Assert.That(frame.GetColumn<int>("Age")[0], Is.EqualTo(1));
            Assert.That(() => query.Collect(), Throws.TypeOf<ObjectDisposedException>());
        });
    }

    [Test]
    public async Task NivaraQuery_DisposeAsync_DoesNotDisposeSourceFrameColumns()
    {
        var frame = CreatePersonFrame();

        var query = frame.Query<Person>();
        await query.DisposeAsync();

        Assert.Multiple(() =>
        {
            Assert.That(frame.RowCount, Is.EqualTo(3));
            Assert.That(frame.GetColumn<int>("Age")[0], Is.EqualTo(1));
        });
    }

    [Test]
    public void NivaraQuery_Collect_AfterDispose_ThrowsObjectDisposedException()
    {
        var frame = CreatePersonFrame();

        var query = frame.Query<Person>();
        using (query.Collect())
        {
        }

        query.Dispose();

        Assert.Multiple(() =>
        {
            Assert.That(() => query.Collect(), Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => query.Where(p => p.Age > 1), Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => _ = query.IsLazy, Throws.TypeOf<ObjectDisposedException>());
        });
    }

    [Test]
    public void NivaraQuery_DisposingDerivedQuery_InvalidatesTheWholeChain()
    {
        // Documents the shared-source contract: one source, one release, like FileStream. A query
        // derived from another shares that source, so disposing either releases it for both and the
        // survivor then reports the release rather than quietly reading from a dead source.
        var frame = CreatePersonFrame();

        var query = frame.Query<Person>();
        var derived = query.Where(p => p.Age > 1);

        query.Dispose();

        // The derived frame's own guard now reports the shared release, so this throws a bare
        // ObjectDisposedException naming the QueryFrame rather than passing the guard and failing
        // deeper inside the source wrapped in QueryExecutionException.
        Assert.Multiple(() =>
        {
            Assert.That(() => derived.Collect(), Throws.TypeOf<ObjectDisposedException>(),
                "the survivor must report the shared release instead of reading from a dead source");
            Assert.That(() => derived.Where(p => p.Age > 2), Throws.TypeOf<ObjectDisposedException>());
            Assert.That(() => _ = derived.IsLazy, Throws.TypeOf<ObjectDisposedException>());
        });
    }

    /// <summary>
    /// Pulls exactly one batch of rows and abandons the enumeration, so the source is left part-way
    /// through the file with its chunk reader - and file handle - still open.
    /// </summary>
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
        public string Group { get; set; } = string.Empty;
    }

    sealed class Person
    {
        public int Age { get; set; }
    }

    const int GroupCount = 10;

    static NivaraFrame CreatePersonFrame()
        => NivaraFrame.Create("Age", NivaraColumn<int>.Create([1, 2, 3]));

    string CreateCsvFile(int rowCount)
    {
        var path = Path.Combine(tempDir, "people.csv");
        var lines = new List<string>(rowCount + 1) { "Age" };
        for (int i = 0; i < rowCount; i++)
            lines.Add($"{i % 100}");
        File.WriteAllText(path, string.Join("\n", lines));
        return path;
    }

    string CreateParquetFile(int rowCount, int rowGroupSize)
    {
        var path = Path.Combine(tempDir, "people.parquet");
        var indices = Enumerable.Range(0, rowCount).ToArray();
        var groups = Enumerable.Range(0, rowCount).Select(i => $"g{i % GroupCount}").ToArray();

        var frame = NivaraFrame.Create(
            ("Index", NivaraColumn<int>.Create(indices)),
            ("Group", NivaraColumn<string>.Create(groups)));
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