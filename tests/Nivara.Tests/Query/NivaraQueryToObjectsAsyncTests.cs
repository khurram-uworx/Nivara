using Nivara.IO;
using Nivara.Linq;
using NUnit.Framework;

namespace Nivara.Tests.Query;

/// <summary>
/// Tests for <see cref="NivaraQuery{T}.ToObjectsAsync"/> — streamed row projection (1.5):
/// constant-memory per-chunk row materialization instead of whole-frame <c>ToListAsync</c>.
/// </summary>
[TestFixture]
public class NivaraQueryToObjectsAsyncTests
{
    sealed class Person
    {
        public string Name { get; set; } = string.Empty;
        public int Age { get; set; }
        public double Salary { get; set; }
    }

    [Test]
    public async Task ToObjectsAsync_InMemoryFrame_MatchesToObjects()
    {
        using var frame = CreatePeopleFrame(5);

        var rows = new List<Person>();
        await foreach (var row in frame.Query<Person>().ToObjectsAsync())
            rows.Add(row);

        var expected = frame.Query<Person>().ToObjects();
        Assert.That(rows.Count, Is.EqualTo(expected.Count));
        for (int i = 0; i < expected.Count; i++)
        {
            Assert.That(rows[i].Name, Is.EqualTo(expected[i].Name), $"row {i} name");
            Assert.That(rows[i].Age, Is.EqualTo(expected[i].Age), $"row {i} age");
            Assert.That(rows[i].Salary, Is.EqualTo(expected[i].Salary), $"row {i} salary");
        }
    }

    [Test]
    public async Task ToObjectsAsync_ParquetChunkSource_StreamsRowsAcrossRowGroups()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraToObjectsAsync", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        var file = Path.Combine(tempDir, "people.parquet");
        try
        {
            using var frame = CreatePeopleFrame(5);
            // 3 row groups of sizes 2, 2, 1 — exercises per-chunk projection over the reused reader.
            NivaraParquetWriter.WriteParquet(frame, file, ParquetWriteOptions.Default.With(rowGroupSize: 2));

            var query = NivaraParquetReader.ScanQuery<Person>(file);
            try
            {
                var rows = new List<Person>();
                await foreach (var row in query.ToObjectsAsync(chunkSize: 2))
                    rows.Add(row);

                Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "P0", "P1", "P2", "P3", "P4" }));
                Assert.That(rows.Select(r => r.Age), Is.EqualTo(new[] { 30, 40, 50, 20, 35 }));
                Assert.That(rows.Select(r => r.Salary), Is.EqualTo(new[] { 75000.0, 80000.0, 90000.0, 60000.0, 85000.0 }));
            }
            finally
            {
                // The reused Parquet reader holds the file open until the owning frame is disposed.
                query.AsQueryFrame().Dispose();
            }
        }
        finally
        {
            if (File.Exists(file))
                File.Delete(file);
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task ToObjectsAsync_ParquetChunkSource_RespectsWhereFilter()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraToObjectsAsync", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        var file = Path.Combine(tempDir, "people.parquet");
        try
        {
            using var frame = CreatePeopleFrame(5);
            NivaraParquetWriter.WriteParquet(frame, file, ParquetWriteOptions.Default.With(rowGroupSize: 2));

            var query = NivaraParquetReader.ScanQuery<Person>(file).Where(p => p.Salary > 65000.0);
            try
            {
                var rows = new List<Person>();
                await foreach (var row in query.ToObjectsAsync(chunkSize: 2))
                    rows.Add(row);

                Assert.That(rows.Select(r => r.Name), Is.EqualTo(new[] { "P0", "P1", "P2", "P4" }));
            }
            finally
            {
                query.AsQueryFrame().Dispose();
            }
        }
        finally
        {
            if (File.Exists(file))
                File.Delete(file);
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void ToObjectsAsync_InvalidChunkSize_ThrowsArgumentOutOfRangeException()
    {
        using var frame = CreatePeopleFrame(3);
        var query = frame.Query<Person>();

        var ex = Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () =>
        {
            await foreach (var _ in query.ToObjectsAsync(chunkSize: 0))
            {
            }
        });
        Assert.That(ex!.ParamName, Is.EqualTo("chunkSize"));
    }

    [Test]
    public async Task ToObjectsAsync_CancelledToken_ThrowsOperationCanceledException()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), "NivaraToObjectsAsync", Guid.NewGuid().ToString());
        Directory.CreateDirectory(tempDir);
        var file = Path.Combine(tempDir, "people.parquet");
        try
        {
            using var frame = CreatePeopleFrame(5);
            NivaraParquetWriter.WriteParquet(frame, file, ParquetWriteOptions.Default.With(rowGroupSize: 2));

            var query = NivaraParquetReader.ScanQuery<Person>(file);
            try
            {
                using var cts = new CancellationTokenSource();
                cts.Cancel();

                async Task Enumerate()
                {
                    await foreach (var _ in query.ToObjectsAsync(ct: cts.Token))
                    {
                    }
                }

                Assert.ThrowsAsync<OperationCanceledException>(Enumerate);
            }
            finally
            {
                query.AsQueryFrame().Dispose();
            }
        }
        finally
        {
            if (File.Exists(file))
                File.Delete(file);
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, true);
        }
    }

    // ── Helpers ──

    static NivaraFrame CreatePeopleFrame(int count)
    {
        var names = Enumerable.Range(0, count).Select(i => $"P{i}").ToArray();
        var ages = new[] { 30, 40, 50, 20, 35, 45, 55, 25, 60, 38 };
        var salaries = new[] { 75000.0, 80000.0, 90000.0, 60000.0, 85000.0, 95000.0, 70000.0, 65000.0, 88000.0, 72000.0 };

        return NivaraFrame.Create(
            ("Name", NivaraColumn<string>.CreateForReferenceType(names)),
            ("Age", NivaraColumn<int>.Create(ages.Take(count).ToArray())),
            ("Salary", NivaraColumn<double>.Create(salaries.Take(count).ToArray())));
    }
}
