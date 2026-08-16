using Nivara.IO;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Tests for the Parquet chunk-streaming rework (1.4): a single reused reader with metadata
/// parsed once, row-group-aligned chunks, true sync read paths, and one-frame-per-row-group
/// streaming.
/// </summary>
[TestFixture]
public class ParquetStreamingTests
{
    [Test]
    public void ParquetLazySource_ReadChunks_ReconstructFullData()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);

            using var source = new ParquetLazySource(file);
            var expected = source.Execute();

            var chunk0 = source.ReadChunk(0, 1000);
            var chunk1 = source.ReadChunk(1, 1000);
            var chunk2 = source.ReadChunk(2, 1000);

            Assert.That(chunk0["Index"].Length, Is.EqualTo(1000));
            Assert.That(chunk1["Index"].Length, Is.EqualTo(1000));
            Assert.That(chunk2["Index"].Length, Is.EqualTo(500));

            Assert.That(chunk0["Index"].GetValue(0), Is.EqualTo(expected["Index"].GetValue(0)));
            Assert.That(chunk0["Index"].GetValue(999), Is.EqualTo(expected["Index"].GetValue(999)));
            Assert.That(chunk1["Index"].GetValue(0), Is.EqualTo(expected["Index"].GetValue(1000)));
            Assert.That(chunk2["Index"].GetValue(499), Is.EqualTo(expected["Index"].GetValue(2499)));

            // Backward re-read after later chunks have been served must stay correct on the
            // reused reader.
            var chunk0Again = source.ReadChunk(0, 1000);
            Assert.That(chunk0Again["Index"].GetValue(999), Is.EqualTo(expected["Index"].GetValue(999)));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task ParquetLazySource_ReadChunkAsync_MatchesSyncAndOutOfRangeIsEmpty()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);
            using var source = new ParquetLazySource(file);

            var sync = source.ReadChunk(1, 1000);
            var async = await source.ReadChunkAsync(1, 1000);
            Assert.That(async["Index"].Length, Is.EqualTo(sync["Index"].Length));
            for (int i = 0; i < sync["Index"].Length; i++)
                Assert.That(async["Index"].GetValue(i), Is.EqualTo(sync["Index"].GetValue(i)));

            var outOfRange = await source.ReadChunkAsync(3, 1000);
            Assert.That(outOfRange, Is.Empty);

            var negative = Assert.Throws<ArgumentOutOfRangeException>(() => source.ReadChunk(-1, 1000));
            Assert.That(negative!.ParamName, Is.EqualTo("chunkIndex"));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task ParquetLazySource_ExecuteAsync_MatchesExecute()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);
            using var source = new ParquetLazySource(file);

            var sync = source.Execute();
            var async = await source.ExecuteAsync();

            Assert.That(async.Keys, Is.EquivalentTo(sync.Keys));
            for (int i = 0; i < sync["Index"].Length; i++)
                Assert.That(async["Index"].GetValue(i), Is.EqualTo(sync["Index"].GetValue(i)));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void ParquetLazySource_Metadata_ExposesSchemaAndRowCounts()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);
            using var source = new ParquetLazySource(file);

            Assert.That(source.RowGroupCount, Is.EqualTo(3));
            Assert.That(source.EstimatedRowCount, Is.EqualTo(2500));
            Assert.That(source.Schema.ColumnNames, Is.EquivalentTo(new[] { "Index" }));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task ParquetLazySource_AsStream_YieldsRowGroupAlignedChunks()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);
            using var queryFrame = NivaraParquetReader.ScanAsQueryFrame(file);

            var chunks = new List<NivaraFrame>();
            await foreach (var chunk in queryFrame.AsStream(chunkSize: 100))
                chunks.Add(chunk);

            try
            {
                // chunkSize is advisory for columnar sources; chunks align to row groups.
                Assert.That(chunks.Select(c => c.RowCount), Is.EqualTo(new[] { 1000, 1000, 500 }));
                Assert.That(chunks.Sum(c => c.RowCount), Is.EqualTo(2500));
                Assert.That(chunks[1].GetColumn<int>("Index")[0], Is.EqualTo(2000));
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
    public void ReadParquetStreaming_YieldsOneFramePerRowGroup()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);

            var chunks = NivaraParquetReader.ReadParquetStreaming(file).ToList();
            try
            {
                Assert.That(chunks.Select(c => c.RowCount), Is.EqualTo(new[] { 1000, 1000, 500 }));

                var merged = NivaraParquetWriter.ConcatenateFrames(chunks);
                try
                {
                    var whole = NivaraParquetReader.ReadParquet(file);
                    try
                    {
                        Assert.That(merged.RowCount, Is.EqualTo(whole.RowCount));
                        for (int i = 0; i < whole.RowCount; i++)
                            Assert.That(merged.GetColumn<int>("Index")[i], Is.EqualTo(whole.GetColumn<int>("Index")[i]));
                    }
                    finally
                    {
                        whole.Dispose();
                    }
                }
                finally
                {
                    merged.Dispose();
                }
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
    public void ReadParquetStreaming_EmptyFile_YieldsSingleEmptyFrame()
    {
        var tempDir = CreateTempDir();
        try
        {
            var emptyColumn = NivaraColumn<int>.Create(Array.Empty<int>());
            var frame = NivaraFrame.Create(("EmptyColumn", emptyColumn));
            var file = Path.Combine(tempDir, "empty.parquet");
            NivaraParquetWriter.WriteParquet(frame, file);

            var chunks = NivaraParquetReader.ReadParquetStreaming(file).ToList();
            try
            {
                Assert.That(chunks.Count, Is.EqualTo(1));
                Assert.That(chunks[0].RowCount, Is.EqualTo(0));
            }
            finally
            {
                foreach (var chunk in chunks)
                    chunk.Dispose();
                frame.Dispose();
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task ParquetLazySource_ConcurrentReadChunkAsync_ReturnsCorrectData()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 3000, rowGroupSize: 1000);
            using var source = new ParquetLazySource(file);
            var expected = source.Execute();

            // Fire concurrent chunk reads against the single reused reader; the internal
            // semaphore must serialize them without corrupting row-group seeks.
            var results = await Task.WhenAll(
                Enumerable.Range(0, 3).Select(i => source.ReadChunkAsync(i, 1000).AsTask()));

            for (int chunkIndex = 0; chunkIndex < results.Length; chunkIndex++)
            {
                var chunk = results[chunkIndex];
                var expectedStart = chunkIndex * 1000;
                Assert.That(chunk["Index"].Length, Is.EqualTo(1000));
                Assert.That(chunk["Index"].GetValue(0), Is.EqualTo(expected["Index"].GetValue(expectedStart)));
                Assert.That(chunk["Index"].GetValue(999), Is.EqualTo(expected["Index"].GetValue(expectedStart + 999)));
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void ParquetLazySource_Dispose_ReleasesFileHandle()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateParquetFile(tempDir, 2500, rowGroupSize: 1000);
            using (var source = new ParquetLazySource(file))
            {
                var result = source.Execute();
                Assert.That(result["Index"].Length, Is.EqualTo(2500));
            }

            // The reused reader owns a single file stream; disposing the source must release it
            // so the file can be deleted on Windows.
            File.Delete(file);
            Assert.That(File.Exists(file), Is.False);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    // ── Helpers ──

    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "NivaraParquetStreamingTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Writes a Parquet file with rows <c>i * 2</c> for <c>i</c> in [0, <paramref name="count"/>),
    /// split into <paramref name="rowGroupSize"/>-row groups.
    /// </summary>
    private static string CreateParquetFile(string tempDir, int count, int rowGroupSize)
    {
        var file = Path.Combine(tempDir, "data.parquet");
        var values = Enumerable.Range(0, count).Select(i => i * 2).ToArray();
        var frame = NivaraFrame.Create(("Index", NivaraColumn<int>.Create(values)));
        try
        {
            NivaraParquetWriter.WriteParquet(frame, file, ParquetWriteOptions.Default.With(rowGroupSize: rowGroupSize));
        }
        finally
        {
            frame.Dispose();
        }
        return file;
    }
}
