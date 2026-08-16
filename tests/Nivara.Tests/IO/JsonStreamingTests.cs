using System.Buffers;
using System.Text;
using System.Text.Json;
using Nivara.Exceptions;
using Nivara.Execution;
using Nivara.Expressions;
using Nivara.IO;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Tests for the true streaming JSON reader (<see cref="JsonRecordStreamReader"/>) and
/// the streaming rework of <see cref="JsonLazySource"/>/<see cref="JsonEagerSource"/>
/// (issue #265).
/// </summary>
[TestFixture]
public class JsonStreamingTests
{
    // ── Walker (JsonRecordStreamReader) ──

    [Test]
    public void JsonRecordStreamReader_ChunkWalk_RecordsSpanningBufferRefills_PreservesOrder()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateJsonFile(tempDir, 6, recordSize: 200_000);
            using var reader = new JsonRecordStreamReader(file, default);

            string[] collected = Array.Empty<string>();
            for (int i = 0; i < 6; i++)
            {
                var range = reader.LocateRange(i, 1, isArray: true);
                Assert.That(range.Rows, Is.EqualTo(1), $"chunk {i} rows");
                Assert.That(range.Eof, Is.EqualTo(i == 5), $"chunk {i} eof");
                var records = ReadRecords(reader, range);
                Assert.That(records.Length, Is.EqualTo(1), $"chunk {i} count");
                collected = collected.Append(records[0].GetProperty("name").GetString()!).ToArray();
            }

            Assert.That(string.Join(",", collected), Is.EqualTo("R0,R1,R2,R3,R4,R5"));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonRecordStreamReader_BackwardAccess_AfterEof_RewalksCorrectly()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateJsonFile(tempDir, 100);
            using var reader = new JsonRecordStreamReader(file, default);

            int total = 0;
            while (true)
            {
                var range = reader.LocateRange(total, 10, isArray: true);
                if (range.Rows == 0)
                    break;
                total += range.Rows;
            }
            Assert.That(total, Is.EqualTo(100));

            var back = reader.LocateRange(0, 10, isArray: true);
            var backRecords = ReadRecords(reader, back);
            Assert.That(backRecords.Length, Is.EqualTo(10));
            Assert.That(backRecords[0].GetProperty("name").GetString(), Is.EqualTo("R0"));
            Assert.That(backRecords[9].GetProperty("name").GetString(), Is.EqualTo("R9"));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonRecordStreamReader_Utf8Bom_Skipped()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = Path.Combine(tempDir, "bom.json");
            var bom = new byte[] { 0xEF, 0xBB, 0xBF };
            File.WriteAllBytes(file, bom.Concat(Encoding.UTF8.GetBytes("[{\"a\": 1}]")).ToArray());
            using var reader = new JsonRecordStreamReader(file, default);

            var range = reader.LocateRange(0, 10, isArray: true);
            var records = ReadRecords(reader, range);
            Assert.That(records.Length, Is.EqualTo(1));
            Assert.That(records[0].GetProperty("a").GetInt32(), Is.EqualTo(1));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonRecordStreamReader_CommentsAndTrailingCommas_RespectReaderOptions()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = Path.Combine(tempDir, "comments.json");
            File.WriteAllText(file, "[\n  // leading\n  {\"a\": 1},\n  {\"a\": 2}, // trailing\n]");
            var options = new JsonReaderOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };
            using var reader = new JsonRecordStreamReader(file, options);

            var range = reader.LocateRange(0, 10, isArray: true);
            Assert.That(range.Rows, Is.EqualTo(2));
            var records = ReadRecords(reader, range);
            Assert.That(records[1].GetProperty("a").GetInt32(), Is.EqualTo(2));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonRecordStreamReader_WhitespaceOnly_ReportsNoTokens()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = Path.Combine(tempDir, "empty.json");
            File.WriteAllText(file, "   \r\n  ");
            using var reader = new JsonRecordStreamReader(file, default);

            var range = reader.LocateRange(0, 10, isArray: true);
            Assert.That(range.Rows, Is.EqualTo(0));
            Assert.That(range.Eof, Is.True);
            Assert.That(reader.SawAnyToken, Is.False);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonRecordStreamReader_NonArray_SingleRecordRange()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = Path.Combine(tempDir, "single.json");
            File.WriteAllText(file, "{\"name\": \"solo\", \"value\": 4}");
            using var reader = new JsonRecordStreamReader(file, default);

            var range = reader.LocateRange(0, 10, isArray: false);
            Assert.That(range.Rows, Is.EqualTo(1));
            Assert.That(range.Eof, Is.True);
            var records = ReadRecords(reader, range);
            Assert.That(records[0].GetProperty("value").GetInt32(), Is.EqualTo(4));

            var past = reader.LocateRange(1, 10, isArray: false);
            Assert.That(past.Rows, Is.EqualTo(0));
            Assert.That(past.Eof, Is.True);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    // ── JsonLazySource chunking ──

    [Test]
    public void JsonLazySource_ReadChunk_ReconstructsFullData()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateJsonFile(tempDir, 2500);
            using var source = new JsonLazySource(file, JsonOptions.Default);

            var expectedValue = source.Execute()["value"];

            var chunk0 = source.ReadChunk(0, 1000);
            var chunk1 = source.ReadChunk(1, 1000);
            var chunk2 = source.ReadChunk(2, 1000);

            Assert.That(chunk0["value"].Length, Is.EqualTo(1000));
            Assert.That(chunk1["value"].Length, Is.EqualTo(1000));
            Assert.That(chunk2["value"].Length, Is.EqualTo(500));

            Assert.That(chunk0["value"].GetValue(0), Is.EqualTo(expectedValue.GetValue(0)));
            Assert.That(chunk1["value"].GetValue(0), Is.EqualTo(expectedValue.GetValue(1000)));
            Assert.That(chunk2["value"].GetValue(499), Is.EqualTo(expectedValue.GetValue(2499)));

            // Backward re-read after the reader has reached EOF must reopen and stay correct.
            var chunk0Again = source.ReadChunk(0, 1000);
            Assert.That(chunk0Again["value"].GetValue(999), Is.EqualTo(expectedValue.GetValue(999)));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public async Task JsonLazySource_ReadChunkAsync_MatchesSyncAndOutOfRangeIsEmpty()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateJsonFile(tempDir, 2500);
            using var source = new JsonLazySource(file, JsonOptions.Default);

            var sync = source.ReadChunk(0, 1000);
            var async = await source.ReadChunkAsync(0, 1000);
            Assert.That(async["value"].Length, Is.EqualTo(sync["value"].Length));
            for (int i = 0; i < sync["value"].Length; i++)
                Assert.That(async["value"].GetValue(i), Is.EqualTo(sync["value"].GetValue(i)));

            var outOfRange = await source.ReadChunkAsync(3, 1000);
            Assert.That(outOfRange, Is.Empty);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonLazySource_SchemaInference_ReadsSampleOnly()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = Path.Combine(tempDir, "malformed_late.json");
            var sb = new StringBuilder("[");
            for (int i = 0; i < 9; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append($"{{\"i\": {i}}}");
            }
            sb.Append(",{\"broken\": ]"); // record index 9 is malformed
            File.WriteAllText(file, sb.ToString());

            var options = JsonOptions.Default.With(schemaInferenceRecords: 5);
            using var source = new JsonLazySource(file, options);

            // Schema inference reads only the first 5 records, so it succeeds.
            Assert.That(source.Schema.ColumnNames, Is.EquivalentTo(new[] { "i" }));

            // Full execution walks past record 5 and hits the malformed record.
            Assert.Throws<DataSourceException>(() => source.Execute());
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonLazySource_FileHandleReleased_AfterStreamingToEnd()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateJsonFile(tempDir, 1000);
            using (var source = new JsonLazySource(file, JsonOptions.Default))
            {
                var result = source.Execute();
                Assert.That(result["name"].Length, Is.EqualTo(1000));
            }

            // The persistent file handle is closed once the source streamed to EOF, so the
            // file can be deleted (Windows fails on open handles).
            File.Delete(file);
            Assert.That(File.Exists(file), Is.False);
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Test]
    public void JsonLazySource_IsArrayFalse_SingleRecordChunking()
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = Path.Combine(tempDir, "single_object.json");
            File.WriteAllText(file, "{\"name\": \"Alice\", \"age\": 30}");
            var options = JsonOptions.Default.With(isArray: false);

            using var source = new JsonLazySource(file, options);
            Assert.That(source.Schema.ColumnNames, Is.EquivalentTo(new[] { "name", "age" }));

            var result = source.Execute();
            Assert.That(result["name"].Length, Is.EqualTo(1));
            Assert.That(result["name"].GetValue(0), Is.EqualTo("Alice"));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    // ── Execution strategy parity over JSON ──

    [Test]
    public void JsonSource_StreamingStrategy_ParityWithLazy()
    {
        AssertStrategyParity(ExecutionStrategy.Streaming, streamingBudget: 1024 * 1024, parallelDegree: null);
    }

    [Test]
    public void JsonSource_ParallelStrategy_ParityWithLazy()
    {
        AssertStrategyParity(ExecutionStrategy.Parallel, streamingBudget: null, parallelDegree: 2);
    }

    private static void AssertStrategyParity(ExecutionStrategy strategy, int? streamingBudget, int? parallelDegree)
    {
        var tempDir = CreateTempDir();
        try
        {
            var file = CreateJsonFile(tempDir, 250);
            var plan = Json.ScanFrame(file)
                .Filter(ColumnExpressions.Col("value") > 100.0)
                .Select(ColumnExpressions.Col("i"), ColumnExpressions.Col("name"))
                .ToQueryPlan();

            var engine = new ExecutionEngine();
            using var lazy = engine.Execute(plan, new NivaraExecutionContext(ExecutionStrategy.Lazy));

            var context = new NivaraExecutionContext(strategy);
            if (streamingBudget is not null)
                context.MemoryBudget = streamingBudget.Value;
            if (parallelDegree is not null)
                context.MaxDegreeOfParallelism = parallelDegree.Value;
            using var actual = engine.Execute(plan, context);

            Assert.That(actual.RowCount, Is.EqualTo(lazy.RowCount), strategy.ToString());
            Assert.That(actual.ColumnNames, Is.EquivalentTo(lazy.ColumnNames), strategy.ToString());
            for (int i = 0; i < lazy.RowCount; i++)
            {
                Assert.That(actual.GetColumn("i").GetValue(i), Is.EqualTo(lazy.GetColumn("i").GetValue(i)), $"{strategy} row {i} i");
                Assert.That(actual.GetColumn("name").GetValue(i), Is.EqualTo(lazy.GetColumn("name").GetValue(i)), $"{strategy} row {i} name");
            }
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    // ── Helpers ──

    private static JsonElement[] ReadRecords(JsonRecordStreamReader reader, JsonRecordRange range)
    {
        int length = checked((int)(range.End - range.Start));
        var rented = ArrayPool<byte>.Shared.Rent(length + 2);
        try
        {
            int read = reader.ReadRange(range.Start, range.End, rented);
            rented[0] = (byte)'[';
            rented[read + 1] = (byte)']';
            using var document = JsonDocument.Parse(
                rented.AsMemory(0, read + 2),
                new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip });
            var records = new List<JsonElement>(range.Rows);
            foreach (var element in document.RootElement.EnumerateArray())
                records.Add(element.Clone());
            return records.ToArray();
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(rented);
        }
    }

    private static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "NivaraJsonStreamingTests", Guid.NewGuid().ToString());
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>
    /// Writes a JSON array of <paramref name="count"/> object records
    /// <c>{"i": n, "name": "Rn", "value": n * 1.5}</c>, optionally padded with a
    /// <paramref name="recordSize"/>-byte string so records span the 64 KB initial buffer.
    /// </summary>
    private static string CreateJsonFile(string tempDir, int count, int recordSize = 0)
    {
        var file = Path.Combine(tempDir, "data.json");
        var sb = new StringBuilder("[");
        var padding = recordSize > 0 ? new string('x', Math.Max(0, recordSize - 64)) : null;
        for (int i = 0; i < count; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append($"{{\"i\": {i}, \"name\": \"R{i}\"");
            if (padding is not null)
                sb.Append($", \"pad\": \"{padding}\"");
            sb.Append($", \"value\": {i * 1.5}}}");
        }
        sb.Append(']');
        File.WriteAllText(file, sb.ToString());
        return file;
    }
}
