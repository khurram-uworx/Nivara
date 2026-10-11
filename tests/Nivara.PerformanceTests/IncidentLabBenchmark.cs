using Nivara.IO;
using Nivara.Samples.Incident;

namespace Nivara.PerformanceTests;

static class IncidentLabBenchmark
{
    static readonly string[] ScenarioIds = ["A", "B", "C", "D"];

    public static void RunDatasetGeneratorTests(string[] args)
    {
        var scale = args.Length > 1 && int.TryParse(args[1], out var s) ? s : 1;
        Console.WriteLine("DatasetGenerator Tests");
        Console.WriteLine($"  Scale: {scale}x");
        Console.WriteLine();

        TestDeterminism();
        Console.WriteLine("  PASS  Determinism");

        TestRowCount(scale);
        Console.WriteLine($"  PASS  RowCount (scale={scale})");

        TestFieldRanges();
        Console.WriteLine("  PASS  FieldRanges");

        TestParquetRowGroups();
        Console.WriteLine("  PASS  ParquetRowGroups");

        TestCsvVariant();
        Console.WriteLine("  PASS  CsvVariant");

        Console.WriteLine();
        Console.WriteLine("All DatasetGenerator tests passed.");
    }

    static void TestDeterminism()
    {
        var dir1 = Path.Combine(Path.GetTempPath(), $"inc-det-1-{Guid.NewGuid():N}");
        var dir2 = Path.Combine(Path.GetTempPath(), $"inc-det-2-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(dir1);
            Directory.CreateDirectory(dir2);
            Console.Write("    Generating run 1... ");
            DatasetGenerator.Generate(dir1, "A", 1);
            Console.Write("run 2... ");
            DatasetGenerator.Generate(dir2, "A", 1);
            Console.Write("comparing... ");

            var csv1 = File.ReadAllBytes(Path.Combine(dir1, "requests.csv"));
            var csv2 = File.ReadAllBytes(Path.Combine(dir2, "requests.csv"));
            if (csv1.Length != csv2.Length || !csv1.AsSpan().SequenceEqual(csv2))
                throw new InvalidOperationException("Determinism test failed: CSV files differ");
            Console.WriteLine("OK");
        }
        finally
        {
            if (Directory.Exists(dir1)) Directory.Delete(dir1, true);
            if (Directory.Exists(dir2)) Directory.Delete(dir2, true);
        }
    }

    static void TestRowCount(int scale)
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"inc-rc-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(tempDir);
            Console.Write("    Generating... ");
            DatasetGenerator.Generate(tempDir, "A", scale);
            Console.Write("loading... ");

            using var qf = NivaraParquetReader.ScanAsQueryFrame(Path.Combine(tempDir, "requests.parquet"));
            using var frame = qf.Collect();
            long expected = 10_000_000L * scale;
            if (frame.RowCount != expected)
                throw new InvalidOperationException(
                    $"RowCount test failed: expected {expected}, got {frame.RowCount}");
            Console.WriteLine($"{frame.RowCount:N0} rows OK");
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
        }
    }

    static void TestFieldRanges()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"inc-fr-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(tempDir);
            Console.Write("    Generating... ");
            DatasetGenerator.Generate(tempDir, "A", 1);
            Console.Write("loading... ");

            using var qf = NivaraParquetReader.ScanAsQueryFrame(Path.Combine(tempDir, "requests.parquet"));
            using var frame = qf.Collect();
            Console.Write($"scanning {frame.RowCount:N0} rows... ");

            var statusCol = frame.GetColumn<int>("StatusCode");
            var durationCol = frame.GetColumn<double>("DurationMs");
            var regionCol = frame.GetColumn<string>("Region");

            var regions = new HashSet<string>();
            for (int i = 0; i < frame.RowCount; i++)
            {
                int sc = (int)statusCol.GetValue(i)!;
                if (sc < 200 || sc > 503)
                    throw new InvalidOperationException(
                        $"FieldRanges test failed: StatusCode {sc} out of range [200,503]");

                double dur = (double)durationCol.GetValue(i)!;
                if (dur <= 0)
                    throw new InvalidOperationException(
                        $"FieldRanges test failed: DurationMs {dur} not > 0");

                regions.Add((string)regionCol.GetValue(i)!);

                if (i % 2_000_000 == 0 && i > 0)
                    Console.Write($"{i:N0}... ");
            }

            if (regions.Count != 10)
                throw new InvalidOperationException(
                    $"FieldRanges test failed: expected 10 regions, got {regions.Count}");
            Console.WriteLine($"{regions.Count} regions OK");
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
        }
    }

    static void TestParquetRowGroups()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"inc-rg-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(tempDir);
            Console.Write("    Generating 5K records with rowGroupSize=1000 (full dataset)... ");
            DatasetGenerator.GenerateFromRecordCount(tempDir, "A", 5_000, rowGroupSize: 1000);
            Console.Write("checking row groups... ");

            var parquetPath = Path.Combine(tempDir, "requests.parquet");
            using var stream = File.OpenRead(parquetPath);
            var parquetReader = Parquet.ParquetReader.CreateAsync(stream)
                .GetAwaiter().GetResult();
            try
            {
                if (parquetReader.RowGroupCount <= 1)
                    throw new InvalidOperationException(
                        $"ParquetRowGroups test failed: expected multiple row groups, got {parquetReader.RowGroupCount}");
                Console.WriteLine($"{parquetReader.RowGroupCount} row groups OK");
            }
            finally
            {
                parquetReader.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
        }
    }

    static void TestCsvVariant()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"inc-csv-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(tempDir);
            Console.Write("    Generating... ");
            DatasetGenerator.Generate(tempDir, "A", 1);
            Console.Write("loading CSV... ");

            using var qf = Csv.ScanAsQueryFrame(Path.Combine(tempDir, "requests.csv"));
            using var frame = qf.Collect();

            var expectedCols = new[] { "Timestamp", "Service", "Endpoint", "DurationMs", "StatusCode", "Region", "TraceId", "IsRetry" };
            if (frame.ColumnCount != expectedCols.Length)
                throw new InvalidOperationException(
                    $"CsvVariant test failed: expected {expectedCols.Length} columns, got {frame.ColumnCount}");

            foreach (var col in expectedCols)
            {
                if (!frame.ColumnNames.Contains(col))
                    throw new InvalidOperationException(
                        $"CsvVariant test failed: missing column '{col}'");
            }
            Console.WriteLine($"{frame.RowCount:N0} rows, {frame.ColumnCount} columns OK");
        }
        finally
        {
            if (Directory.Exists(tempDir)) Directory.Delete(tempDir, true);
        }
    }

}
