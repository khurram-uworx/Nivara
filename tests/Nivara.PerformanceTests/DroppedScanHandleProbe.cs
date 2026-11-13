using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Nivara.IO;
using Nivara.Linq;
using Nivara.Query;

namespace Nivara.PerformanceTests;

/// <summary>
/// On-demand probe for issue #1189: does a dropped (never-disposed) scan release its
/// file handle at garbage collection, and after how many collection passes?
/// </summary>
/// <remarks>
/// Run with
/// <c>dotnet run -c Release --project tests/Nivara.PerformanceTests -- --dropped-scan</c>.
/// After the fix (finalizers on <c>JsonLazySource</c> and <c>CsvLazySource</c>) every
/// source should release on the first collection pass.
/// </remarks>
internal static class DroppedScanHandleProbe
{
    const int RowCount = 10_000;
    const int PartialChunkSize = 100;

    static QueryFrame? s_root;

    public static int Run()
    {
        Console.WriteLine("=== Dropped scan handle release probe (#1189) ===");
        Console.WriteLine($"Runtime: {Environment.Version}  Platform: {RuntimeInformation.OSArchitecture}");
        if (IsDebugBuild())
            Console.WriteLine("WARNING: Debug build - numbers are not representative. Run with -c Release.");
        Console.WriteLine();

        // Negative control: can this probe see a held handle at all?
        var controlDir = CreateTempDir();
        var controlPath = Path.Combine(controlDir, "control.txt");
        File.WriteAllText(controlPath, "negative control");
        try
        {
            AssertUnlocked(controlPath, "nothing holds the file yet (negative control)");
            using (new FileStream(controlPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                AssertLocked(controlPath, "a FileShare.Read holder is open (negative control)");
            AssertUnlocked(controlPath, "the holder was disposed (negative control)");
            Console.WriteLine("negative control: OK (probe can observe a held handle)");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"negative control: FAILED - {ex.Message}");
            return 1;
        }
        finally
        {
            TryDeleteDir(controlDir);
        }
        Console.WriteLine();

        bool anyFailed = false;
        foreach (var source in new[] { "parquet", "csv", "json" })
        {
            var dir = CreateTempDir();
            string path = string.Empty;
            int releasedOnRound = 0;
            WeakReference? frameRef = null;
            WeakReference? sourceRef = null;
            try
            {
                path = CreateFile(source, dir, RowCount);
                AssertUnlocked(path, $"no handle open before scan ({source})");

                ProbeOneSource(source, path, out frameRef, out sourceRef);

                for (int i = 0; i < 5 && releasedOnRound == 0; i++)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                    if (IsUnlocked(path))
                    {
                        releasedOnRound = i + 1;
                        Console.WriteLine($"    round {i + 1}: unlocked=True frameAlive={frameRef?.IsAlive} sourceAlive={sourceRef?.IsAlive}");
                    }
                    else
                    {
                        Console.WriteLine($"    round {i + 1}: unlocked=False frameAlive={frameRef?.IsAlive} sourceAlive={sourceRef?.IsAlive}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{source,-8}: EXCEPTION {ex.GetType().Name}: {ex.Message}");
                anyFailed = true;
                releasedOnRound = -1;
            }
            finally
            {
                if (releasedOnRound > 0)
                    Console.WriteLine($"{source,-8}: released on GC pass {releasedOnRound}");
                else if (releasedOnRound == 0)
                {
                    Console.WriteLine($"{source,-8}: NOT released after 5 GC passes  -> FAIL");
                    anyFailed = true;
                }
                TryDeleteDir(dir);
            }
        }

        Console.WriteLine();
        Console.WriteLine(anyFailed
            ? "RESULT: at least one source did not release within 5 GC passes."
            : "RESULT: every source released within 5 GC passes.");
        return anyFailed ? 1 : 0;
    }

    static void ProbeOneSource(string source, string path, out WeakReference? frameRef, out WeakReference? sourceRef)
    {
        var frame = CreateScan(source, path);
        frameRef = new WeakReference(frame);

        object? src = null;
        try
        {
            var hf = typeof(QueryFrame).GetField("handle", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
            var h = hf.GetValue(frame)!;
            var sf = h.GetType().GetField("source", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!;
            src = sf.GetValue(h);
        }
        catch { /* best effort */ }
        sourceRef = src is null ? null : new WeakReference(src);

        ReadOneChunkThenStop(frame).GetAwaiter().GetResult();
        AssertLocked(path, $"partial read leaves the handle open ({source})");

        s_root = frame;
        frame = null;
        ClearRoot();
    }

    static string CreateTempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "NivaraDroppedScanHandleProbe", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    static void TryDeleteDir(string dir)
    {
        try { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
        catch { /* best effort */ }
    }

    static string CreateFile(string source, string dir, int rowCount)
    {
        return source switch
        {
            "parquet" => CreateParquetFile(dir, rowCount),
            "csv" => CreateCsvFile(dir, rowCount),
            "json" => CreateJsonFile(dir, rowCount),
            _ => throw new ArgumentException(source),
        };
    }

    static QueryFrame CreateScan(string source, string path)
    {
        return source switch
        {
            "parquet" => NivaraParquetReader.ScanAsQueryFrame(path),
            "csv" => Csv.ScanAsQueryFrame(path),
            "json" => Json.ScanAsQueryFrame(path),
            _ => throw new ArgumentException(source),
        };
    }

    static async Task ReadOneChunkThenStop(QueryFrame frame)
    {
        await foreach (var chunk in frame.AsStream(PartialChunkSize))
        {
            chunk.Dispose();
            break;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    static void ClearRoot() => s_root = null;

    static bool IsUnlocked(string path)
    {
        try { using var p = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None); return true; }
        catch (IOException) { return false; }
    }

    static void AssertUnlocked(string path, string because)
    {
        if (!IsUnlocked(path)) throw new InvalidOperationException($"'{Path.GetFileName(path)}' expected unlocked: {because}");
    }

    static void AssertLocked(string path, string because)
    {
        if (IsUnlocked(path)) throw new InvalidOperationException($"'{Path.GetFileName(path)}' expected locked: {because}");
    }

    static string CreateCsvFile(string dir, int rowCount)
    {
        var path = Path.Combine(dir, "people.csv");
        var lines = new List<string>(rowCount + 1) { "Name,Age,Salary" };
        for (int i = 0; i < rowCount; i++) lines.Add($"P{i},{i % 100},{50000 + i}");
        File.WriteAllText(path, string.Join("\n", lines));
        return path;
    }

    static string CreateJsonFile(string dir, int rowCount)
    {
        var path = Path.Combine(dir, "people.json");
        var records = new List<string>(rowCount);
        for (int i = 0; i < rowCount; i++) records.Add($"{{\"Name\": \"P{i}\", \"Age\": {i % 100}, \"Salary\": {50000 + i}}}");
        File.WriteAllText(path, "[" + string.Join(",", records) + "]");
        return path;
    }

    static string CreateParquetFile(string dir, int rowCount)
    {
        var path = Path.Combine(dir, "people.parquet");
        var indices = Enumerable.Range(0, rowCount).ToArray();
        var names = Enumerable.Range(0, rowCount).Select(i => $"P{i}").ToArray();
        var ages = Enumerable.Range(0, rowCount).Select(i => i % 100).ToArray();
        var salaries = Enumerable.Range(0, rowCount).Select(i => 50000 + i).ToArray();
        var frame = NivaraFrame.Create(
            ("Index", NivaraColumn<int>.Create(indices)),
            ("Name", NivaraColumn<string>.Create(names)),
            ("Age", NivaraColumn<int>.Create(ages)),
            ("Salary", NivaraColumn<int>.Create(salaries)));
        try { NivaraParquetWriter.WriteParquet(frame, path, ParquetWriteOptions.Default.With(rowGroupSize: 1000)); }
        finally { frame.Dispose(); }
        return path;
    }

    static bool IsDebugBuild()
    {
#if DEBUG
        return true;
#else
        return false;
#endif
    }
}
