using Apache.Arrow;
using Apache.Arrow.Types;

namespace Nivara.SimdProbe;

/// <summary>
/// Probe: which Apache.Arrow table/column construction route preserves <b>per-column</b> chunk
/// counts, and which one only gives every column the same chunk grid?
///
/// <para><b>Question.</b> Issue #862 wants opt-in chunk preservation through an Arrow round-trip.
/// <c>Table.TableFromRecordBatches</c> is the obvious route, but it builds one <c>Column</c> per
/// field from one array per batch, so every column necessarily shares a single chunk grid. The
/// plan needs two columns with <i>different</i> chunk counts to survive, which is only expressible
/// through <c>new Table(Schema, IList&lt;Column&gt;)</c> with each <c>Column</c> built from its own
/// array list.</para>
///
/// <para><b>Why this is a probe and not an assumption.</b> The constructors exist, but whether
/// <c>Table</c> validates per-column chunk counts, recomputes <c>RowCount</c>, or throws is not
/// stated anywhere the plan could cite. An out-of-tree <c>Add-Type</c> attempt against these
/// members failed on a <c>System.Runtime</c> 8-vs-10 identity mismatch, so the check has to run
/// against the in-tree reference. This mode is the gate: if case 2 fails, the export design falls
/// back to the batched grid and must be re-raised before any IO code is written.</para>
///
/// <para><b>Also asserts.</b> That the batched route really does collapse to one grid (case 1),
/// that a two-chunk <c>Column</c> round-trips (case 3), and that a bare <c>ChunkedArray</c>
/// reports <c>ArrayCount</c>/<c>Length</c> (case 4).</para>
/// </summary>
internal static class ChunkedArrowProbe
{
    static readonly Field FieldA = new("a", Int32Type.Default, nullable: true);
    static readonly Field FieldB = new("b", Int32Type.Default, nullable: true);

    public static int Run()
    {
        Console.WriteLine("=== Apache.Arrow chunk-grid construction (#862) ===");
        Console.WriteLine($"Runtime: {Environment.Version}   Apache.Arrow: {typeof(Table).Assembly.GetName().Version}");
        Console.WriteLine();

        int failures = 0;

        failures += CheckBatchedGridIsShared();
        failures += CheckPerColumnChunkCounts();
        failures += CheckColumnRoundTrip();
        failures += CheckChunkedArray();

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? "All chunked-arrow checks PASSED."
            : $"{failures} chunked-arrow check(s) FAILED.");

        return failures;
    }

    static int CheckBatchedGridIsShared()
    {
        var batches = new List<RecordBatch>();
        int offset = 0;
        for (int batchIndex = 0; batchIndex < 3; batchIndex++)
        {
            var a = BuildArray(offset, 4);
            var b = BuildArray(offset + 100, 4);
            var schema = new Schema(new[] { FieldA, FieldB }, null);
            batches.Add(new RecordBatch(schema, new IArrowArray[] { a, b }, 4));
            offset += 4;
        }

        var table = Table.TableFromRecordBatches(new Schema(new[] { FieldA, FieldB }, null), batches);
        int countA = table.Column(0).Data.ArrayCount;
        int countB = table.Column(1).Data.ArrayCount;

        bool ok = countA == 3 && countB == 3 && table.RowCount == 12;
        Report("TableFromRecordBatches shares one grid (3 batches -> 3 chunks per column)", ok,
            $"a={countA} b={countB} rows={table.RowCount}");

        return ok ? 0 : 1;
    }

    static int CheckPerColumnChunkCounts()
    {
        var columnA = new Column(FieldA, new IArrowArray[] { BuildArray(0, 4), BuildArray(4, 4) });
        var columnB = new Column(FieldB, new IArrowArray[]
        {
            BuildArray(100, 3),
            BuildArray(103, 3),
            BuildArray(106, 2),
        });

        var table = new Table(new Schema(new[] { FieldA, FieldB }, null), new[] { columnA, columnB });

        int countA = table.Column(0).Data.ArrayCount;
        int countB = table.Column(1).Data.ArrayCount;

        bool ok = countA == 2 && countB == 3 && table.RowCount == 8 && table.ColumnCount == 2;
        Report("new Table(schema, columns) preserves differing per-column chunk counts", ok,
            $"a={countA} b={countB} rows={table.RowCount} columns={table.ColumnCount}");

        return ok ? 0 : 1;
    }

    static int CheckColumnRoundTrip()
    {
        var column = new Column(FieldA, new IArrowArray[] { BuildArray(0, 4), BuildArray(4, 4) });
        var data = column.Data;

        bool ok = data.ArrayCount == 2
            && data.Length == 8
            && column.Length == 8
            && data.Array(0).Length == 4
            && data.Array(1).Length == 4;
        Report("new Column(field, IList<IArrowArray>) round-trips a two-chunk column", ok,
            $"chunks={data.ArrayCount} length={data.Length}");

        return ok ? 0 : 1;
    }

    static int CheckChunkedArray()
    {
        var chunked = new ChunkedArray(new IArrowArray[] { BuildArray(0, 4), BuildArray(4, 5) });

        bool ok = chunked.ArrayCount == 2
            && chunked.Length == 9
            && chunked.Array(0).Length == 4
            && chunked.Array(1).Length == 5;
        Report("new ChunkedArray(IList<IArrowArray>) reports ArrayCount and Length", ok,
            $"chunks={chunked.ArrayCount} length={chunked.Length}");

        return ok ? 0 : 1;
    }

    static Int32Array BuildArray(int start, int length)
    {
        var builder = new Int32Array.Builder();
        for (int i = 0; i < length; i++)
            builder.Append(start + i);
        return builder.Build();
    }

    static void Report(string description, bool ok, string detail)
    {
        Console.WriteLine($"{(ok ? "PASS" : "FAIL")}  {description}");
        Console.WriteLine($"      {detail}");
    }
}
