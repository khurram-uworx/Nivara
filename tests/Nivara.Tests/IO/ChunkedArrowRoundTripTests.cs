using Apache.Arrow;
using Apache.Arrow.Types;
using Nivara.IO;
using Nivara.Storage;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Evidence for the opt-in chunk-preserving Arrow path (#862). The default path still flattens;
/// with <see cref="ArrowConversionOptions.PreserveChunkStructure"/> the import keeps one chunk per
/// Arrow array, the export emits one array per chunk, and values and nulls are unchanged.
/// </summary>
[TestFixture]
public class ChunkedArrowRoundTripTests
{
    static readonly ArrowConversionOptions Preserve = new() { PreserveChunkStructure = true };
    static readonly ArrowConversionOptions PreserveZeroCopy = new() { PreserveChunkStructure = true, UseZeroCopy = true };

    static readonly Field FieldA = new("a", Int32Type.Default, nullable: true);
    static readonly Field FieldB = new("b", Int32Type.Default, nullable: true);

    static Table ThreeBatchTable()
    {
        var batches = new List<RecordBatch>();
        int offset = 0;
        for (int batchIndex = 0; batchIndex < 3; batchIndex++)
        {
            var a = BuildArray(offset, 4, nullAt: batchIndex == 1 ? 2 : -1);
            var b = BuildArray(offset + 100, 4, nullAt: -1);
            var schema = new Apache.Arrow.Schema(new[] { FieldA, FieldB }, null);
            batches.Add(new RecordBatch(schema, new IArrowArray[] { a, b }, 4));
            offset += 4;
        }

        return Table.TableFromRecordBatches(new Apache.Arrow.Schema(new[] { FieldA, FieldB }, null), batches);
    }

    static Int32Array BuildArray(int start, int length, int nullAt)
    {
        var builder = new Int32Array.Builder();
        for (int i = 0; i < length; i++)
        {
            if (i == nullAt)
                builder.AppendNull();
            else
                builder.Append(start + i);
        }

        return builder.Build();
    }

    static NivaraColumn<int> Chunked(params int[][] chunkValues)
    {
        var chunks = chunkValues
            .Select(values => new ColumnStorage<int>(values))
            .ToArray();
        return new NivaraColumn<int>(new ChunkedColumn<int>(chunks));
    }

    [Test]
    public void FromArrowTable_DefaultOption_FlattensToColumnStorage()
    {
        var frame = ArrowInterop.FromArrowTable(ThreeBatchTable());

        var column = (NivaraColumn<int>)frame.GetColumn("a");

        Assert.That(column.Storage, Is.InstanceOf<ColumnStorage<int>>());
    }

    [Test]
    public void FromArrowTable_PreserveOption_KeepsOneChunkPerArrowArray()
    {
        var frame = ArrowInterop.FromArrowTable(ThreeBatchTable(), Preserve);

        var column = (NivaraColumn<int>)frame.GetColumn("a");

        Assert.That(column.Storage, Is.InstanceOf<ChunkedColumn<int>>());
        var chunked = (ChunkedColumn<int>)column.Storage;
        Assert.That(chunked.ChunkStorages.Length, Is.EqualTo(3));
        Assert.That(column.Length, Is.EqualTo(12));
    }

    [Test]
    public void FromArrowTable_PreserveOption_PreservesValuesAndNulls()
    {
        var frame = ArrowInterop.FromArrowTable(ThreeBatchTable(), Preserve);

        var column = (NivaraColumn<int>)frame.GetColumn("a");

        Assert.That(column.HasNulls, Is.True);
        Assert.That(column.NullCount, Is.EqualTo(1));
        Assert.That(column.IsNull(6), Is.True);
        Assert.That(column[0], Is.EqualTo(0));
        Assert.That(column[5], Is.EqualTo(5));
        Assert.That(column[7], Is.EqualTo(7));
        Assert.That(column[11], Is.EqualTo(11));
    }

    [Test]
    public void ArrowRoundTrip_PreserveOption_KeepsThreeChunksPerColumn()
    {
        var frame = ArrowInterop.FromArrowTable(ThreeBatchTable(), Preserve);

        var table = ArrowInterop.ToArrowTable(frame, Preserve);

        Assert.Multiple(() =>
        {
            Assert.That(table.Column(0).Data.ArrayCount, Is.EqualTo(3));
            Assert.That(table.Column(1).Data.ArrayCount, Is.EqualTo(3));
            Assert.That(table.RowCount, Is.EqualTo(12));
        });
    }

    [Test]
    public void ArrowRoundTrip_PreserveOption_DoesNotChangeValues()
    {
        var source = ThreeBatchTable();

        var roundTrip = ArrowInterop.ToArrowTable(ArrowInterop.FromArrowTable(source, Preserve), Preserve);

        var a = (Int32Array)roundTrip.Column(0).Data.Array(0);
        var b = (Int32Array)roundTrip.Column(0).Data.Array(1);
        Assert.Multiple(() =>
        {
            Assert.That(a.GetValue(0), Is.EqualTo(0));
            Assert.That(a.IsNull(2), Is.False);
            Assert.That(b.GetValue(0), Is.EqualTo(4));
            Assert.That(b.IsNull(2), Is.True);
            Assert.That(b.GetValue(3), Is.EqualTo(7));
        });
    }

    [Test]
    public void ToArrowArray_ChunkedSeriesWithPreserveOption_Throws()
    {
        var series = new NivaraSeries<int>(Chunked(new[] { 1, 2 }, new[] { 3, 4 }));

        var exception = Assert.Throws<NotSupportedException>(() => ArrowInterop.ToArrowArray(series, Preserve));

        Assert.That(exception!.Message, Does.Contain("ToArrowTable"));
    }

    [Test]
    public void ToArrowTable_PreserveZeroCopy_EachChunkAliasesItsBackingArray()
    {
        var first = new ColumnStorage<int>(new[] { 10, 20 });
        var second = new ColumnStorage<int>(new[] { 30, 40 });
        var column = new NivaraColumn<int>(new ChunkedColumn<int>(new[] { first, second }));
        var frame = NivaraFrame.Create("values", column);

        var table = ArrowInterop.ToArrowTable(frame, PreserveZeroCopy);

        ((IColumnStorage<int>)first).AsWritableSpan()[0] = 99;

        var chunk = (Int32Array)table.Column(0).Data.Array(0);
        Assert.That(chunk.Values.Slice(0, 2).ToArray(), Is.EqualTo(new[] { 99, 20 }));
    }

    static ColumnStorage<int> NullChunk(int[] values, int nullAt)
    {
        if (nullAt < 0)
            return new ColumnStorage<int>(values);

        var mask = new bool[values.Length];
        mask[nullAt] = true;
        return new ColumnStorage<int>(values, new ReadOnlyMemory<bool>(mask));
    }

    static NivaraColumn<int> ChunkedColumn(params (int[] Values, int NullAt)[] chunks)
    {
        var storages = chunks.Select(c => NullChunk(c.Values, c.NullAt)).ToArray();
        return new NivaraColumn<int>(new ChunkedColumn<int>(storages));
    }

    static NivaraColumn<int> FlatColumn(int[] values, int nullAt = -1)
    {
        if (nullAt < 0)
            return new NivaraColumn<int>(new ColumnStorage<int>(values));

        var mask = new bool[values.Length];
        mask[nullAt] = true;
        return new NivaraColumn<int>(new ColumnStorage<int>(values, new ReadOnlyMemory<bool>(mask)));
    }

    [Test]
    public void ToArrowTable_Preserve_MixedChunkGrid_KeepsPerColumnChunkCounts()
    {
        // a: 2 chunks (lengths 2 + 4), b: 3 chunks (lengths 2 + 2 + 2), c: flat (length 6).
        var a = ChunkedColumn((new[] { 0, 1 }, -1), (new[] { 2, 3, 4, 5 }, 2));
        var b = ChunkedColumn((new[] { 10, 11 }, -1), (new[] { 12, 13 }, -1), (new[] { 14, 15 }, 0));
        var c = FlatColumn(new[] { 20, 21, 22, 23, 24, 25 }, 5);

        var frame = NivaraFrame.Create(("a", a), ("b", b), ("c", c));

        var table = ArrowInterop.ToArrowTable(frame, Preserve);

        Assert.Multiple(() =>
        {
            Assert.That(table.Column(0).Data.ArrayCount, Is.EqualTo(2));
            Assert.That(table.Column(1).Data.ArrayCount, Is.EqualTo(3));
            Assert.That(table.Column(2).Data.ArrayCount, Is.EqualTo(1));
            Assert.That(table.RowCount, Is.EqualTo(6));
        });

        // a values and null survive intact (chunk lengths 2 + 4).
        var a0 = (Int32Array)table.Column(0).Data.Array(0);
        var a1 = (Int32Array)table.Column(0).Data.Array(1);
        Assert.Multiple(() =>
        {
            Assert.That(a0.GetValue(0), Is.EqualTo(0));
            Assert.That(a0.GetValue(1), Is.EqualTo(1));
            Assert.That(a1.GetValue(0), Is.EqualTo(2));
            Assert.That(a1.GetValue(1), Is.EqualTo(3));
            Assert.That(a1.IsNull(2), Is.True);
            Assert.That(a1.GetValue(3), Is.EqualTo(5));
        });

        // b values and null survive intact (3 chunks of 2).
        var b0 = (Int32Array)table.Column(1).Data.Array(0);
        var b1 = (Int32Array)table.Column(1).Data.Array(1);
        var b2 = (Int32Array)table.Column(1).Data.Array(2);
        Assert.Multiple(() =>
        {
            Assert.That(b0.GetValue(0), Is.EqualTo(10));
            Assert.That(b0.GetValue(1), Is.EqualTo(11));
            Assert.That(b1.GetValue(0), Is.EqualTo(12));
            Assert.That(b1.GetValue(1), Is.EqualTo(13));
            Assert.That(b2.IsNull(0), Is.True);
            Assert.That(b2.GetValue(1), Is.EqualTo(15));
        });

        // c flat survives as a single array with a trailing null.
        var c0 = (Int32Array)table.Column(2).Data.Array(0);
        Assert.Multiple(() =>
        {
            Assert.That(c0.Length, Is.EqualTo(6));
            Assert.That(c0.GetValue(0), Is.EqualTo(20));
            Assert.That(c0.GetValue(4), Is.EqualTo(24));
            Assert.That(c0.IsNull(5), Is.True);
        });
    }

    [Test]
    public void ToArrowTable_Preserve_AllFlatFallback_MatchesDefaultRoute()
    {
        var flatFrame = NivaraFrame.Create(
            ("a", FlatColumn(new[] { 0, 1, 2, 3, 4, 5 }, 2)),
            ("b", FlatColumn(new[] { 10, 11, 12, 13, 14, 15 }, -1)),
            ("c", FlatColumn(new[] { 20, 21, 22, 23, 24, 25 }, 5)));

        var preserved = ArrowInterop.ToArrowTable(flatFrame, Preserve);
        var defaulted = ArrowInterop.ToArrowTable(flatFrame, null);

        Assert.Multiple(() =>
        {
            for (int i = 0; i < flatFrame.ColumnCount; i++)
                Assert.That(preserved.Column(i).Data.ArrayCount, Is.EqualTo(1));
        });

        Assert.That(preserved.RowCount, Is.EqualTo(defaulted.RowCount));
        for (int col = 0; col < flatFrame.ColumnCount; col++)
        {
            var pa = (Int32Array)preserved.Column(col).Data.Array(0);
            var da = (Int32Array)defaulted.Column(col).Data.Array(0);
            Assert.That(pa.Length, Is.EqualTo(da.Length));
            for (int i = 0; i < pa.Length; i++)
            {
                Assert.That(pa.IsNull(i), Is.EqualTo(da.IsNull(i)));
                if (!pa.IsNull(i))
                    Assert.That(pa.GetValue(i), Is.EqualTo(da.GetValue(i)));
            }
        }
    }

    static readonly Field FieldS = new("s", StringType.Default, nullable: true);

    static Table TwoChunkStringTable()
    {
        var schema = new Apache.Arrow.Schema(new[] { FieldS }, null);

        var firstBuilder = new StringArray.Builder();
        firstBuilder.Append("a0");
        firstBuilder.Append("a1");
        firstBuilder.AppendNull();
        firstBuilder.Append("a3");

        var secondBuilder = new StringArray.Builder();
        secondBuilder.Append("b0");
        secondBuilder.AppendNull();
        secondBuilder.Append("b2");
        secondBuilder.Append("b3");

        var batches = new List<RecordBatch>
        {
            new RecordBatch(schema, new IArrowArray[] { firstBuilder.Build() }, 4),
            new RecordBatch(schema, new IArrowArray[] { secondBuilder.Build() }, 4),
        };
        return Table.TableFromRecordBatches(schema, batches);
    }

    [Test]
    public void FromArrowTable_PreserveOption_String_KeepsChunksAndNulls()
    {
        var frame = ArrowInterop.FromArrowTable(TwoChunkStringTable(), Preserve);

        var column = (NivaraColumn<string>)frame.GetColumn("s");

        Assert.That(column.Storage, Is.InstanceOf<ChunkedColumn<string>>());
        var chunked = (ChunkedColumn<string>)column.Storage;
        Assert.That(chunked.ChunkStorages.Length, Is.EqualTo(2));
        Assert.That(column.Length, Is.EqualTo(8));

        Assert.That(column.HasNulls, Is.True);
        Assert.That(column.NullCount, Is.EqualTo(2));
        Assert.That(column.IsNull(2), Is.True);
        Assert.That(column.IsNull(5), Is.True);

        Assert.That(column[0], Is.EqualTo("a0"));
        Assert.That(column[1], Is.EqualTo("a1"));
        Assert.That(column[3], Is.EqualTo("a3"));
        Assert.That(column[4], Is.EqualTo("b0"));
        Assert.That(column[6], Is.EqualTo("b2"));
        Assert.That(column[7], Is.EqualTo("b3"));
    }

    [Test]
    public void ArrowRoundTrip_PreserveOption_String_KeepsChunksValuesAndNulls()
    {
        var roundTrip = ArrowInterop.ToArrowTable(
            ArrowInterop.FromArrowTable(TwoChunkStringTable(), Preserve), Preserve);

        Assert.That(roundTrip.Column(0).Data.ArrayCount, Is.EqualTo(2));

        var a = (StringArray)roundTrip.Column(0).Data.Array(0);
        var b = (StringArray)roundTrip.Column(0).Data.Array(1);
        Assert.Multiple(() =>
        {
            Assert.That(a.IsNull(2), Is.True);
            Assert.That(b.IsNull(1), Is.True);
            Assert.That(a.GetString(0), Is.EqualTo("a0"));
            Assert.That(a.GetString(1), Is.EqualTo("a1"));
            Assert.That(a.GetString(3), Is.EqualTo("a3"));
            Assert.That(b.GetString(0), Is.EqualTo("b0"));
            Assert.That(b.GetString(2), Is.EqualTo("b2"));
            Assert.That(b.GetString(3), Is.EqualTo("b3"));
        });
    }

    [Test]
    public void FromArrowTable_DefaultAndPreserve_String_AgreeOnNullSemantics()
    {
        var table = TwoChunkStringTable();

        var flat = (NivaraColumn<string>)ArrowInterop.FromArrowTable(table).GetColumn("s");
        var chunked = (NivaraColumn<string>)ArrowInterop.FromArrowTable(table, Preserve).GetColumn("s");

        Assert.That(flat.Storage, Is.InstanceOf<ColumnStorage<string>>());
        Assert.That(flat.HasNulls, Is.EqualTo(chunked.HasNulls));
        Assert.That(flat.NullCount, Is.EqualTo(chunked.NullCount));
        Assert.That(flat.Length, Is.EqualTo(chunked.Length));

        for (int i = 0; i < flat.Length; i++)
        {
            Assert.That(flat.IsNull(i), Is.EqualTo(chunked.IsNull(i)), $"IsNull mismatch at {i}");
            if (!flat.IsNull(i))
                Assert.That(flat[i], Is.EqualTo(chunked[i]), $"value mismatch at {i}");
        }
    }
}
