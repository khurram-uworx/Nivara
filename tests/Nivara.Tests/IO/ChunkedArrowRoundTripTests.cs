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
}
