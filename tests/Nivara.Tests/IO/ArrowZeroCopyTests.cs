using Apache.Arrow;
using Apache.Arrow.Ipc;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Evidence for the zero-copy export path added by #835. Every test here pins a claim the
/// landing commit made without a test: that the Arrow data buffer aliases the column's
/// backing array, that a slice offset is preserved, that the arrow array outlives the column,
/// and that an aliased buffer survives IPC serialization when its length is not an 8-byte
/// multiple.
/// </summary>
[TestFixture]
public class ArrowZeroCopyTests
{
    static readonly ArrowConversionOptions ZeroCopy = new() { UseZeroCopy = true };

    static Int32Array Int32ArrayOf(Table table) => (Int32Array)table.Column(0).Data.Array(0);

    [Test]
    public void ToArrowTable_ZeroCopyFlatPrimitive_SharesColumnBackingArray()
    {
        var column = NivaraColumn<int>.Create(new[] { 10, 20, 30, 40 });
        var frame = NivaraFrame.Create("values", column);

        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);

        var dataBuffer = Int32ArrayOf(table).Data.Buffers[1];
        Assert.That(dataBuffer.Length, Is.EqualTo(column.Length * sizeof(int)));

        // MemoryMarshal.TryGetArray cannot see through a MemoryManager<byte>, so aliasing is
        // proven by observing a write through the column's writable span.
        column.AsWritableSpan()[2] = 99;

        Assert.That(
            Int32ArrayOf(table).Values.Slice(0, column.Length).ToArray(),
            Is.EqualTo(new[] { 10, 20, 99, 40 }));
    }

    [Test]
    public void ToArrowTable_ZeroCopySlicedColumn_PreservesOffsetAndValues()
    {
        var source = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var slice = source.Slice(2, 3);
        var frame = NivaraFrame.Create("values", slice);

        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);

        var array = Int32ArrayOf(table);
        Assert.Multiple(() =>
        {
            Assert.That(array.Length, Is.EqualTo(3));
            Assert.That(array.Offset, Is.EqualTo(0));
            Assert.That(array.Values.Slice(0, 3).ToArray(), Is.EqualTo(new[] { 3, 4, 5 }));
        });

        slice.AsWritableSpan()[0] = 30;
        Assert.That(Int32ArrayOf(table).Values.Slice(0, 3).ToArray(), Is.EqualTo(new[] { 30, 4, 5 }));
    }

    [Test]
    public void ToArrowTable_ZeroCopyColumn_AfterColumnDisposed_ArrowArrayStillReads()
    {
        var column = NivaraColumn<int>.Create(new[] { 7, 8, 9 });
        var frame = NivaraFrame.Create("values", column);

        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);

        column.Dispose();

        Assert.That(Int32ArrayOf(table).Values.Slice(0, 3).ToArray(), Is.EqualTo(new[] { 7, 8, 9 }));
    }

    [Test]
    public void ToArrowTable_ZeroCopyThreeRowIntColumn_IPCRoundTrip()
    {
        var column = NivaraColumn<int>.Create(new[] { -1, 0, 1234567 });
        var frame = NivaraFrame.Create("values", column);
        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);

        Assert.That(
            Int32ArrayOf(table).Data.Buffers[1].Length,
            Is.EqualTo(12),
            "3 ints is 12 bytes - not an 8-byte multiple, which is the alignment case under test.");

        using var stream = new MemoryStream();
        using (var writer = new ArrowStreamWriter(stream, table.Schema, leaveOpen: true))
        {
            writer.WriteStart();
            writer.WriteRecordBatch(new RecordBatch(
                table.Schema,
                new IArrowArray[] { Int32ArrayOf(table) },
                (int)table.RowCount));
            writer.WriteEnd();
        }

        stream.Position = 0;

        using var reader = new ArrowStreamReader(stream);
        using var readBack = reader.ReadNextRecordBatch();
        Assert.That(readBack, Is.Not.Null);

        var readBackArray = (Int32Array)readBack!.Arrays.First();
        Assert.Multiple(() =>
        {
            Assert.That(readBackArray.Length, Is.EqualTo(3));
            Assert.That(readBackArray.Values.Slice(0, 3).ToArray(), Is.EqualTo(new[] { -1, 0, 1234567 }));
        });
    }

    [Test]
    public void ToArrowArray_ZeroCopySeries_MatchesTablePath()
    {
        using var series = NivaraSeries<int>.Create(new[] { 11, 12, 13 });

        var array = (Int32Array)ArrowInterop.ToArrowArray(series, ZeroCopy);

        Assert.Multiple(() =>
        {
            Assert.That(array.Length, Is.EqualTo(3));
            Assert.That(array.NullCount, Is.EqualTo(0));
            Assert.That(array.Values.Slice(0, 3).ToArray(), Is.EqualTo(new[] { 11, 12, 13 }));
        });
    }
}
