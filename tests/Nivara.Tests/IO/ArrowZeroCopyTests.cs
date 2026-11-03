using Apache.Arrow;
using Apache.Arrow.Ipc;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Evidence for the zero-copy export path added by #835 and extended by #836. Every test here
/// pins a claim about that path: the Arrow data buffer aliases the column's backing array, a
/// slice offset is preserved, the Arrow array outlives the column, an aliased buffer survives
/// IPC serialization when its length is not an 8-byte multiple, and a null-bearing column
/// shares its values buffer while its validity bitmap is rebuilt from the null mask.
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

    [Test]
    public unsafe void ToArrowTable_ZeroCopyDataBuffer_PinAtByteOffset_PointsAtThatElement()
    {
        var column = NivaraColumn<int>.Create(new[] { 10, 20, 30, 40 });
        var frame = NivaraFrame.Create("values", column);
        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);

        var dataBuffer = Int32ArrayOf(table).Data.Buffers[1];
        using var handle = dataBuffer.Memory.Slice(sizeof(int), sizeof(int)).Pin();

        Assert.That(*(int*)handle.Pointer, Is.EqualTo(20));
    }

    [Test]
    public unsafe void ToArrowTable_ZeroCopySlicedBuffer_PinAtZero_PointsAtSliceStart()
    {
        var source = NivaraColumn<int>.Create(new[] { 1, 2, 3, 4, 5 });
        var slice = source.Slice(2, 3);
        var frame = NivaraFrame.Create("values", slice);
        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);

        var dataBuffer = Int32ArrayOf(table).Data.Buffers[1];
        using var handle = dataBuffer.Memory.Pin();

        Assert.That(*(int*)handle.Pointer, Is.EqualTo(3));
    }

    [Test]
    public void ToArrowTable_ZeroCopyNullBearingColumn_BuildsValidityAndSharesValues()
    {
        var column = NivaraColumnFactory.CreateFromNullable(new int?[] { 1, null, 3, null });
        var frame = NivaraFrame.Create("values", column);

        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);
        var copyPathTable = ArrowInterop.ToArrowTable(frame);

        var array = Int32ArrayOf(table);
        var copyPathArray = (Int32Array)copyPathTable.Column(0).Data.Array(0);

        Assert.Multiple(() =>
        {
            Assert.That(array.NullCount, Is.EqualTo(2));
            Assert.That(array.IsNull(1), Is.True);
            Assert.That(array.IsNull(3), Is.True);
            Assert.That(array.IsNull(0), Is.False);
            Assert.That(array.IsNull(2), Is.False);
            Assert.That(
                array.Data.Buffers[0].Span.ToArray(),
                Is.EqualTo(copyPathArray.Data.Buffers[0].Span.ToArray()),
                "the rebuilt validity bitmap must match the copying path bit for bit");
        });

        column.AsWritableSpan()[2] = 99;
        Assert.That(Int32ArrayOf(table).Values[2], Is.EqualTo(99));
    }

    [Test]
    public void ToArrowTable_ZeroCopyNullBearingColumn_RoundTripsNulls()
    {
        var column = NivaraColumnFactory.CreateFromNullable(new int?[] { 1, null, 3, null });
        var frame = NivaraFrame.Create("values", column);

        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);
        var roundTrip = ArrowInterop.FromArrowTable(table);
        var roundTripColumn = roundTrip.GetColumn<int>("values");

        Assert.Multiple(() =>
        {
            Assert.That(roundTripColumn.Length, Is.EqualTo(4));
            Assert.That(roundTripColumn.IsNull(1), Is.True);
            Assert.That(roundTripColumn.IsNull(3), Is.True);
            Assert.That(roundTripColumn[0], Is.EqualTo(1));
            Assert.That(roundTripColumn[2], Is.EqualTo(3));
        });
    }

    [Test]
    public void ToArrowTable_ZeroCopySlicedNullBearingColumn_PreservesMaskOffset()
    {
        var source = NivaraColumnFactory.CreateFromNullable(new int?[] { null, 2, null, 4, 5 });
        var slice = source.Slice(1, 3);
        var frame = NivaraFrame.Create("values", slice);

        var table = ArrowInterop.ToArrowTable(frame, ZeroCopy);
        var array = Int32ArrayOf(table);

        Assert.Multiple(() =>
        {
            Assert.That(array.Length, Is.EqualTo(3));
            Assert.That(array.NullCount, Is.EqualTo(1));
            Assert.That(array.IsNull(1), Is.True);
            Assert.That(array.IsNull(0), Is.False);
            Assert.That(array.IsNull(2), Is.False);
            Assert.That(array.Values[0], Is.EqualTo(2));
            Assert.That(array.Values[2], Is.EqualTo(4));
        });
    }

    [Test]
    public void ToArrowTable_ZeroCopyStringColumn_ThrowsUnsupportedTypeNamingColumn()
    {
        var column = NivaraColumn<string>.Create(new[] { "alpha", "beta" });
        var frame = NivaraFrame.Create("labels", column);

        var exception = Assert.Throws<UnsupportedTypeException>(
            () => ArrowInterop.ToArrowTable(frame, ZeroCopy));

        Assert.Multiple(() =>
        {
            Assert.That(exception!.UnsupportedType, Is.EqualTo(typeof(string)));
            Assert.That(exception.Message, Does.Contain("labels"));
            Assert.That(exception.Message, Does.Contain("copying path"));
        });
    }

    [Test]
    public void ToArrowArray_ZeroCopyNullBearingSeries_PreservesNulls()
    {
        var column = NivaraColumnFactory.CreateFromNullable(new int?[] { 1, null });
        using var series = new NivaraSeries<int>(column);

        var array = (Int32Array)ArrowInterop.ToArrowArray(series, ZeroCopy);

        Assert.Multiple(() =>
        {
            Assert.That(array.Length, Is.EqualTo(2));
            Assert.That(array.NullCount, Is.EqualTo(1));
            Assert.That(array.IsNull(1), Is.True);
            Assert.That(array.Values[0], Is.EqualTo(1));
        });
    }
}
