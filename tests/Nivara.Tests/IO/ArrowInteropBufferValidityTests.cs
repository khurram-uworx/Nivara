using Apache.Arrow;
using Apache.Arrow.Arrays;
using Apache.Arrow.Types;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

[TestFixture]
public class ArrowInteropBufferValidityTests
{
    [Test]
    public void RoundTrip_IntColumn_AllNull_PreservesNulls()
    {
        var data = new int?[] { null, null, null };
        var column = NivaraColumnFactory.CreateFromNullable(data);
        var frame = NivaraFrame.Create(("Vals", column));

        var table = ArrowInterop.ToArrowTable(frame);
        var roundTrip = ArrowInterop.FromArrowTable(table);

        var result = (NivaraColumn<int>)roundTrip.GetColumn("Vals");
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result.HasNulls, Is.True);
        Assert.That(result.NullCount, Is.EqualTo(3));
    }

    [Test]
    public void RoundTrip_IntColumn_AllValid_PreservesValues()
    {
        var data = new[] { 10, 20, 30 };
        var column = NivaraColumn<int>.Create(data);
        var frame = NivaraFrame.Create(("Vals", column));

        var table = ArrowInterop.ToArrowTable(frame);
        var roundTrip = ArrowInterop.FromArrowTable(table);

        var result = roundTrip.GetColumn<int>("Vals");
        Assert.That(result.Length, Is.EqualTo(3));
        Assert.That(result.HasNulls, Is.False);
        Assert.That(result[0], Is.EqualTo(10));
        Assert.That(result[1], Is.EqualTo(20));
        Assert.That(result[2], Is.EqualTo(30));
    }

    [Test]
    public void RoundTrip_IntColumn_Mixed_PreservesNullsAndValues()
    {
        var data = new int?[] { 1, null, 3, null, 5 };
        var column = NivaraColumnFactory.CreateFromNullable(data);
        var frame = NivaraFrame.Create(("Vals", column));

        var table = ArrowInterop.ToArrowTable(frame);
        var roundTrip = ArrowInterop.FromArrowTable(table);

        var result = roundTrip.GetColumn<int>("Vals");
        Assert.That(result.Length, Is.EqualTo(5));
        Assert.That(result.IsNull(0), Is.False);
        Assert.That(result[0], Is.EqualTo(1));
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result[2], Is.EqualTo(3));
        Assert.That(result.IsNull(3), Is.True);
        Assert.That(result[4], Is.EqualTo(5));
    }

    [Test]
    public void FromArrowTable_ChunkedIntColumn_DecodesAndConcatenatesMasks()
    {
        // Chunk 1: [1, null]
        var b1 = new Int32Array.Builder();
        b1.Append(1);
        b1.AppendNull();
        var a1 = b1.Build();

        // Chunk 2: [null, 4]
        var b2 = new Int32Array.Builder();
        b2.AppendNull();
        b2.Append(4);
        var a2 = b2.Build();

        var field = new Field("Vals", Int32Type.Default, true);
        var schema = new Apache.Arrow.Schema(new[] { field }, null);
        var batch1 = new RecordBatch(schema, new IArrowArray[] { a1 }, 2);
        var batch2 = new RecordBatch(schema, new IArrowArray[] { a2 }, 2);
        var table = Table.TableFromRecordBatches(schema, new[] { batch1, batch2 });

        var frame = ArrowInterop.FromArrowTable(table);
        var col = frame.GetColumn<int>("Vals");

        Assert.That(col.Length, Is.EqualTo(4));
        Assert.That(col[0], Is.EqualTo(1));
        Assert.That(col.IsNull(1), Is.True);
        Assert.That(col.IsNull(2), Is.True);
        Assert.That(col[3], Is.EqualTo(4));
    }
}
