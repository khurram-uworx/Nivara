using Apache.Arrow;
using Apache.Arrow.Types;
using Nivara;
using Nivara.IO;
using NUnit.Framework;

namespace Nivara.Tests.IO;

/// <summary>
/// Validation-first tests for issue #834: proves that Apache Arrow's bit-packed validity buffer
/// and Nivara's byte-wise <see cref="bool"/> null mask translate exactly in both directions,
/// across empty / all-null / all-valid / mixed shapes, using the span-first
/// <see cref="ArrowValidityMask"/> helper as the reference encoding.
/// </summary>
/// <remarks>
/// These tests deliberately keep the existing builders+<c>Append</c> data path and do not claim
/// zero-copy or production wiring of the helper. Nivara stores an all-valid column as an empty
/// mask and Arrow elides its validity buffer; exactness is asserted under that documented
/// absent-mask convention. Padding bits (trailing bits of the final byte) are undefined by the
/// Arrow spec and are compared only over defined bits.
/// </remarks>
[TestFixture]
public class ArrowValidityInteropTests
{
    private static NivaraSeries<int> MakeIntSeries(params int?[] values) =>
        new NivaraSeries<int>(NivaraColumnFactory.CreateFromNullable(values));

    private static NivaraSeries<bool> MakeBoolSeries(params bool?[] values) =>
        new NivaraSeries<bool>(NivaraColumnFactory.CreateFromNullable(values));

    private static NivaraSeries<string> MakeStringSeries(params string?[] values) =>
        new NivaraSeries<string>(NivaraColumn<string>.Create(values.Select(v => v!).ToArray()));

    private static void AssertExportValidityMatchesHelper(IArrowArray array, ReadOnlySpan<bool> nivaraMask)
    {
        int length = (int)array.Length;

        int expectedNullCount = 0;
        foreach (bool isNull in nivaraMask) if (isNull) expectedNullCount++;
        Assert.That(array.NullCount, Is.EqualTo(expectedNullCount));

        var arrowArray = (Apache.Arrow.Array)array;

        // Elided Arrow validity buffer == all valid, matching Nivara's empty mask.
        if (arrowArray.NullBitmapBuffer.Length == 0)
        {
            Assert.That(expectedNullCount, Is.EqualTo(0));
            return;
        }

        int required = ArrowValidityMask.GetBitmapLength(length);
        if (required == 0) return;

        var helperBytes = new byte[required];
        ArrowValidityMask.ToValidityBitmap(nivaraMask, length, helperBytes);

        int compareBytes = Math.Min(required, arrowArray.NullBitmapBuffer.Length);
        for (int b = 0; b < compareBytes; b++)
        {
            byte arrowByte = arrowArray.NullBitmapBuffer.Span[b];
            // Mask undefined trailing bits of the final byte on the Arrow side.
            if (b == required - 1)
            {
                int usedBits = length & 7;
                if (usedBits != 0) arrowByte &= (byte)((1 << usedBits) - 1);
            }
            Assert.That(arrowByte, Is.EqualTo(helperBytes[b]), $"validity byte {b}");
        }
    }

    private static void AssertImportMaskMatchesHelper<T>(NivaraSeries<T> series, IArrowArray array)
    {
        int length = (int)array.Length;
        var arrowBytes = ((Apache.Arrow.Array)array).NullBitmapBuffer.Span;
        var helperMask = ArrowValidityMask.ToNullMask(arrowBytes, length);

        Assert.That(series.Values.Length, Is.EqualTo(length));
        for (int i = 0; i < length; i++)
        {
            Assert.That(series.Values.IsNull(i), Is.EqualTo(helperMask[i]), $"IsNull at index {i}");
        }
    }

    // ---------- Export: Nivara -> Arrow validity buffer ----------

    [Test]
    public void ToArrowArray_EmptyColumn_NullBufferMatchesHelper()
    {
        var series = new NivaraSeries<int>(NivaraColumn<int>.Create(System.Array.Empty<int>()));
        var array = ArrowInterop.ToArrowArray(series);
        Assert.That(array.Length, Is.EqualTo(0));

        ReadOnlySpan<bool> mask = ReadOnlySpan<bool>.Empty;
        AssertExportValidityMatchesHelper(array, mask);
    }

    [Test]
    public void ToArrowArray_AllNullIntColumn_NullBufferMatchesHelper()
    {
        var series = MakeIntSeries(null, null, null, null);
        var array = ArrowInterop.ToArrowArray(series);

        var mask = series.Values.TryGetNullMask(out var m) ? m : ReadOnlySpan<bool>.Empty;
        Assert.That(mask.Length, Is.EqualTo(4));
        AssertExportValidityMatchesHelper(array, mask);
    }

    [Test]
    public void ToArrowArray_AllValidIntColumn_ElidedBufferMatchesHelper()
    {
        var series = MakeIntSeries(1, 2, 3, 4);
        var array = ArrowInterop.ToArrowArray(series);

        var mask = series.Values.TryGetNullMask(out var m) ? m : ReadOnlySpan<bool>.Empty;
        Assert.That(mask.Length, Is.EqualTo(0));
        AssertExportValidityMatchesHelper(array, mask);
    }

    [Test]
    public void ToArrowArray_MixedNullIntColumn_NullBufferMatchesHelper()
    {
        var series = MakeIntSeries(1, null, 3, null, 5);
        var array = ArrowInterop.ToArrowArray(series);

        var mask = series.Values.TryGetNullMask(out var m) ? m : ReadOnlySpan<bool>.Empty;
        AssertExportValidityMatchesHelper(array, mask);
    }

    [Test]
    public void ToArrowArray_MixedNullStringColumn_NullBufferMatchesHelper()
    {
        var series = MakeStringSeries("a", null, "", "d", null);
        var array = ArrowInterop.ToArrowArray(series);

        var mask = series.Values.TryGetNullMask(out var m) ? m : ReadOnlySpan<bool>.Empty;
        AssertExportValidityMatchesHelper(array, mask);
    }

    [Test]
    public void ToArrowArray_MixedNullBoolColumn_NullBufferMatchesHelper()
    {
        var series = MakeBoolSeries(true, null, false, null, true);
        var array = ArrowInterop.ToArrowArray(series);

        var mask = series.Values.TryGetNullMask(out var m) ? m : ReadOnlySpan<bool>.Empty;
        AssertExportValidityMatchesHelper(array, mask);
    }

    [Test]
    public void ToArrowTable_MultiColumn_MixedNulls_EachColumnValidityMatchesHelper()
    {
        var intColumn = NivaraColumnFactory.CreateFromNullable(new int?[] { 1, null, 3 });
        var stringColumn = NivaraColumn<string>.Create(new string?[] { "a", null, "c" }.Select(v => v!).ToArray());

        var frame = NivaraFrame.Create(("Ints", intColumn), ("Strs", stringColumn));
        var table = ArrowInterop.ToArrowTable(frame);

        Assert.That(table.ColumnCount, Is.EqualTo(2));

        var intChunk = (IArrowArray)table.Column(0).Data.Array(0);
        var intMask = intColumn.TryGetNullMask(out var im) ? im : ReadOnlySpan<bool>.Empty;
        AssertExportValidityMatchesHelper(intChunk, intMask);

        var stringChunk = (IArrowArray)table.Column(1).Data.Array(0);
        var stringMask = stringColumn.TryGetNullMask(out var sm) ? sm : ReadOnlySpan<bool>.Empty;
        AssertExportValidityMatchesHelper(stringChunk, stringMask);
    }

    // ---------- Import: Arrow -> Nivara mask ----------

    [Test]
    public void FromArrowArray_EmptyArray_MaskMatchesHelper()
    {
        var builder = new Int32Array.Builder();
        var array = builder.Build();
        var series = ArrowInterop.FromArrowArray<int>(array);
        AssertImportMaskMatchesHelper(series, array);
    }

    [Test]
    public void FromArrowArray_AllNullIntColumn_MaskMatchesHelper()
    {
        var builder = new Int32Array.Builder();
        for (int i = 0; i < 5; i++) builder.AppendNull();
        var array = builder.Build();
        var series = ArrowInterop.FromArrowArray<int>(array);
        AssertImportMaskMatchesHelper(series, array);
    }

    [Test]
    public void FromArrowArray_AllValidIntColumn_MaskMatchesHelper()
    {
        var builder = new Int32Array.Builder();
        for (int i = 0; i < 5; i++) builder.Append(i);
        var array = builder.Build();
        var series = ArrowInterop.FromArrowArray<int>(array);
        AssertImportMaskMatchesHelper(series, array);
    }

    [Test]
    public void FromArrowArray_MixedNullIntColumn_MaskMatchesHelper()
    {
        var builder = new Int32Array.Builder();
        builder.Append(1);
        builder.AppendNull();
        builder.Append(3);
        builder.AppendNull();
        builder.Append(5);
        var array = builder.Build();
        var series = ArrowInterop.FromArrowArray<int>(array);
        AssertImportMaskMatchesHelper(series, array);
    }

    [Test]
    public void FromArrowArray_MixedNullStringColumn_MaskMatchesHelper()
    {
        var builder = new StringArray.Builder();
        builder.Append("a");
        builder.AppendNull();
        builder.Append("c");
        var array = builder.Build();
        var series = ArrowInterop.FromArrowArray<string>(array);
        AssertImportMaskMatchesHelper(series, array);
    }

    [Test]
    public void FromArrowArray_HandBuiltValidityBitmap_PaddingBitsIgnored()
    {
        int length = 10;
        int required = ArrowValidityMask.GetBitmapLength(length);

        byte[] validity = new byte[required];
        // Defined slots: even indices valid (bit set), odd indices null.
        for (int i = 0; i < length; i++)
        {
            if (i % 2 == 0) validity[i >> 3] |= (byte)(1 << (i & 7));
        }
        // Dirty every padding bit in the final byte to 1.
        if (required > 0) validity[required - 1] = (byte)(validity[required - 1] | 0xFF);

        int nullCount = length / 2;
        int[] values = new int[length];

        var valueBytes = new byte[values.Length * sizeof(int)];
        Buffer.BlockCopy(values, 0, valueBytes, 0, valueBytes.Length);

        var validityBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(validity));
        var valueBuffer = new ArrowBuffer(new ReadOnlyMemory<byte>(valueBytes));

        var data = new ArrayData(Int32Type.Default, length, nullCount, 0,
            new[] { validityBuffer, valueBuffer });

        var array = (Int32Array)ArrowArrayFactory.BuildArray(data);
        var series = ArrowInterop.FromArrowArray<int>(array);

        AssertImportMaskMatchesHelper(series, array);

        var expectedMask = ArrowValidityMask.ToNullMask(validity, length);
        for (int i = 0; i < length; i++)
        {
            Assert.That(series.Values.IsNull(i), Is.EqualTo(expectedMask[i]), $"index {i}");
        }
    }

    // ---------- Round-trip ----------

    [Test]
    public void RoundTrip_AllNullIntColumn_PreservesNullMask()
    {
        var original = MakeIntSeries(null, null, null);
        var arrow = ArrowInterop.ToArrowArray(original);
        var roundTrip = ArrowInterop.FromArrowArray<int>(arrow);

        Assert.That(roundTrip.Values.Length, Is.EqualTo(original.Values.Length));
        for (int i = 0; i < original.Values.Length; i++)
        {
            Assert.That(roundTrip.Values.IsNull(i), Is.EqualTo(original.Values.IsNull(i)), $"index {i}");
        }
    }

    [Test]
    public void RoundTrip_AllValidIntColumn_PreservesNullMask()
    {
        var original = MakeIntSeries(1, 2, 3);
        var arrow = ArrowInterop.ToArrowArray(original);
        var roundTrip = ArrowInterop.FromArrowArray<int>(arrow);

        Assert.That(roundTrip.Values.Length, Is.EqualTo(original.Values.Length));
        for (int i = 0; i < original.Values.Length; i++)
        {
            Assert.That(roundTrip.Values.IsNull(i), Is.EqualTo(original.Values.IsNull(i)), $"index {i}");
        }
        Assert.That(roundTrip.Values.HasNulls, Is.False);
    }

    [Test]
    public void RoundTrip_MixedNullIntColumn_PreservesNullMask()
    {
        var original = MakeIntSeries(1, null, 3, null, 5);
        var arrow = ArrowInterop.ToArrowArray(original);
        var roundTrip = ArrowInterop.FromArrowArray<int>(arrow);

        Assert.That(roundTrip.Values.Length, Is.EqualTo(original.Values.Length));
        for (int i = 0; i < original.Values.Length; i++)
        {
            Assert.That(roundTrip.Values.IsNull(i), Is.EqualTo(original.Values.IsNull(i)), $"index {i}");
        }
    }

    [Test]
    public void RoundTrip_EmptyIntColumn_PreservesNullMask()
    {
        var original = new NivaraSeries<int>(NivaraColumn<int>.Create(System.Array.Empty<int>()));
        var arrow = ArrowInterop.ToArrowArray(original);
        var roundTrip = ArrowInterop.FromArrowArray<int>(arrow);

        Assert.That(roundTrip.Values.Length, Is.EqualTo(0));
    }
}
