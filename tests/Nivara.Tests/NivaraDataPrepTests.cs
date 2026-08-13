using NUnit.Framework;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Nivara.Tests;

[TestFixture]
public class NivaraDataPrepTests
{
    [Test]
    public void Normalization_ProducesZeroMeanUnitVariance()
    {
        // Create test data with known statistics
        var data = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f };
        var col = NivaraColumn<float>.Create(data);
        var frame = NivaraFrame.Create(("Values", col));

        // Normalize the data
        var normalizedFrame = frame.Normalize("Values");

        // Extract normalized values
        var normalizedValues = new float[normalizedFrame.RowCount];
        var normalizedCol = normalizedFrame.GetColumn<float>("Values");
        for (int i = 0; i < normalizedFrame.RowCount; i++)
        {
            normalizedValues[i] = normalizedCol[i];
        }

        // Verify zero mean (within floating point precision)
        var mean = normalizedValues.Average();
        Assert.That(mean, Is.EqualTo(0.0f).Within(1e-6f));

        // Verify unit variance
        var variance = normalizedValues.Select(x => Math.Pow(x - mean, 2)).Average();
        var stdDev = Math.Sqrt(variance);
        Assert.That(stdDev, Is.EqualTo(1.0f).Within(1e-6f));
    }

    [Test]
    public void Standardize_IsAliasForNormalize()
    {
        var data = new float[] { 1f, 2f, 3f, 4f, 5f };
        var frame = NivaraFrame.Create(("Values", NivaraColumn<float>.Create(data)));

        var normalized = frame.Normalize("Values").GetColumn<float>("Values");
        var standardized = frame.Standardize("Values").GetColumn<float>("Values");

        for (int i = 0; i < data.Length; i++)
            Assert.That(standardized[i], Is.EqualTo(normalized[i]));
    }

    [Test]
    public void Standardize_SkipsNulls_PreservesNullMask()
    {
        var data = new float?[] { 1f, 2f, null, 4f, 5f };
        var frame = NivaraFrame.Create(("Values", NivaraColumn.CreateFromNullable(data)));

        var result = frame.Standardize("Values");
        var resultCol = result.GetColumn<float>("Values");

        for (int i = 0; i < data.Length; i++)
            Assert.That(resultCol.IsNull(i), Is.EqualTo(data[i] == null));

        // Statistics must be computed over non-null values only
        var nonNull = data.Where(v => v.HasValue).Select(v => v!.Value).ToArray();
        var mean = nonNull.Average();
        var stdDev = Math.Sqrt(nonNull.Select(x => Math.Pow(x - mean, 2)).Average());

        for (int i = 0; i < data.Length; i++)
        {
            if (data[i] is float v)
                Assert.That(resultCol[i], Is.EqualTo((float)((v - mean) / stdDev)).Within(1e-5f));
        }
    }

    [Test]
    public void Standardize_DefaultsToAllSupportedNumericColumns()
    {
        var frame = NivaraFrame.Create(new (string, IColumn)[]
        {
            ("Num", NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f })),
            ("Count", NivaraColumn<int>.Create(new int[] { 10, 20, 30 })),
            ("Text", NivaraColumn<string>.Create(new string[] { "a", "b", "c" })),
        });

        var result = frame.Standardize();

        var numValues = result.GetColumn<float>("Num").ToArray();
        Assert.That(numValues.Average(), Is.EqualTo(0f).Within(1e-6f));

        // Every INumber column is normalized; non-numeric columns are left untouched
        AssertNormalizedDoubleColumn(result, "Count");
        Assert.That(result.GetColumn<string>("Text").ToArray(), Is.EqualTo(new[] { "a", "b", "c" }));
    }

    [Test]
    public void Standardize_ZeroVariance_LeavesValuesUnchanged()
    {
        var data = new float[] { 5f, 5f, 5f };
        var frame = NivaraFrame.Create(("Values", NivaraColumn<float>.Create(data)));

        var resultCol = frame.Standardize("Values").GetColumn<float>("Values");

        for (int i = 0; i < data.Length; i++)
            Assert.That(resultCol[i], Is.EqualTo(5f));
    }

    [Test]
    public void Standardize_ZeroVarianceWithNulls_PreservesNulls()
    {
        var data = new float?[] { 3f, null, 3f, 3f };
        var frame = NivaraFrame.Create(("Values", NivaraColumn.CreateFromNullable(data)));

        var resultCol = frame.Standardize("Values").GetColumn<float>("Values");

        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(resultCol.IsNull(i), Is.EqualTo(data[i] == null));
            if (data[i] is float v) Assert.That(resultCol[i], Is.EqualTo(v));
        }
    }

    [Test]
    public void Standardize_NullFrame_ThrowsArgumentNull()
    {
        NivaraFrame? nullFrame = null;

        Assert.Throws<ArgumentNullException>(() => nullFrame!.Standardize());
        Assert.Throws<ArgumentNullException>(() => nullFrame!.Normalize());
    }

    [Test]
    public void Normalize_ExplicitUnsupportedColumn_ThrowsNotSupported()
    {
        var frame = NivaraFrame.Create(("Text", NivaraColumn<string>.Create(new string[] { "a", "b", "c" })));

        Assert.Throws<NotSupportedException>(() => frame.Normalize("Text"));
        Assert.Throws<NotSupportedException>(() => frame.Standardize("Text"));
    }

    [Test]
    public void Normalize_AutoSelect_NormalizesAllNumericColumns()
    {
        var frame = NivaraFrame.Create(new (string, IColumn)[]
        {
            ("Num", NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f })),
            ("Count", NivaraColumn<int>.Create(new int[] { 10, 20, 30 })),
        });

        var result = frame.Normalize();

        AssertNormalizedFloatColumn(result, "Num");
        AssertNormalizedDoubleColumn(result, "Count");
    }

    [Test]
    public void Normalize_IntColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_LongColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new long[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_ShortColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new short[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_ByteColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_DecimalColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new decimal[] { 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m, 9m, 10m });

    [Test]
    public void Normalize_HalfColumn_ProducesZeroMeanUnitVariance()
    {
        var values = new Half[] { (Half)1, (Half)2, (Half)3, (Half)4, (Half)5, (Half)6, (Half)7, (Half)8, (Half)9, (Half)10 };
        var frame = NivaraFrame.Create(("Values", NivaraColumn<Half>.Create(values)));

        var col = frame.Normalize("Values").GetColumn<Half>("Values");
        var floats = col.ToArray().Select(x => (float)x).ToArray();

        Assert.That(floats.Average(), Is.EqualTo(0f).Within(1e-2f));
        Assert.That(Math.Sqrt(floats.Select(x => Math.Pow(x - floats.Average(), 2)).Average()), Is.EqualTo(1f).Within(1e-2f));
    }

    [Test]
    public void Standardize_IntColumnWithNulls_SkipsNulls_PreservesNullMask()
    {
        var data = new int?[] { 1, 2, null, 4, 5 };
        var frame = NivaraFrame.Create(("Values", NivaraColumn.CreateFromNullable(data)));

        var resultCol = frame.Standardize("Values").GetColumn<double>("Values");

        var nonNull = new List<double>();
        for (int i = 0; i < data.Length; i++)
        {
            Assert.That(resultCol.IsNull(i), Is.EqualTo(data[i] == null));
            if (data[i].HasValue) nonNull.Add(resultCol[i]);
        }

        Assert.That(nonNull.Average(), Is.EqualTo(0d).Within(1e-9));
        Assert.That(Math.Sqrt(nonNull.Select(x => Math.Pow(x - nonNull.Average(), 2)).Average()), Is.EqualTo(1d).Within(1e-9));
    }

    [Test]
    public void Normalize_UIntColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new uint[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_UShortColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new ushort[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_SByteColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new sbyte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_NIntColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new nint[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });

    [Test]
    public void Normalize_NUIntColumn_ProducesZeroMeanUnitVariance()
        => AssertIntegerColumnNormalizes(new nuint[] { 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u });

    [Test]
    public void Normalize_DoubleColumn_ProducesZeroMeanUnitVariance()
    {
        var data = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var frame = NivaraFrame.Create(("Values", NivaraColumn<double>.Create(data)));

        AssertNormalizedDoubleColumn(frame.Normalize("Values"), "Values");
    }

    [Test]
    public void Normalize_NFloatColumn_ProducesZeroMeanUnitVariance()
    {
        var data = new NFloat[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f };
        var frame = NivaraFrame.Create(("Values", NivaraColumn<NFloat>.Create(data)));

        var col = frame.Normalize("Values").GetColumn<NFloat>("Values");
        var floats = col.ToArray().Select(x => (float)x).ToArray();

        Assert.That(floats.Average(), Is.EqualTo(0f).Within(1e-5f));
        Assert.That(Math.Sqrt(floats.Select(x => Math.Pow(x - floats.Average(), 2)).Average()), Is.EqualTo(1f).Within(1e-5f));
    }

    [Test]
    public void Standardize_IntColumnZeroVariance_LeavesValuesUnchanged()
    {
        var data = new int[] { 5, 5, 5 };
        var frame = NivaraFrame.Create(("Values", NivaraColumn<int>.Create(data)));

        var resultCol = frame.Standardize("Values").GetColumn<int>("Values");

        Assert.That(resultCol.ToArray(), Is.EqualTo(new[] { 5, 5, 5 }));
    }

    [Test]
    public void Normalize_ExplicitUnsupportedTypes_ThrowNotSupported()
    {
        var frame = NivaraFrame.Create(new (string, IColumn)[]
        {
            ("Flag", NivaraColumn<bool>.Create(new bool[] { true, false, true })),
            ("When", NivaraColumn<DateTime>.Create(new DateTime[] { DateTime.Now, DateTime.Now.AddDays(1), DateTime.Now.AddDays(2) })),
        });

        Assert.Throws<NotSupportedException>(() => frame.Normalize("Flag"));
        Assert.Throws<NotSupportedException>(() => frame.Standardize("Flag"));
        Assert.Throws<NotSupportedException>(() => frame.Normalize("When"));
        Assert.Throws<NotSupportedException>(() => frame.Standardize("When"));
    }

    [Test]
    public void Normalize_CrossFamilyParity_FloatAndIntMatch()
    {
        var intData = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
        var floatData = new float[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f };
        var intFrame = NivaraFrame.Create(("Values", NivaraColumn<int>.Create(intData)));
        var floatFrame = NivaraFrame.Create(("Values", NivaraColumn<float>.Create(floatData)));

        var fromInt = intFrame.Normalize("Values").GetColumn<double>("Values");
        var fromFloat = floatFrame.Normalize("Values").GetColumn<float>("Values");

        for (int i = 0; i < intData.Length; i++)
            Assert.That(fromInt[i], Is.EqualTo((double)fromFloat[i]).Within(1e-5));
    }

    private static void AssertNormalizedFloatColumn(NivaraFrame frame, string columnName)
    {
        var col = frame.GetColumn<float>(columnName);
        var values = col.ToArray();
        Assert.That(values.Average(), Is.EqualTo(0f).Within(1e-6f));
        Assert.That(Math.Sqrt(values.Select(x => Math.Pow(x - values.Average(), 2)).Average()), Is.EqualTo(1f).Within(1e-6f));
    }

    private static void AssertNormalizedDoubleColumn(NivaraFrame frame, string columnName)
    {
        var col = frame.GetColumn<double>(columnName);
        var values = col.ToArray();
        Assert.That(values.Average(), Is.EqualTo(0d).Within(1e-9));
        Assert.That(Math.Sqrt(values.Select(x => Math.Pow(x - values.Average(), 2)).Average()), Is.EqualTo(1d).Within(1e-9));
    }

    private static void AssertIntegerColumnNormalizes<T>(T[] data)
        where T : struct, INumber<T>
    {
        var frame = NivaraFrame.Create(("Values", NivaraColumn<T>.Create(data)));

        AssertNormalizedDoubleColumn(frame.Normalize("Values"), "Values");
    }
}
