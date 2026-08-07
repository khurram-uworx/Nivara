using NUnit.Framework;

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
        var frame = NivaraFrame.Create(("Values", NivaraColumn<float>.CreateFromNullable(data)));

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

        // Unsupported numeric and non-numeric columns are left untouched
        Assert.That(result.GetColumn<int>("Count").ToArray(), Is.EqualTo(new[] { 10, 20, 30 }));
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
        var frame = NivaraFrame.Create(("Values", NivaraColumn<float>.CreateFromNullable(data)));

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
        var frame = NivaraFrame.Create(("Count", NivaraColumn<int>.Create(new int[] { 1, 2, 3 })));

        Assert.Throws<NotSupportedException>(() => frame.Normalize("Count"));
        Assert.Throws<NotSupportedException>(() => frame.Standardize("Count"));
    }

    [Test]
    public void Normalize_AutoSelect_SkipsUnsupportedNumericColumns()
    {
        var frame = NivaraFrame.Create(new (string, IColumn)[]
        {
            ("Num", NivaraColumn<float>.Create(new float[] { 1f, 2f, 3f })),
            ("Count", NivaraColumn<int>.Create(new int[] { 10, 20, 30 })),
        });

        var result = frame.Normalize();

        Assert.That(result.GetColumn<int>("Count").ToArray(), Is.EqualTo(new[] { 10, 20, 30 }));
        Assert.That(result.GetColumn<float>("Num").ToArray().Average(), Is.EqualTo(0f).Within(1e-6f));
    }
}
