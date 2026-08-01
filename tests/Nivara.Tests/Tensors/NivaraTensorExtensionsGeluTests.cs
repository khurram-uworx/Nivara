using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.Tensors;

/// <summary>
/// Direct coverage for the public column-level exact-GELU extensions added to
/// NivaraTensorExtensions: GeluExact() forward and GeluExactGradient() VJP.
/// These wrap the SIMD Abramowitz-Stegun erf kernel and carry their own
/// null-mask propagation paths (OR semantics), distinct from the AutoDiff op.
/// </summary>
[TestFixture]
public class NivaraTensorExtensionsGeluTests
{
    static double Erf(double x)
    {
        if (x < 0) return -Erf(-x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        return 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
    }

    static float GeluExactRef(float x)
    {
        double v = x;
        return (float)(v * 0.5 * (1.0 + Erf(v * 0.7071067811865475)));
    }

    static float GeluExactGradRef(float x)
    {
        double v = x;
        double cdf = 0.5 * (1.0 + Erf(v * 0.7071067811865475));
        double pdf = Math.Exp(-0.5 * v * v) * 0.3989422804014327;
        return (float)((cdf + v * pdf));
    }

    static void AssertSequenceEqual(NivaraColumn<float> actual, float[] expected, float tolerance, string label)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), $"{label}: length mismatch");
        for (int i = 0; i < expected.Length; i++)
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(tolerance), $"{label}: mismatch at {i}");
    }

    [Test]
    public void GeluExact_NoNulls_MatchesErfReference()
    {
        var values = new[] { -3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.5f, 3.0f, -0.25f };
        var column = NivaraColumn<float>.Create(values);

        var result = column.GeluExact();

        Assert.That(result.HasNulls, Is.False);
        for (int i = 0; i < values.Length; i++)
            Assert.That(result[i], Is.EqualTo(GeluExactRef(values[i])).Within(1e-5f));
    }

    [Test]
    public void GeluExact_VectorLengthBoundary_NegativeInputs_MatchesReference()
    {
        var values = new float[100];
        for (int i = 0; i < values.Length; i++)
            values[i] = -3.0f + (i * 0.09f);
        var column = NivaraColumn<float>.Create(values);

        var result = column.GeluExact();

        for (int i = 0; i < values.Length; i++)
            Assert.That(result[i], Is.EqualTo(GeluExactRef(values[i])).Within(1e-5f),
                $"negative-input vector path mismatch at {i}");
    }

    [Test]
    public void GeluExact_Double_MatchesErfReference()
    {
        var values = new[] { -2.7, 0.0, 0.3, 1.9 };
        var column = NivaraColumn<double>.Create(values);

        var result = column.GeluExact();

        for (int i = 0; i < values.Length; i++)
        {
            double v = values[i];
            double expected = v * 0.5 * (1.0 + Erf(v * 0.7071067811865475));
            Assert.That(result[i], Is.EqualTo(expected).Within(1e-12));
        }
    }

    [Test]
    public void GeluExact_WithNulls_PropagatesNullMaskAndComputesValues()
    {
        var values = new float?[] { -1.0f, null, 0.5f, null, 2.0f };
        var column = NivaraColumn<float>.CreateFromNullable(values);

        var result = column.GeluExact();

        for (int i = 0; i < values.Length; i++)
        {
            Assert.That(result.IsNull(i), Is.EqualTo(values[i] == null), $"mask mismatch at {i}");
            if (values[i] is { } v)
                Assert.That(result[i], Is.EqualTo(GeluExactRef(v)).Within(1e-5f));
        }
    }

    [Test]
    public void GeluExactGradient_NoNulls_MatchesErfReference()
    {
        var input = NivaraColumn<float>.Create(new[] { -3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.5f, -0.25f });
        var gradOutput = NivaraColumn<float>.Create(new[] { 1f, 2f, 3f, 4f, 5f, 6f, 7f });

        var result = input.GeluExactGradient(gradOutput);

        Assert.That(result.HasNulls, Is.False);
        for (int i = 0; i < input.Length; i++)
            Assert.That(result[i], Is.EqualTo(GeluExactGradRef(input[i]) * (i + 1f)).Within(1e-5f));
    }

    [Test]
    public void GeluExactGradient_WithNulls_MergesInputAndGradMasks()
    {
        var input = NivaraColumn<float>.CreateFromNullable(new float?[] { -1.0f, null, 0.5f, 2.0f });
        var gradOutput = NivaraColumn<float>.CreateFromNullable(new float?[] { 1f, 1f, null, 1f });

        var result = input.GeluExactGradient(gradOutput);

        // Mask OR semantics: index 1 (input null), index 2 (gradOutput null).
        Assert.That(result.IsNull(0), Is.False);
        Assert.That(result.IsNull(1), Is.True);
        Assert.That(result.IsNull(2), Is.True);
        Assert.That(result.IsNull(3), Is.False);
        Assert.That(result[0], Is.EqualTo(GeluExactGradRef(-1.0f)).Within(1e-5f));
        Assert.That(result[3], Is.EqualTo(GeluExactGradRef(2.0f)).Within(1e-5f));
    }
}
