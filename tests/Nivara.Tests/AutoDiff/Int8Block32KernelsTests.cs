using Nivara.AutoDiff.Nn;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

public class Int8Block32KernelsTests
{
    [Test]
    public void MatMulTransposedB_MatchesDenseFloatReference()
    {
        const int aRows = 2;
        const int aCols = 40;
        const int bCols = 7;

        var rng = new Random(42);
        var a = new float[aRows * aCols];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);

        // dense float weights [bCols, aCols]
        var dense = new float[bCols * aCols];
        for (int i = 0; i < dense.Length; i++) dense[i] = (float)(rng.NextDouble() * 2 - 1);

        var (qw, qs) = Int8Block32Quantizer.QuantizeFromCheckpoint(dense, bCols, aCols);
        var w = Int8Block32Quantizer.ToWeight(qw, qs, aCols, bCols);

        var expected = new float[aRows * bCols];
        // dense reference: a @ dense^T
        for (int r = 0; r < aRows; r++)
            for (int c = 0; c < bCols; c++)
            {
                float acc = 0f;
                for (int k = 0; k < aCols; k++) acc += a[r * aCols + k] * dense[c * aCols + k];
                expected[r * bCols + c] = acc;
            }

        var actual = new float[aRows * bCols];
        Int8Block32Kernels.MatMulTransposedB(a, w, actual, aRows, aCols, bCols);

        for (int i = 0; i < expected.Length; i++)
        {
            // quantization introduces error; allow relative tolerance
            Assert.That(actual[i], Is.EqualTo(expected[i]).Within(Math.Max(1e-2f, Math.Abs(expected[i]) * 0.05f)),
                $"index {i}: actual={actual[i]} expected={expected[i]}");
        }
    }

    [Test]
    public void MatMulTransposedB_QuantizationError_Bounded()
    {
        // single row, single output, verify dequant round-trip error is small
        const int aCols = 32;
        const int bCols = 1;
        var a = new float[aCols];
        for (int i = 0; i < aCols; i++) a[i] = 1f;
        var dense = new float[bCols * aCols];
        for (int i = 0; i < dense.Length; i++) dense[i] = 0.5f;

        var (qw, qs) = Int8Block32Quantizer.QuantizeFromCheckpoint(dense, bCols, aCols);
        var w = Int8Block32Quantizer.ToWeight(qw, qs, aCols, bCols);

        var actual = new float[bCols];
        Int8Block32Kernels.MatMulTransposedB(a, w, actual, 1, aCols, bCols);

        // expected: 32 * (0.5 * 1.0) = 16
        Assert.That(actual[0], Is.EqualTo(16f).Within(0.2f));
    }
}
