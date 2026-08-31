using NUnit.Framework;
using System.Numerics;
using System.Numerics.Tensors;
using Nivara.Helpers;
using Nivara.Primitives;
using Nivara.Tensors;

namespace Nivara.Tests.Primitives;

[TestFixture]
public class WidenPrimitivesPhase1Tests
{
    static bool WidenEnabled
    {
        get
        {
            var p = typeof(NivaraPrimitives).GetProperty(nameof(NivaraPrimitives.UseWidenSimd))!;
            return (bool)p.GetValue(null)!;
        }
        set
        {
            var p = typeof(NivaraPrimitives).GetProperty(nameof(NivaraPrimitives.UseWidenSimd))!;
            p.SetValue(null, value);
        }
    }

    static void WithWidenEnabled(Action test)
    {
        var prev = WidenEnabled;
        WidenEnabled = true;
        try { test(); }
        finally { WidenEnabled = prev; }
    }

    static void WithWidenDisabled(Action test)
    {
        var prev = WidenEnabled;
        WidenEnabled = false;
        try { test(); }
        finally { WidenEnabled = prev; }
    }

    static void AssertElementwiseEqual(ReadOnlySpan<Half> expected, ReadOnlySpan<Half> actual)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
        {
            float e = (float)expected[i];
            float a = (float)actual[i];
            Assert.That(a, Is.EqualTo(e).Within(MathF.Abs(e) * 1e-2f + 1e-3f), $"Half mismatch at {i}");
        }
    }

    static void AssertElementwiseEqual(ReadOnlySpan<BFloat16> expected, ReadOnlySpan<BFloat16> actual)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        for (int i = 0; i < expected.Length; i++)
        {
            float e = (float)expected[i];
            float a = (float)actual[i];
            Assert.That(a, Is.EqualTo(e).Within(MathF.Abs(e) * 1e-2f + 1e-3f), $"BFloat16 mismatch at {i}");
        }
    }

    [Test]
    public void Dot_BFloat16_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 256;
            Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(n), Is.True,
                "test must genuinely exercise the widen path, not the scalar fallback");
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();

            BFloat16 expected = TensorPrimitives.Dot(a, b);
            BFloat16 actual = WidenPrimitives.Dot<BFloat16>(a, b);

            Assert.That((float)actual, Is.EqualTo((float)expected).Within(MathF.Abs((float)expected) * 1e-2f + 1e-3f));
        });
    }

    [Test]
    public void Dot_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();

            Half expected = TensorPrimitives.Dot(a, b);
            Half actual = WidenPrimitives.Dot<Half>(a, b);

            Assert.That((float)actual, Is.EqualTo((float)expected).Within(MathF.Abs((float)expected) * 1e-2f + 1e-3f));
        });
    }

    [Test]
    public void Dot_BFloat16_ToggleOff_MatchesTensorPrimitives()
    {
        WithWidenDisabled(() =>
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();

            BFloat16 expected = TensorPrimitives.Dot(a, b);
            BFloat16 actual = WidenPrimitives.Dot<BFloat16>(a, b);

            Assert.That((float)actual, Is.EqualTo((float)expected));
        });
    }

    [Test]
    public void Dot_Half_ToggleOff_MatchesTensorPrimitives()
    {
        WithWidenDisabled(() =>
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();

            Half expected = TensorPrimitives.Dot(a, b);
            Half actual = WidenPrimitives.Dot<Half>(a, b);

            Assert.That((float)actual, Is.EqualTo((float)expected));
        });
    }

    [Test]
    public void Dot_Float_ToggleOn_PassesThroughToTensorPrimitives()
    {
        WithWidenEnabled(() =>
        {
            int n = 256;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();

            float expected = TensorPrimitives.Dot(a, b);
            float actual = WidenPrimitives.Dot<float>(a, b);

            Assert.That(actual, Is.EqualTo(expected));
        });
    }

    [Test]
    public void Dot_BelowThreshold_FallsBackToScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 16;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();

            BFloat16 expected = TensorPrimitives.Dot(a, b);
            BFloat16 actual = WidenPrimitives.Dot<BFloat16>(a, b);

            Assert.That((float)actual, Is.EqualTo((float)expected));
        });
    }

    [Test]
    [TestCase(typeof(BFloat16))]
    [TestCase(typeof(Half))]
    public void Add_BFloat16_ToggleOn_MatchesScalar(Type type)
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            Assert.That(WidenPrimitives.ShouldWiden(type, n), Is.True,
                "test must genuinely exercise the widen path, not the scalar fallback");
            var rng = new Random(42);

            if (type == typeof(BFloat16))
            {
                var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
                var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
                var expected = new BFloat16[n];
                TensorPrimitives.Add(a, b, expected);
                var actual = new BFloat16[n];
                WidenPrimitives.Add<BFloat16>(a, b, actual);
                AssertElementwiseEqual(expected, actual);
            }
            else
            {
                var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
                var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
                var expected = new Half[n];
                TensorPrimitives.Add(a, b, expected);
                var actual = new Half[n];
                WidenPrimitives.Add<Half>(a, b, actual);
                AssertElementwiseEqual(expected, actual);
            }
        });
    }

    [Test]
    public void Subtract_BFloat16_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expected = new BFloat16[n];
            TensorPrimitives.Subtract(a, b, expected);
            var actual = new BFloat16[n];
            WidenPrimitives.Subtract<BFloat16>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void Multiply_BFloat16_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var expected = new BFloat16[n];
            TensorPrimitives.Multiply(a, b, expected);
            var actual = new BFloat16[n];
            WidenPrimitives.Multiply<BFloat16>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void Divide_BFloat16_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 + 0.5)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 + 0.5)).ToArray();
            var expected = new BFloat16[n];
            TensorPrimitives.Divide(a, b, expected);
            var actual = new BFloat16[n];
            WidenPrimitives.Divide<BFloat16>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void Add_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expected = new Half[n];
            TensorPrimitives.Add(a, b, expected);
            var actual = new Half[n];
            WidenPrimitives.Add<Half>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void Subtract_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expected = new Half[n];
            TensorPrimitives.Subtract(a, b, expected);
            var actual = new Half[n];
            WidenPrimitives.Subtract<Half>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void Multiply_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var expected = new Half[n];
            TensorPrimitives.Multiply(a, b, expected);
            var actual = new Half[n];
            WidenPrimitives.Multiply<Half>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void Divide_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 + 0.5)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 + 0.5)).ToArray();
            var expected = new Half[n];
            TensorPrimitives.Divide(a, b, expected);
            var actual = new Half[n];
            WidenPrimitives.Divide<Half>(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void ToggleOff_Elementwise_MatchTensorPrimitives()
    {
        WithWidenDisabled(() =>
        {
            int n = 200;
            var rng = new Random(42);

            var ab = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var bb = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expB = new BFloat16[n];
            var actB = new BFloat16[n];
            TensorPrimitives.Add(ab, bb, expB);
            WidenPrimitives.Add<BFloat16>(ab, bb, actB);
            AssertElementwiseEqual(expB, actB);
            TensorPrimitives.Subtract(ab, bb, expB);
            WidenPrimitives.Subtract<BFloat16>(ab, bb, actB);
            AssertElementwiseEqual(expB, actB);
            TensorPrimitives.Multiply(ab, bb, expB);
            WidenPrimitives.Multiply<BFloat16>(ab, bb, actB);
            AssertElementwiseEqual(expB, actB);
            TensorPrimitives.Divide(ab, bb, expB);
            WidenPrimitives.Divide<BFloat16>(ab, bb, actB);
            AssertElementwiseEqual(expB, actB);

            var ah = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var bh = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expH = new Half[n];
            var actH = new Half[n];
            TensorPrimitives.Add(ah, bh, expH);
            WidenPrimitives.Add<Half>(ah, bh, actH);
            AssertElementwiseEqual(expH, actH);
            TensorPrimitives.Subtract(ah, bh, expH);
            WidenPrimitives.Subtract<Half>(ah, bh, actH);
            AssertElementwiseEqual(expH, actH);
            TensorPrimitives.Multiply(ah, bh, expH);
            WidenPrimitives.Multiply<Half>(ah, bh, actH);
            AssertElementwiseEqual(expH, actH);
            TensorPrimitives.Divide(ah, bh, expH);
            WidenPrimitives.Divide<Half>(ah, bh, actH);
            AssertElementwiseEqual(expH, actH);
        });
    }

    [Test]
    public void NumericTensorKernels_Add_BFloat16_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expected = new BFloat16[n];
            TensorPrimitives.Add(a, b, expected);
            var actual = new BFloat16[n];
            NumericTensorKernels<BFloat16>.Add(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void NumericTensorKernels_Multiply_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var expected = new Half[n];
            TensorPrimitives.Multiply(a, b, expected);
            var actual = new Half[n];
            NumericTensorKernels<Half>.Multiply(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void NumericTensorKernels_Subtract_BFloat16_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (BFloat16)(float)(rng.NextDouble() * 4 - 2)).ToArray();
            var expected = new BFloat16[n];
            TensorPrimitives.Subtract(a, b, expected);
            var actual = new BFloat16[n];
            NumericTensorKernels<BFloat16>.Subtract(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void NumericTensorKernels_Divide_Half_ToggleOn_MatchesScalar()
    {
        WithWidenEnabled(() =>
        {
            int n = 200;
            var rng = new Random(42);
            var a = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 + 0.5)).ToArray();
            var b = Enumerable.Range(0, n).Select(_ => (Half)(float)(rng.NextDouble() * 4 + 0.5)).ToArray();
            var expected = new Half[n];
            TensorPrimitives.Divide(a, b, expected);
            var actual = new Half[n];
            NumericTensorKernels<Half>.Divide(a, b, actual);
            AssertElementwiseEqual(expected, actual);
        });
    }

    [Test]
    public void MultiplyCore_BFloat16_MatMul_ToggleOn()
    {
        WithWidenEnabled(() =>
        {
            int rows = 3, cols = 256, k = 5;
            Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(cols), Is.True,
                "matmul must genuinely exercise the widen path, not the scalar fallback");
            var rng = new Random(42);
            var a = Enumerable.Range(0, rows * cols).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, cols * k).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var result = new BFloat16[rows * k];

            TensorsHelper.MultiplyCore(a, b, result, rows, cols, k);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < k; j++)
                {
                    float expected = 0;
                    for (int p = 0; p < cols; p++)
                        expected += (float)a[i * cols + p] * (float)b[p * k + j];
                    Assert.That((float)result[i * k + j], Is.EqualTo(expected).Within(MathF.Abs(expected) * 0.02f + 0.01f),
                        $"Mismatch at [{i},{j}]");
                }
        });
    }

    [Test]
    public void MultiplyCore_Half_MatMul_ToggleOn()
    {
        WithWidenEnabled(() =>
        {
            int rows = 3, cols = 256, k = 5;
            Assert.That(WidenPrimitives.ShouldWiden<Half>(cols), Is.True,
                "matmul must genuinely exercise the widen path, not the scalar fallback");
            var rng = new Random(42);
            var a = Enumerable.Range(0, rows * cols).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, cols * k).Select(_ => (Half)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var result = new Half[rows * k];

            TensorsHelper.MultiplyCore(a, b, result, rows, cols, k);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < k; j++)
                {
                    float expected = 0;
                    for (int p = 0; p < cols; p++)
                        expected += (float)a[i * cols + p] * (float)b[p * k + j];
                    Assert.That((float)result[i * k + j], Is.EqualTo(expected).Within(MathF.Abs(expected) * 0.02f + 0.01f),
                        $"Mismatch at [{i},{j}]");
                }
        });
    }

    [Test]
    public void MultiplyCore_BFloat16_MatMul_ToggleOff_MatchScalarDot()
    {
        WithWidenDisabled(() =>
        {
            int rows = 3, cols = 256, k = 5;
            Assert.That(WidenPrimitives.ShouldWiden<BFloat16>(cols), Is.False,
                "toggle off must disable the widen path");
            var rng = new Random(42);
            var a = Enumerable.Range(0, rows * cols).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var b = Enumerable.Range(0, cols * k).Select(_ => (BFloat16)(float)(rng.NextDouble() * 2 - 1)).ToArray();
            var result = new BFloat16[rows * k];

            TensorsHelper.MultiplyCore(a, b, result, rows, cols, k);

            for (int j = 0; j < k; j++)
            {
                var col = new BFloat16[cols];
                for (int p = 0; p < cols; p++)
                    col[p] = b[p * k + j];
                for (int i = 0; i < rows; i++)
                {
                    BFloat16 expected = TensorPrimitives.Dot<BFloat16>(a.AsSpan(i * cols, cols), col);
                    Assert.That(result[i * k + j], Is.EqualTo(expected),
                        $"Scalar dot mismatch at [{i},{j}]");
                }
            }
        });
    }

    [Test]
    public void MultiplyCore_Float_Unchanged()
    {
        int rows = 3, cols = 4, k = 5;
        var rng = new Random(42);
        var a = Enumerable.Range(0, rows * cols).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var b = Enumerable.Range(0, cols * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        var result = new float[rows * k];

        TensorsHelper.MultiplyCore(a, b, result, rows, cols, k);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < k; j++)
            {
                float expected = 0;
                for (int p = 0; p < cols; p++)
                    expected += a[i * cols + p] * b[p * k + j];
                Assert.That(result[i * k + j], Is.EqualTo(expected).Within(1e-4f));
            }
    }
}
