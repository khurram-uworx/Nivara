using Nivara.AutoDiff.Operations;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
/// <summary>
/// Kernel correctness for the mixed-precision matmul added in #387:
/// F32 activations × BF16/Half weights, widening inside the kernel.
/// </summary>
public class MixedPrecisionMatmulTests
{
    [Test]
    public void MatMulTransposedB_Bf16Weights_KnownValues()
    {
        // A = [[1,2,3],[4,5,6]] (2x3), W (as BF16) = [[7,8,9],[10,11,12]] (2x3)
        // result[i,j] = sum_k A[i,k]*W[j,k]
        var a = new[] { 1f, 2f, 3f, 4f, 5f, 6f };
        var w = Array.ConvertAll(new[] { 7f, 8f, 9f, 10f, 11f, 12f }, x => (BFloat16)x);
        var actual = new float[4];
        GradKernels.MatMulTransposedB(a, w, actual, 2, 3, 2);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
            {
                float expected = 0f;
                for (int k = 0; k < 3; k++)
                    expected += a[i * 3 + k] * (float)w[j * 3 + k];
                Assert.That(actual[i * 2 + j], Is.EqualTo(expected).Within(1e-5f), $"Index {i},{j}");
            }
    }

    [Test]
    public void MatMulTransposedB_Bf16Weights_MatchesFloatWidenedReference()
    {
        var rng = new Random(1234);
        (int Rows, int Cols, int BCols)[] shapes = [(1, 1, 1), (2, 3, 2), (4, 7, 5), (8, 16, 8)];
        foreach (var (rows, cols, bcols) in shapes)
        {
            var a = new float[rows * cols];
            var wF = new float[cols * bcols];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < wF.Length; i++) wF[i] = (float)(rng.NextDouble() * 2 - 1);
            var wBf = Array.ConvertAll(wF, x => (BFloat16)x);

            var actual = new float[rows * bcols];
            GradKernels.MatMulTransposedB(a, wBf, actual, rows, cols, bcols);

            // reference: widen BF16 -> float, then float matmul
            var wWidened = Array.ConvertAll(wBf, x => (float)x);
            var expected = new float[rows * bcols];
            GradKernels.MatMulTransposedB<float>(a, wWidened, expected, rows, cols, bcols);

            for (int i = 0; i < actual.Length; i++)
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f), $"shape {rows}x{cols}x{bcols} idx {i}");
        }
    }

    [Test]
    public void MatMulTransposedB_HalfWeights_MatchesFloatWidenedReference()
    {
        var rng = new Random(5678);
        (int Rows, int Cols, int BCols)[] shapes = [(1, 1, 1), (2, 3, 2), (4, 7, 5)];
        foreach (var (rows, cols, bcols) in shapes)
        {
            var a = new float[rows * cols];
            var wF = new float[cols * bcols];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < wF.Length; i++) wF[i] = (float)(rng.NextDouble() * 2 - 1);
            var wH = Array.ConvertAll(wF, x => (Half)x);

            var actual = new float[rows * bcols];
            GradKernels.MatMulTransposedB(a, wH, actual, rows, cols, bcols);

            var wWidened = Array.ConvertAll(wH, x => (float)x);
            var expected = new float[rows * bcols];
            GradKernels.MatMulTransposedB<float>(a, wWidened, expected, rows, cols, bcols);

            for (int i = 0; i < actual.Length; i++)
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-3f), $"shape {rows}x{cols}x{bcols} idx {i}");
        }
    }

    [Test]
    public void MatMul_Bf16Weights_MatchesFloatWidenedReference()
    {
        var rng = new Random(9999);
        (int Rows, int Cols, int BCols)[] shapes = [(2, 3, 2), (4, 5, 3)];
        foreach (var (rows, cols, bcols) in shapes)
        {
            var a = new float[rows * cols];
            var wF = new float[cols * bcols];
            for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
            for (int i = 0; i < wF.Length; i++) wF[i] = (float)(rng.NextDouble() * 2 - 1);
            var wBf = Array.ConvertAll(wF, x => (BFloat16)x);

            var actual = new float[rows * bcols];
            GradKernels.MatMul(a, wBf, actual, rows, cols, bcols);

            var wWidened = Array.ConvertAll(wBf, x => (float)x);
            var expected = new float[rows * bcols];
            // plain matmul a @ wWidened
            for (int i = 0; i < rows; i++)
                for (int n = 0; n < bcols; n++)
                {
                    float s = 0f;
                    for (int k = 0; k < cols; k++) s += a[i * cols + k] * wWidened[k * bcols + n];
                    expected[i * bcols + n] = s;
                }

            for (int i = 0; i < actual.Length; i++)
                Assert.That(actual[i], Is.EqualTo(expected[i]).Within(1e-5f), $"idx {i}");
        }
    }
}
