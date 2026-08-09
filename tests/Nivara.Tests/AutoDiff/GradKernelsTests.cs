using Nivara.AutoDiff.Operations;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Known-value tests for the span-based <see cref="GradKernels"/>, asserting
/// against scalar mathematical formulas (independent of the column extension
/// kernels they replaced).
/// </summary>
[TestFixture]
public class GradKernelsTests
{
    static readonly float[] FloatInputs = { -3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.5f, 3.0f, -0.25f };
    static readonly double[] DoubleInputs = { -3.5, -1.0, 0.0, 0.5, 1.0, 2.5, 3.0, -0.25 };

    static T ErfOf<T>(T x) where T : struct, IFloatingPointIeee754<T>
        => T.CreateChecked(Erf(double.CreateChecked(x)));

    static double Erf(double x)
    {
        double az = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * az);
        double p = 1.061405429 * t - 1.453152027;
        p = p * t + 1.421413741;
        p = p * t - 0.284496736;
        p = p * t + 0.254829592;
        double erfAbs = 1.0 - p * t * Math.Exp(-az * az);
        return x < 0.0 ? -erfAbs : erfAbs;
    }

    static void AssertKernel<T>(
        T[] input,
        Func<T, T> expected,
        Action<ReadOnlySpan<T>, Span<T>> kernel,
        double tolerance = 1e-5) where T : struct, IFloatingPointIeee754<T>
    {
        var actual = new T[input.Length];
        kernel(input, actual);
        for (int i = 0; i < input.Length; i++)
            Assert.That(double.CreateChecked(actual[i]),
                Is.EqualTo(double.CreateChecked(expected(input[i]))).Within(tolerance),
                $"Index {i}");
    }

    static void AssertGradientKernel<T>(
        T[] input,
        T[] gradOutput,
        Func<T, T, T> expected,
        Action<ReadOnlySpan<T>, ReadOnlySpan<T>, Span<T>> kernel,
        double tolerance = 1e-5) where T : struct, IFloatingPointIeee754<T>
    {
        var actual = new T[input.Length];
        kernel(input, gradOutput, actual);
        for (int i = 0; i < input.Length; i++)
            Assert.That(double.CreateChecked(actual[i]),
                Is.EqualTo(double.CreateChecked(expected(input[i], gradOutput[i]))).Within(tolerance),
                $"Index {i}");
    }

    [Test]
    public void Sigmoid_Float_KnownValues()
    {
        AssertKernel(FloatInputs, x => 1f / (1f + MathF.Exp(-x)), GradKernels.Sigmoid<float>);
    }

    [Test]
    public void Sigmoid_Double_KnownValues()
    {
        AssertKernel(DoubleInputs, x => 1.0 / (1.0 + Math.Exp(-x)), GradKernels.Sigmoid<double>);
    }

    [Test]
    public void SigmoidGradient_Float_KnownValues()
    {
        var input = new[] { 0.7310585786300049f, 0.2689414213699951f, 0.5f, 0.6224593312018546f, 0.7310585786300049f, 0.9241418199787566f, 0.9525741268224334f, 0.43782349911420193f };
        var grad = new[] { 1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.25f, 2.0f, -0.5f };
        AssertGradientKernel(input, grad, (s, g) => s * (1f - s) * g, GradKernels.SigmoidGradient<float>);
    }

    [Test]
    public void Tanh_Float_KnownValues()
    {
        AssertKernel(FloatInputs, MathF.Tanh, GradKernels.Tanh<float>);
    }

    [Test]
    public void Tanh_Double_KnownValues()
    {
        AssertKernel(DoubleInputs, Math.Tanh, GradKernels.Tanh<double>);
    }

    [Test]
    public void TanhGradient_Float_KnownValues()
    {
        var input = new[] { -0.9981778976111987f, -0.7615941559557649f, 0f, 0.46211715726000974f, 0.7615941559557649f, 0.9866142981514303f, 0.9950547536867305f, -0.24491866240370913f };
        var grad = new[] { 0.5f, 1.0f, -1.0f, 2.0f, -0.5f, 1.5f, -2.0f, 0.25f };
        AssertGradientKernel(input, grad, (t, g) => (1f - t * t) * g, GradKernels.TanhGradient<float>);
    }

    [Test]
    public void Relu_Float_KnownValues()
    {
        AssertKernel(FloatInputs, x => MathF.Max(x, 0f), GradKernels.Relu<float>);
    }

    [Test]
    public void ReluGradient_Float_KnownValues()
    {
        var grad = Enumerable.Range(1, FloatInputs.Length).Select(i => (float)i).ToArray();
        AssertGradientKernel(FloatInputs, grad, (x, g) => x > 0f ? g : 0f, GradKernels.ReluGradient<float>);
    }

    [Test]
    public void LeakyRelu_Float_KnownValues()
    {
        const float slope = 0.01f;
        AssertKernel(FloatInputs, x => x > 0f ? x : slope * x,
            (input, output) => GradKernels.LeakyRelu<float>(input, slope, output));
    }

    [Test]
    public void LeakyReluGradient_Float_KnownValues()
    {
        const float slope = 0.01f;
        var grad = Enumerable.Range(1, FloatInputs.Length).Select(i => (float)i * 0.5f).ToArray();
        AssertGradientKernel(FloatInputs, grad, (x, g) => x > 0f ? g : slope * g,
            (input, g, output) => GradKernels.LeakyReluGradient<float>(input, g, slope, output));
    }

    [Test]
    public void Exp_Float_KnownValues()
    {
        AssertKernel(FloatInputs, MathF.Exp, GradKernels.Exp<float>);
    }

    [Test]
    public void Log_Float_KnownValues()
    {
        var input = new[] { 0.5f, 1.0f, 2.0f, 3.0f, 0.25f, 10.0f, 4.0f, 1.5f };
        AssertKernel(input, MathF.Log, GradKernels.Log<float>);
    }

    [Test]
    public void LogGradient_Float_KnownValues()
    {
        var input = new[] { 0.5f, 1.0f, 2.0f, 3.0f, 0.25f, 10.0f, 4.0f, 1.5f };
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i).ToArray();
        AssertGradientKernel(input, grad, (x, g) => g / x, GradKernels.LogGradient<float>);
    }

    [Test]
    public void Abs_Float_KnownValues()
    {
        AssertKernel(FloatInputs, MathF.Abs, GradKernels.Abs<float>);
    }

    [Test]
    public void AbsGradient_Float_KnownValues()
    {
        var grad = Enumerable.Range(1, FloatInputs.Length).Select(i => (float)i * 0.25f).ToArray();
        AssertGradientKernel(FloatInputs, grad, (x, g) => MathF.Sign(x) * g, GradKernels.AbsGradient<float>);
    }

    [Test]
    public void Clamp_Float_KnownValues()
    {
        AssertKernel(FloatInputs, x => Math.Clamp(x, -1.0f, 2.0f),
            (input, output) => GradKernels.Clamp<float>(input, -1.0f, 2.0f, output));
    }

    [Test]
    public void ClipGradient_Float_KnownValues()
    {
        var grad = Enumerable.Range(1, FloatInputs.Length).Select(i => (float)i).ToArray();
        AssertGradientKernel(FloatInputs, grad, (x, g) => x >= -1.0f && x <= 2.0f ? g : 0f,
            (input, g, output) => GradKernels.ClipGradient<float>(input, g, -1.0f, 2.0f, output));
    }

    [Test]
    public void Negate_Float_KnownValues()
    {
        AssertKernel(FloatInputs, x => -x, GradKernels.Negate<float>);
    }

    [Test]
    public void Divide_Float_KnownValues()
    {
        var numerator = new[] { 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f };
        var denominator = new[] { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
        var actual = new float[numerator.Length];
        GradKernels.Divide<float>(numerator, denominator, actual);
        for (int i = 0; i < numerator.Length; i++)
            Assert.That(actual[i], Is.EqualTo(numerator[i] / denominator[i]).Within(1e-6), $"Index {i}");
    }

    static void AssertRowSoftmax(float[] input, float[] actual, int classCount, double tolerance = 1e-5)
    {
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            double max = input[r * classCount];
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, input[r * classCount + j]);

            double sum = 0.0;
            for (int j = 0; j < classCount; j++)
                sum += Math.Exp(input[r * classCount + j] - max);

            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                Assert.That(actual[idx], Is.EqualTo((float)(Math.Exp(input[idx] - max) / sum)).Within(tolerance),
                    $"Index {idx}");
            }
        }
    }

    [Test]
    public void Softmax_RowWise_KnownValues()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var actual = new float[input.Length];
        GradKernels.Softmax<float>(input, actual, 3);
        AssertRowSoftmax(input, actual, 3);
    }

    [Test]
    public void Softmax_SingleVector_KnownValues()
    {
        var input = new[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var actual = new float[input.Length];
        GradKernels.Softmax<float>(input, actual, input.Length);
        AssertRowSoftmax(input, actual, input.Length);
    }

    [Test]
    public void SoftmaxGradient_RowWise_KnownValues()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var grad = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f };
        var softmax = new float[input.Length];
        GradKernels.Softmax<float>(input, softmax, 3);
        var actual = new float[input.Length];
        GradKernels.SoftmaxGradient<float>(softmax, grad, actual, 3);

        const int classCount = 3;
        for (int r = 0; r < input.Length / classCount; r++)
        {
            double dot = 0.0;
            for (int j = 0; j < classCount; j++)
                dot += softmax[r * classCount + j] * grad[r * classCount + j];
            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                double expected = softmax[idx] * (grad[idx] - dot);
                Assert.That(actual[idx], Is.EqualTo((float)expected).Within(1e-5), $"Index {idx}");
            }
        }
    }

    [Test]
    public void LogSoftmax_RowWise_KnownValues()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var actual = new float[input.Length];
        GradKernels.LogSoftmax<float>(input, actual, 3);

        const int classCount = 3;
        for (int r = 0; r < input.Length / classCount; r++)
        {
            double max = input[r * classCount];
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, input[r * classCount + j]);
            double sum = 0.0;
            for (int j = 0; j < classCount; j++)
                sum += Math.Exp(input[r * classCount + j] - max);
            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                Assert.That(actual[idx], Is.EqualTo((float)(input[idx] - max - Math.Log(sum))).Within(1e-5),
                    $"Index {idx}");
            }
        }
    }

    [Test]
    public void LogSoftmaxGradient_RowWise_KnownValues()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var grad = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f };
        var actual = new float[input.Length];
        GradKernels.LogSoftmaxGradient<float>(input, grad, actual, 3);

        const int classCount = 3;
        for (int r = 0; r < input.Length / classCount; r++)
        {
            double max = input[r * classCount];
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, input[r * classCount + j]);
            double sumExp = 0.0;
            double sumGrad = 0.0;
            for (int j = 0; j < classCount; j++)
            {
                sumExp += Math.Exp(input[r * classCount + j] - max);
                sumGrad += grad[r * classCount + j];
            }
            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                double soft = Math.Exp(input[idx] - max) / sumExp;
                double expected = grad[idx] - soft * sumGrad;
                Assert.That(actual[idx], Is.EqualTo((float)expected).Within(1e-5), $"Index {idx}");
            }
        }
    }

    static float GeluExpected(float x)
    {
        const float sqrt2OverPi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float z = sqrt2OverPi * (x + coeff * x * x * x);
        return 0.5f * x * (1f + MathF.Tanh(z));
    }

    // ─────────────────────────────────────────────────────────────
    //  Dim-aware (strided) Softmax / LogSoftmax
    // ─────────────────────────────────────────────────────────────

    static void AssertStridedSoftmax(float[] input, float[] actual, int outer, int classCount, int inner, double tolerance = 1e-5)
    {
        int sliceLength = classCount * inner;
        for (int b = 0; b < outer; b++)
        {
            for (int o = 0; o < inner; o++)
            {
                int start = b * sliceLength + o;
                double max = input[start];
                for (int k = 1; k < classCount; k++)
                    max = Math.Max(max, input[start + k * inner]);
                double sum = 0.0;
                for (int k = 0; k < classCount; k++)
                    sum += Math.Exp(input[start + k * inner] - max);
                for (int k = 0; k < classCount; k++)
                {
                    int idx = start + k * inner;
                    Assert.That(actual[idx], Is.EqualTo((float)(Math.Exp(input[idx] - max) / sum)).Within(tolerance),
                        $"Slice ({b},{o}) index {idx}");
                }
            }
        }
    }

    [Test]
    public void SoftmaxDim_2D_Dim0_MatchesColumnWise()
    {
        // Shape [3, 4]: dim 0 → outer=1, classCount=3, inner=4 (strided by 4).
        var input = new float[]
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.5f, 0.5f, 2.5f, 3.5f,
            -1.0f, 0.0f, 1.0f, 2.0f,
        };
        var actual = new float[input.Length];
        GradKernels.SoftmaxDim<float>(input, actual, 1, 3, 4);
        AssertStridedSoftmax(input, actual, 1, 3, 4);
    }

    [Test]
    public void SoftmaxDim_3D_Dim1_MatchesExpected()
    {
        // Shape [2, 3, 4]: dim 1 → outer=2, classCount=3, inner=4.
        var input = new float[24];
        for (int i = 0; i < input.Length; i++)
            input[i] = (i % 7) - 3f;
        var actual = new float[input.Length];
        GradKernels.SoftmaxDim<float>(input, actual, 2, 3, 4);
        AssertStridedSoftmax(input, actual, 2, 3, 4);
    }

    [Test]
    public void SoftmaxDimGradient_2D_Dim0_KnownValues()
    {
        var input = new float[]
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.5f, 0.5f, 2.5f, 3.5f,
            -1.0f, 0.0f, 1.0f, 2.0f,
        };
        var grad = new float[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f, 1.1f, 1.2f, 1.3f };
        var softmax = new float[input.Length];
        GradKernels.SoftmaxDim<float>(input, softmax, 1, 3, 4);
        var actual = new float[input.Length];
        GradKernels.SoftmaxDimGradient<float>(softmax, grad, actual, 1, 3, 4);

        const int classCount = 3, inner = 4;
        for (int o = 0; o < inner; o++)
        {
            double dot = 0.0;
            for (int k = 0; k < classCount; k++)
                dot += softmax[o + k * inner] * grad[o + k * inner];
            for (int k = 0; k < classCount; k++)
            {
                int idx = o + k * inner;
                double expected = softmax[idx] * (grad[idx] - dot);
                Assert.That(actual[idx], Is.EqualTo((float)expected).Within(1e-5), $"Column {o} index {idx}");
            }
        }
    }

    [Test]
    public void LogSoftmaxDim_2D_Dim0_MatchesColumnWise()
    {
        var input = new float[]
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            1.5f, 0.5f, 2.5f, 3.5f,
            -1.0f, 0.0f, 1.0f, 2.0f,
        };
        var actual = new float[input.Length];
        GradKernels.LogSoftmaxDim<float>(input, actual, 1, 3, 4);

        const int classCount = 3, inner = 4;
        for (int o = 0; o < inner; o++)
        {
            double max = input[o];
            for (int k = 1; k < classCount; k++)
                max = Math.Max(max, input[o + k * inner]);
            double sum = 0.0;
            for (int k = 0; k < classCount; k++)
                sum += Math.Exp(input[o + k * inner] - max);
            for (int k = 0; k < classCount; k++)
            {
                int idx = o + k * inner;
                Assert.That(actual[idx], Is.EqualTo((float)(input[idx] - max - Math.Log(sum))).Within(1e-5),
                    $"Column {o} index {idx}");
            }
        }
    }

    [Test]
    public void LogSoftmaxDimGradient_3D_Dim2_KnownValues()
    {
        // Shape [2, 3, 4]: dim 2 → outer=6, classCount=4, inner=1 (contiguous fast path).
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 1.5f, 0.5f, 2.5f, 3.5f, -1.0f, 0.0f, 1.0f, 2.0f, 0.5f, -0.5f, 1.5f, 2.5f, 3.0f, 1.0f, 0.0f, -1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        var grad = new float[input.Length];
        for (int i = 0; i < grad.Length; i++)
            grad[i] = (i % 5) * 0.1f - 0.2f;
        var actual = new float[input.Length];
        GradKernels.LogSoftmaxDimGradient<float>(input, grad, actual, 6, 4, 1);

        const int classCount = 4;
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            double max = input[r * classCount];
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, input[r * classCount + j]);
            double sumExp = 0.0;
            double sumGrad = 0.0;
            for (int j = 0; j < classCount; j++)
            {
                sumExp += Math.Exp(input[r * classCount + j] - max);
                sumGrad += grad[r * classCount + j];
            }
            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                double soft = Math.Exp(input[idx] - max) / sumExp;
                Assert.That(actual[idx], Is.EqualTo((float)(grad[idx] - soft * sumGrad)).Within(1e-5), $"Row {r} index {idx}");
            }
        }
    }

    [Test]
    public void SoftmaxDim_2D_DimLast_MatchesRowWise()
    {
        var input = new float[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var viaDim = new float[input.Length];
        GradKernels.SoftmaxDim<float>(input, viaDim, 3, 3, 1);
        var viaRow = new float[input.Length];
        GradKernels.Softmax<float>(input, viaRow, 3);
        Assert.That(viaDim, Is.EqualTo(viaRow).Within(1e-6));
    }

    [Test]
    public void SoftmaxDim_InvalidLayout_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.SoftmaxDim<float>(new float[12], new float[12], 2, 3, 4));
    }

    static float GeluGradientExpected(float x)
    {
        const float sqrt2OverPi = 0.7978845608028654f;
        const float coeff = 0.044715f;
        float z = sqrt2OverPi * (x + coeff * x * x * x);
        float tanhZ = MathF.Tanh(z);
        float dTanh = 1f - tanhZ * tanhZ;
        return 0.5f * (1f + tanhZ) + 0.5f * x * dTanh * sqrt2OverPi * (1f + 3f * coeff * x * x);
    }

    [Test]
    public void Gelu_Float_KnownValues()
    {
        AssertKernel(FloatInputs, GeluExpected, GradKernels.Gelu<float>, 1e-4);
    }

    [Test]
    public void Gelu_Double_KnownValues()
    {
        AssertKernel(DoubleInputs, x =>
        {
            const double sqrt2OverPi = 0.7978845608028654;
            const double coeff = 0.044715;
            double z = sqrt2OverPi * (x + coeff * x * x * x);
            return 0.5 * x * (1.0 + Math.Tanh(z));
        }, GradKernels.Gelu<double>, 1e-8);
    }

    [Test]
    public void GeluGradient_Float_KnownValues()
    {
        var grad = Enumerable.Range(1, FloatInputs.Length).Select(i => (float)i * 0.5f).ToArray();
        AssertGradientKernel(FloatInputs, grad, (x, g) => GeluGradientExpected(x) * g, GradKernels.GeluGradient<float>, 1e-4);
    }

    [Test]
    public void GeluExact_Float_KnownValues()
    {
        AssertKernel(FloatInputs, x => 0.5f * x * (1f + (float)Erf(x * 0.7071067811865476f)), GradKernels.GeluExact<float>, 1e-4);
    }

    [Test]
    public void GeluExactGradient_Float_KnownValues()
    {
        var grad = Enumerable.Range(1, FloatInputs.Length).Select(i => (float)i * 0.5f).ToArray();
        AssertGradientKernel(FloatInputs, grad, (x, g) =>
        {
            const float invSqrt2 = 0.7071067811865476f;
            const float invSqrt2Pi = 0.3989422804014327f;
            float cdf = 0.5f * (1f + (float)Erf(x * invSqrt2));
            float pdf = MathF.Exp(-0.5f * x * x) * invSqrt2Pi;
            return (cdf + x * pdf) * g;
        }, GradKernels.GeluExactGradient<float>, 1e-4);
    }

    [Test]
    public void MatMul_2x3By3x2_KnownValues()
    {
        var a = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var b = new float[] { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
        var actual = new float[2 * 2];
        GradKernels.MatMul<float>(a, b, actual, 2, 3, 2);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                float expected = 0f;
                for (int k = 0; k < 3; k++)
                    expected += a[i * 3 + k] * b[k * 2 + j];
                Assert.That(actual[i * 2 + j], Is.EqualTo(expected).Within(1e-5), $"Index {i},{j}");
            }
        }
    }

    [Test]
    public void Transpose_2x3_KnownValues()
    {
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var actual = new float[input.Length];
        GradKernels.Transpose<float>(input, actual, 2, 3);

        Assert.That(actual, Is.EqualTo(new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f }).Within(1e-6));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Half (F16) variants
    //  All FloatInputs values are exactly representable in Half, so the
    //  expected values are computed from the exact inputs in double and compared
    //  with tolerances that absorb the reduced output precision.
    // ══════════════════════════════════════════════════════════════════════

    static readonly Half[] HalfInputs = Array.ConvertAll(FloatInputs, x => (Half)x);

    static readonly float[] LogInputs = { 0.5f, 1.0f, 2.0f, 3.0f, 0.25f, 10.0f, 4.0f, 1.5f };
    static readonly Half[] HalfLogInputs = Array.ConvertAll(LogInputs, x => (Half)x);

    static T D<T>(double v) where T : struct, IFloatingPointIeee754<T> => T.CreateChecked(v);

    static void AssertRowSoftmax<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> actual, int classCount, double tolerance)
        where T : struct, IFloatingPointIeee754<T>
    {
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            double max = double.CreateChecked(input[r * classCount]);
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, double.CreateChecked(input[r * classCount + j]));

            double sum = 0.0;
            for (int j = 0; j < classCount; j++)
                sum += Math.Exp(double.CreateChecked(input[r * classCount + j]) - max);

            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                Assert.That(double.CreateChecked(actual[idx]),
                    Is.EqualTo(Math.Exp(double.CreateChecked(input[idx]) - max) / sum).Within(tolerance),
                    $"Index {idx}");
            }
        }
    }

    [Test]
    public void Sigmoid_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(1.0 / (1.0 + Math.Exp(-double.CreateChecked(x)))), GradKernels.Sigmoid<Half>, 1e-2);
    }

    [Test]
    public void Tanh_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(Math.Tanh(double.CreateChecked(x))), GradKernels.Tanh<Half>, 1e-2);
    }

    [Test]
    public void Relu_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(Math.Max(double.CreateChecked(x), 0.0)), GradKernels.Relu<Half>, 1e-6);
    }

    [Test]
    public void Exp_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(Math.Exp(double.CreateChecked(x))), GradKernels.Exp<Half>, 1e-2);
    }

    [Test]
    public void Log_Half_KnownValues()
    {
        AssertKernel(HalfLogInputs, x => D<Half>(Math.Log(double.CreateChecked(x))), GradKernels.Log<Half>, 1e-2);
    }

    [Test]
    public void Abs_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(Math.Abs(double.CreateChecked(x))), GradKernels.Abs<Half>, 1e-6);
    }

    [Test]
    public void Clamp_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(Math.Clamp(double.CreateChecked(x), -1.0, 2.0)),
            (input, output) => GradKernels.Clamp<Half>(input, (Half)(-1.0), (Half)2.0, output), 1e-6);
    }

    [Test]
    public void Negate_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(-double.CreateChecked(x)), GradKernels.Negate<Half>, 1e-6);
    }

    [Test]
    public void Divide_Half_KnownValues()
    {
        var numerator = Array.ConvertAll(new[] { 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f }, x => (Half)x);
        var denominator = Array.ConvertAll(new[] { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }, x => (Half)x);
        var actual = new Half[numerator.Length];
        GradKernels.Divide<Half>(numerator, denominator, actual);
        for (int i = 0; i < numerator.Length; i++)
            Assert.That(double.CreateChecked(actual[i]),
                Is.EqualTo(double.CreateChecked(numerator[i]) / double.CreateChecked(denominator[i])).Within(1e-2),
                $"Index {i}");
    }

    [Test]
    public void Gelu_Half_KnownValues()
    {
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;
        AssertKernel(HalfInputs, x =>
        {
            double v = double.CreateChecked(x);
            double z = sqrt2OverPi * (v + coeff * v * v * v);
            return D<Half>(0.5 * v * (1.0 + Math.Tanh(z)));
        }, GradKernels.Gelu<Half>, 1e-2);
    }

    [Test]
    public void GeluExact_Half_KnownValues()
    {
        AssertKernel(HalfInputs, x => D<Half>(0.5 * double.CreateChecked(x) * (1.0 + Erf(double.CreateChecked(x) * 0.7071067811865476))), GradKernels.GeluExact<Half>, 1e-2);
    }

    [Test]
    public void Softmax_RowWise_Half_KnownValues()
    {
        var input = Array.ConvertAll(new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f }, x => (Half)x);
        var actual = new Half[input.Length];
        GradKernels.Softmax<Half>(input, actual, 3);
        AssertRowSoftmax(input, actual, 3, 1e-2);
    }

    [Test]
    public void LogSoftmax_RowWise_Half_KnownValues()
    {
        var input = Array.ConvertAll(new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f }, x => (Half)x);
        var actual = new Half[input.Length];
        GradKernels.LogSoftmax<Half>(input, actual, 3);
        AssertRowLogSoftmax(input, actual, 3, 1e-2);
    }

    static void AssertRowLogSoftmax<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> actual, int classCount, double tolerance)
        where T : struct, IFloatingPointIeee754<T>
    {
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            double max = double.CreateChecked(input[r * classCount]);
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, double.CreateChecked(input[r * classCount + j]));

            double sum = 0.0;
            for (int j = 0; j < classCount; j++)
                sum += Math.Exp(double.CreateChecked(input[r * classCount + j]) - max);

            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                Assert.That(double.CreateChecked(actual[idx]),
                    Is.EqualTo(double.CreateChecked(input[idx]) - max - Math.Log(sum)).Within(tolerance),
                    $"Index {idx}");
            }
        }
    }

    [Test]
    public void MatMul_Half_Unsupported_ThrowsNotSupported()
    {
        var a = Array.ConvertAll(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, x => (Half)x);
        var b = Array.ConvertAll(new[] { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }, x => (Half)x);
        var actual = new Half[4];
        Assert.Throws<NotSupportedException>(() => GradKernels.MatMul<Half>(a, b, actual, 2, 3, 2));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Gradient kernels (Half / float)
    // ══════════════════════════════════════════════════════════════════════

    [Test]
    public void SigmoidGradient_Half_KnownValues()
    {
        var input = Array.ConvertAll(new[] { 0.7310585786300049f, 0.2689414213699951f, 0.5f, 0.6224593312018546f, 0.7310585786300049f, 0.9241418199787566f, 0.9525741268224334f, 0.43782349911420193f }, x => (Half)x);
        var grad = Array.ConvertAll(new[] { 1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.25f, 2.0f, -0.5f }, x => (Half)x);
        AssertGradientKernel(input, grad, (s, g) => D<Half>(double.CreateChecked(s) * (1.0 - double.CreateChecked(s)) * double.CreateChecked(g)), GradKernels.SigmoidGradient<Half>, 1e-2);
    }

    [Test]
    public void TanhGradient_Half_KnownValues()
    {
        var input = Array.ConvertAll(new[] { -0.9981778976111987f, -0.7615941559557649f, 0f, 0.46211715726000974f, 0.7615941559557649f, 0.9866142981514303f, 0.9950547536867305f, -0.24491866240370913f }, x => (Half)x);
        var grad = Array.ConvertAll(new[] { 0.5f, 1.0f, -1.0f, 2.0f, -0.5f, 1.5f, -2.0f, 0.25f }, x => (Half)x);
        AssertGradientKernel(input, grad, (t, g) => D<Half>((1.0 - double.CreateChecked(t) * double.CreateChecked(t)) * double.CreateChecked(g)), GradKernels.TanhGradient<Half>, 1e-2);
    }

    [Test]
    public void ReluGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfInputs.Length).Select(i => (Half)i).ToArray();
        AssertGradientKernel(HalfInputs, grad, (x, g) => double.CreateChecked(x) > 0.0 ? g : (Half)0, GradKernels.ReluGradient<Half>, 1e-6);
    }

    [Test]
    public void LeakyReluGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfInputs.Length).Select(i => (Half)(i * 0.5)).ToArray();
        const float slope = 0.01f;
        AssertGradientKernel(HalfInputs, grad, (x, g) => double.CreateChecked(x) > 0.0 ? g : D<Half>(slope * double.CreateChecked(g)),
            (input, g, output) => GradKernels.LeakyReluGradient<Half>(input, g, (Half)slope, output), 1e-3);
    }

    [Test]
    public void GeluGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfInputs.Length).Select(i => (Half)(i * 0.5)).ToArray();
        AssertGradientKernel(HalfInputs, grad, (x, g) => D<Half>(GeluGradientExpected(double.CreateChecked(x)) * double.CreateChecked(g)), GradKernels.GeluGradient<Half>, 1e-2);
    }

    static double GeluGradientExpected(double x)
    {
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;
        double z = sqrt2OverPi * (x + coeff * x * x * x);
        double tanhZ = Math.Tanh(z);
        double dTanh = 1.0 - tanhZ * tanhZ;
        return 0.5 * (1.0 + tanhZ) + 0.5 * x * dTanh * sqrt2OverPi * (1.0 + 3.0 * coeff * x * x);
    }

    [Test]
    public void GeluExactGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfInputs.Length).Select(i => (Half)(i * 0.5)).ToArray();
        AssertGradientKernel(HalfInputs, grad, (x, g) => D<Half>(GeluExactGradientExpected(double.CreateChecked(x)) * double.CreateChecked(g)), GradKernels.GeluExactGradient<Half>, 1e-2);
    }

    static double GeluExactGradientExpected(double x)
    {
        const double invSqrt2 = 0.7071067811865476;
        const double invSqrt2Pi = 0.3989422804014327;
        double cdf = 0.5 * (1.0 + Erf(x * invSqrt2));
        double pdf = Math.Exp(-0.5 * x * x) * invSqrt2Pi;
        return cdf + x * pdf;
    }

    [Test]
    public void LogGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfLogInputs.Length).Select(i => (Half)i).ToArray();
        AssertGradientKernel(HalfLogInputs, grad, (x, g) => D<Half>(double.CreateChecked(g) / double.CreateChecked(x)), GradKernels.LogGradient<Half>, 1e-2);
    }

    [Test]
    public void AbsGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfInputs.Length).Select(i => (Half)(i * 0.25)).ToArray();
        AssertGradientKernel(HalfInputs, grad, (x, g) => D<Half>(Math.Sign(double.CreateChecked(x))) * g, GradKernels.AbsGradient<Half>, 1e-6);
    }

    [Test]
    public void ClipGradient_Half_KnownValues()
    {
        var grad = Enumerable.Range(1, HalfInputs.Length).Select(i => (Half)i).ToArray();
        AssertGradientKernel(HalfInputs, grad, (x, g) => double.CreateChecked(x) >= -1.0 && double.CreateChecked(x) <= 2.0 ? g : (Half)0,
            (input, g, output) => GradKernels.ClipGradient<Half>(input, g, (Half)(-1.0), (Half)2.0, output), 1e-6);
    }

    [Test]
    public void SoftmaxGradient_RowWise_Half_KnownValues()
    {
        var input = Array.ConvertAll(new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f }, x => (Half)x);
        var grad = Array.ConvertAll(new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f }, x => (Half)x);
        var softmax = new Half[input.Length];
        GradKernels.Softmax<Half>(input, softmax, 3);
        var actual = new Half[input.Length];
        GradKernels.SoftmaxGradient<Half>(softmax, grad, actual, 3);
        AssertSoftmaxGradient(softmax, grad, actual, 3, 1e-2);
    }

    static void AssertSoftmaxGradient<T>(ReadOnlySpan<T> softmax, ReadOnlySpan<T> grad, ReadOnlySpan<T> actual, int classCount, double tolerance)
        where T : struct, IFloatingPointIeee754<T>
    {
        int rows = softmax.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            double dot = 0.0;
            for (int j = 0; j < classCount; j++)
                dot += double.CreateChecked(softmax[r * classCount + j]) * double.CreateChecked(grad[r * classCount + j]);
            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                double expected = double.CreateChecked(softmax[idx]) * (double.CreateChecked(grad[idx]) - dot);
                Assert.That(double.CreateChecked(actual[idx]), Is.EqualTo(expected).Within(tolerance), $"Index {idx}");
            }
        }
    }

    [Test]
    public void LogSoftmaxGradient_RowWise_Half_KnownValues()
    {
        var input = Array.ConvertAll(new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f }, x => (Half)x);
        var grad = Array.ConvertAll(new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f }, x => (Half)x);
        var actual = new Half[input.Length];
        GradKernels.LogSoftmaxGradient<Half>(input, grad, actual, 3);
        AssertLogSoftmaxGradient(input, grad, actual, 3, 1e-2);
    }

    static void AssertLogSoftmaxGradient<T>(ReadOnlySpan<T> input, ReadOnlySpan<T> grad, ReadOnlySpan<T> actual, int classCount, double tolerance)
        where T : struct, IFloatingPointIeee754<T>
    {
        int rows = input.Length / classCount;
        for (int r = 0; r < rows; r++)
        {
            double max = double.CreateChecked(input[r * classCount]);
            for (int j = 1; j < classCount; j++)
                max = Math.Max(max, double.CreateChecked(input[r * classCount + j]));
            double sumExp = 0.0;
            double sumGrad = 0.0;
            for (int j = 0; j < classCount; j++)
            {
                sumExp += Math.Exp(double.CreateChecked(input[r * classCount + j]) - max);
                sumGrad += double.CreateChecked(grad[r * classCount + j]);
            }
            for (int j = 0; j < classCount; j++)
            {
                int idx = r * classCount + j;
                double soft = Math.Exp(double.CreateChecked(input[idx]) - max) / sumExp;
                double expected = double.CreateChecked(grad[idx]) - soft * sumGrad;
                Assert.That(double.CreateChecked(actual[idx]), Is.EqualTo(expected).Within(tolerance), $"Index {idx}");
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Additional float coverage: single-row softmax grads, MatMul variants
    // ══════════════════════════════════════════════════════════════════════

    [Test]
    public void SoftmaxGradient_SingleRow_KnownValues()
    {
        var input = new[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var grad = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f };
        var softmax = new float[input.Length];
        GradKernels.Softmax<float>(input, softmax, input.Length);
        var actual = new float[input.Length];
        GradKernels.SoftmaxGradient<float>(softmax, grad, actual, input.Length);

        double dot = 0.0;
        for (int j = 0; j < input.Length; j++)
            dot += softmax[j] * grad[j];
        for (int j = 0; j < input.Length; j++)
        {
            double expected = softmax[j] * (grad[j] - dot);
            Assert.That(actual[j], Is.EqualTo((float)expected).Within(1e-5), $"Index {j}");
        }
    }

    [Test]
    public void LogSoftmaxGradient_SingleRow_KnownValues()
    {
        var input = new[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var grad = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f };
        var actual = new float[input.Length];
        GradKernels.LogSoftmaxGradient<float>(input, grad, actual, input.Length);

        double max = input.Max();
        double sumExp = 0.0;
        double sumGrad = 0.0;
        for (int j = 0; j < input.Length; j++)
        {
            sumExp += Math.Exp(input[j] - max);
            sumGrad += grad[j];
        }
        for (int j = 0; j < input.Length; j++)
        {
            double soft = Math.Exp(input[j] - max) / sumExp;
            double expected = grad[j] - soft * sumGrad;
            Assert.That(actual[j], Is.EqualTo((float)expected).Within(1e-5), $"Index {j}");
        }
    }

    [Test]
    public void MatMul_2x3By3x4_KnownValues()
    {
        var a = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var b = new[] { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f };
        var actual = new float[2 * 4];
        GradKernels.MatMul<float>(a, b, actual, 2, 3, 4);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float expected = 0f;
                for (int k = 0; k < 3; k++)
                    expected += a[i * 3 + k] * b[k * 4 + j];
                Assert.That(actual[i * 4 + j], Is.EqualTo(expected).Within(1e-5), $"Index {i},{j}");
            }
        }
    }

    [Test]
    public void MatMulTransposedB_2x3Transposed_KnownValues()
    {
        var a = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var b = new[] { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
        var actual = new float[2 * 2];
        GradKernels.MatMulTransposedB<float>(a, b, actual, 2, 3, 2);

        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                float expected = 0f;
                for (int k = 0; k < 3; k++)
                    expected += a[i * 3 + k] * b[j * 3 + k];
                Assert.That(actual[i * 2 + j], Is.EqualTo(expected).Within(1e-5), $"Index {i},{j}");
            }
        }
    }

    [Test]
    public void MatMulTransposedB_Half_Unsupported_ThrowsNotSupported()
    {
        var a = Array.ConvertAll(new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }, x => (Half)x);
        var b = Array.ConvertAll(new[] { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }, x => (Half)x);
        var actual = new Half[4];
        Assert.Throws<NotSupportedException>(() => GradKernels.MatMulTransposedB<Half>(a, b, actual, 2, 3, 2));
    }

    // ══════════════════════════════════════════════════════════════════════
    //  Error paths
    // ══════════════════════════════════════════════════════════════════════

    [Test]
    public void ElementwiseKernels_ShortOutputSpan_Throw()
    {
        var input = new float[4];
        var shortOutput = new float[3];
        Assert.Throws<ArgumentException>(() => GradKernels.Sigmoid<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Tanh<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Relu<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Exp<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Log<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Abs<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Negate<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Clamp<float>(input, -1f, 1f, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.LeakyRelu<float>(input, 0.01f, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Gelu<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.GeluExact<float>(input, shortOutput));
        Assert.Throws<ArgumentException>(() => GradKernels.Divide<float>(input, input, shortOutput));
    }

    [Test]
    public void Softmax_ShortOutputSpan_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.Softmax<float>(new float[6], new float[3], 3));
    }

    [Test]
    public void LogSoftmax_ShortOutputSpan_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.LogSoftmax<float>(new float[6], new float[3], 3));
    }

    [Test]
    public void SoftmaxGradient_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.SoftmaxGradient<float>(new float[6], new float[4], new float[6], 3));
        Assert.Throws<ArgumentException>(() => GradKernels.SoftmaxGradient<float>(new float[6], new float[6], new float[3], 3));
    }

    [Test]
    public void LogSoftmaxGradient_LengthMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.LogSoftmaxGradient<float>(new float[6], new float[4], new float[6], 3));
        Assert.Throws<ArgumentException>(() => GradKernels.LogSoftmaxGradient<float>(new float[6], new float[6], new float[3], 3));
    }

    [Test]
    public void MatMul_ShortResult_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.MatMul<float>(new float[6], new float[6], new float[3], 2, 3, 2));
    }

    [Test]
    public void MatMulTransposedB_ShortResult_Throws()
    {
        Assert.Throws<ArgumentException>(() => GradKernels.MatMulTransposedB<float>(new float[6], new float[6], new float[3], 2, 3, 2));
    }
}
