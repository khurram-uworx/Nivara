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
}
