using Nivara.AutoDiff.Operations;
using Nivara.Tensors;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Known-value and parity tests for the span-based <see cref="GradKernels"/>.
/// Parity is established against the column-extension kernels they replace
/// (<see cref="NivaraTensorExtensions"/>), verifying identical numerics while
/// running on plain spans with no null machinery.
/// </summary>
[TestFixture]
public class GradKernelsTests
{
    static float[] FloatInputs() => new[] { -3.5f, -1.0f, 0.0f, 0.5f, 1.0f, 2.5f, 3.0f, -0.25f };
    static double[] DoubleInputs() => new[] { -3.5, -1.0, 0.0, 0.5, 1.0, 2.5, 3.0, -0.25 };

    static void AssertFloatParity(float[] input, Func<NivaraColumn<float>, NivaraColumn<float>> columnOp, Action<ReadOnlySpan<float>, Span<float>> kernelOp, double tolerance = 1e-5)
    {
        var column = NivaraColumn<float>.Create(input);
        var expected = columnOp(column).ToArray();
        var actual = new float[input.Length];
        kernelOp(input, actual);
        Assert.That(actual, Is.EqualTo(expected).Within(tolerance));
    }

    static void AssertFloatGradientParity(
        float[] input, float[] gradOutput,
        Func<NivaraColumn<float>, NivaraColumn<float>, NivaraColumn<float>> columnOp,
        Action<ReadOnlySpan<float>, ReadOnlySpan<float>, Span<float>> kernelOp,
        double tolerance = 1e-5)
    {
        var inCol = NivaraColumn<float>.Create(input);
        var gradCol = NivaraColumn<float>.Create(gradOutput);
        var expected = columnOp(inCol, gradCol).ToArray();
        var actual = new float[input.Length];
        kernelOp(input, gradOutput, actual);
        Assert.That(actual, Is.EqualTo(expected).Within(tolerance));
    }

    static void AssertDoubleParity(double[] input, Func<NivaraColumn<double>, NivaraColumn<double>> columnOp, Action<ReadOnlySpan<double>, Span<double>> kernelOp, double tolerance = 1e-9)
    {
        var column = NivaraColumn<double>.Create(input);
        var expected = columnOp(column).ToArray();
        var actual = new double[input.Length];
        kernelOp(input, actual);
        Assert.That(actual, Is.EqualTo(expected).Within(tolerance));
    }

    [Test]
    public void Sigmoid_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Sigmoid(), GradKernels.Sigmoid<float>);
    }

    [Test]
    public void Sigmoid_Double_MatchesColumnExtension()
    {
        AssertDoubleParity(DoubleInputs(), c => c.Sigmoid(), GradKernels.Sigmoid<double>);
    }

    [Test]
    public void SigmoidGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = new[] { 1.0f, -2.0f, 0.5f, 3.0f, -1.0f, 0.25f, 2.0f, -0.5f };
        var sigmoid = new float[input.Length];
        GradKernels.Sigmoid<float>(input, sigmoid);
        AssertFloatGradientParity(sigmoid, grad, (s, g) => s.SigmoidGradient(g), GradKernels.SigmoidGradient<float>);
    }

    [Test]
    public void Tanh_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Tanh(), GradKernels.Tanh<float>);
    }

    [Test]
    public void Tanh_Double_MatchesColumnExtension()
    {
        AssertDoubleParity(DoubleInputs(), c => c.Tanh(), GradKernels.Tanh<double>);
    }

    [Test]
    public void TanhGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = new[] { 0.5f, 1.0f, -1.0f, 2.0f, -0.5f, 1.5f, -2.0f, 0.25f };
        var tanh = new float[input.Length];
        GradKernels.Tanh<float>(input, tanh);
        AssertFloatGradientParity(tanh, grad, (t, g) => t.TanhGradient(g), GradKernels.TanhGradient<float>);
    }

    [Test]
    public void Relu_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Relu(), GradKernels.Relu<float>);
    }

    [Test]
    public void ReluGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i).ToArray();
        AssertFloatGradientParity(input, grad, (i, g) => i.ReluGradient(g), GradKernels.ReluGradient<float>);
    }

    [Test]
    public void LeakyRelu_Float_MatchesColumnExtension()
    {
        var slope = 0.01f;
        AssertFloatParity(FloatInputs(), c => c.LeakyRelu(slope), (input, output) => GradKernels.LeakyRelu<float>(input, slope, output));
    }

    [Test]
    public void LeakyReluGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i * 0.5f).ToArray();
        const float slope = 0.01f;
        AssertFloatGradientParity(input, grad, (i, g) => i.LeakyReluGradient(g, slope), (i, g, o) => GradKernels.LeakyReluGradient<float>(i, g, slope, o));
    }

    [Test]
    public void Exp_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Exp(), GradKernels.Exp<float>);
    }

    [Test]
    public void Log_Float_MatchesColumnExtension()
    {
        var input = new[] { 0.5f, 1.0f, 2.0f, 3.0f, 0.25f, 10.0f, 4.0f, 1.5f };
        AssertFloatParity(input, c => c.Log(), GradKernels.Log<float>);
    }

    [Test]
    public void LogGradient_Float_MatchesColumnExtension()
    {
        var input = new[] { 0.5f, 1.0f, 2.0f, 3.0f, 0.25f, 10.0f, 4.0f, 1.5f };
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i).ToArray();
        AssertFloatGradientParity(input, grad, (i, g) => i.LogGradient(g), GradKernels.LogGradient<float>);
    }

    [Test]
    public void Abs_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Abs(), GradKernels.Abs<float>);
    }

    [Test]
    public void AbsGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i * 0.25f).ToArray();
        AssertFloatGradientParity(input, grad, (i, g) => i.AbsGradient(g), GradKernels.AbsGradient<float>);
    }

    [Test]
    public void Clamp_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Clamp(-1.0f, 2.0f), (input, output) => GradKernels.Clamp<float>(input, -1.0f, 2.0f, output));
    }

    [Test]
    public void ClipGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i).ToArray();
        AssertFloatGradientParity(input, grad, (i, g) => i.ClipGradient(g, -1.0f, 2.0f), (i, g, o) => GradKernels.ClipGradient<float>(i, g, -1.0f, 2.0f, o));
    }

    [Test]
    public void Negate_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Negate(), GradKernels.Negate<float>);
    }

    [Test]
    public void Divide_Float_MatchesColumnExtension()
    {
        var numerator = new[] { 1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f };
        var denominator = new[] { 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f };
        var left = NivaraColumn<float>.Create(numerator);
        var right = NivaraColumn<float>.Create(denominator);
        var expected = left.Divide(right).ToArray();
        var actual = new float[numerator.Length];
        GradKernels.Divide<float>(numerator, denominator, actual);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }

    [Test]
    public void Softmax_RowWise_MatchesColumnExtension()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var column = NivaraColumn<float>.Create(input);
        var expected = column.Softmax(3).ToArray();
        var actual = new float[input.Length];
        GradKernels.Softmax<float>(input, actual, 3);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void Softmax_SingleVector_MatchesColumnExtension()
    {
        var input = new[] { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f };
        var column = NivaraColumn<float>.Create(input);
        var expected = column.Softmax(input.Length).ToArray();
        var actual = new float[input.Length];
        GradKernels.Softmax<float>(input, actual, input.Length);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void SoftmaxGradient_RowWise_MatchesColumnExtension()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var grad = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f };
        var softmax = new float[input.Length];
        GradKernels.Softmax<float>(input, softmax, 3);
        var softCol = NivaraColumn<float>.Create(softmax);
        var gradCol = NivaraColumn<float>.Create(grad);
        var expected = softCol.SoftmaxGradient(gradCol, 3).ToArray();
        var actual = new float[input.Length];
        GradKernels.SoftmaxGradient<float>(softmax, grad, actual, 3);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void LogSoftmax_RowWise_MatchesColumnExtension()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var column = NivaraColumn<float>.Create(input);
        var expected = column.LogSoftmax(3).ToArray();
        var actual = new float[input.Length];
        GradKernels.LogSoftmax<float>(input, actual, 3);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void LogSoftmaxGradient_RowWise_MatchesColumnExtension()
    {
        var input = new[] { 1.0f, 2.0f, 3.0f, 1.5f, 0.5f, 2.5f, -1.0f, 0.0f, 1.0f };
        var grad = new[] { 0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f, 0.7f, -0.8f, 0.9f };
        var inCol = NivaraColumn<float>.Create(input);
        var gradCol = NivaraColumn<float>.Create(grad);
        var expected = inCol.LogSoftmaxGradient(gradCol, 3).ToArray();
        var actual = new float[input.Length];
        GradKernels.LogSoftmaxGradient<float>(input, grad, actual, 3);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void Gelu_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.Gelu(), GradKernels.Gelu<float>, 1e-4);
    }

    [Test]
    public void Gelu_Double_MatchesColumnExtension()
    {
        AssertDoubleParity(DoubleInputs(), c => c.Gelu(), GradKernels.Gelu<double>, 1e-8);
    }

    [Test]
    public void GeluGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i * 0.5f).ToArray();
        AssertFloatGradientParity(input, grad, (i, g) => i.GeluGradient(g), GradKernels.GeluGradient<float>, 1e-4);
    }

    [Test]
    public void GeluExact_Float_MatchesColumnExtension()
    {
        AssertFloatParity(FloatInputs(), c => c.GeluExact(), GradKernels.GeluExact<float>, 1e-4);
    }

    [Test]
    public void GeluExactGradient_Float_MatchesColumnExtension()
    {
        var input = FloatInputs();
        var grad = Enumerable.Range(1, input.Length).Select(i => (float)i * 0.5f).ToArray();
        AssertFloatGradientParity(input, grad, (i, g) => i.GeluExactGradient(g), GradKernels.GeluExactGradient<float>, 1e-4);
    }

    [Test]
    public void MatMul_2x3By3x2_MatchesColumnExtension()
    {
        var a = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var b = new float[] { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
        var aCol = NivaraColumn<float>.Create(a);
        var bCol = NivaraColumn<float>.Create(b);
        var expected = aCol.MatMul(bCol, 2, 3, 2).ToArray();
        var actual = new float[2 * 2];
        GradKernels.MatMul<float>(a, b, actual, 2, 3, 2);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-5));
    }

    [Test]
    public void Transpose_2x3_MatchesColumnExtension()
    {
        var input = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
        var column = NivaraColumn<float>.Create(input);
        var expected = column.Transpose(2, 3).ToArray();
        var actual = new float[input.Length];
        GradKernels.Transpose<float>(input, actual, 2, 3);
        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
        Assert.That(actual, Is.EqualTo(new[] { 1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f }));
    }
}
