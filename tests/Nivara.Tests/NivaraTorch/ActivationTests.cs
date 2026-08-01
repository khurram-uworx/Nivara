using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class ActivationTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void LeakyRelu_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("leaky_relu_1d_input.bin");
        var expected = TestHelpers.LoadBin("leaky_relu_1d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.LeakyRelu(inputTensor, 0.01f);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "LeakyRelu_1D");
    }

    [Test]
    public void LeakyRelu_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("leaky_relu_4d_input.bin");
        var expected = TestHelpers.LoadBin("leaky_relu_4d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 4, 4);
        var output = Activation.LeakyRelu(inputTensor, 0.01f);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "LeakyRelu_4D");
    }

    [Test]
    public void Sigmoid_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("sigmoid_1d_input.bin");
        var expected = TestHelpers.LoadBin("sigmoid_1d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.Sigmoid(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Sigmoid_1D");
    }

    [Test]
    public void Sigmoid_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("sigmoid_4d_input.bin");
        var expected = TestHelpers.LoadBin("sigmoid_4d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 4, 4);
        var output = Activation.Sigmoid(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Sigmoid_4D");
    }

    [Test]
    public void Tanh_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("tanh_1d_input.bin");
        var expected = TestHelpers.LoadBin("tanh_1d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.Tanh(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Tanh_1D");
    }

    [Test]
    public void Tanh_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("tanh_4d_input.bin");
        var expected = TestHelpers.LoadBin("tanh_4d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 4, 4);
        var output = Activation.Tanh(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Tanh_4D");
    }

    static float GeluTanhApprox(float x)
    {
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;
        double inner = sqrt2OverPi * (x + coeff * x * x * x);
        return (float)(0.5 * x * (1.0 + Math.Tanh(inner)));
    }

    static double Erf(double x)
    {
        if (x < 0) return -Erf(-x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        return 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
    }

    static float GeluExact(float x)
        => (float)(0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0))));

    static float GeluGradientTanhApprox(float x)
    {
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;
        double val = x;
        double inner = sqrt2OverPi * (val + coeff * val * val * val);
        double tanhVal = Math.Tanh(inner);
        double sech2 = 1.0 - tanhVal * tanhVal;
        double dInnerDx = sqrt2OverPi * (1.0 + 3.0 * coeff * val * val);
        return (float)(0.5 * (1.0 + tanhVal) + 0.5 * val * sech2 * dInnerDx);
    }

    [Test]
    public void Gelu_KnownValues_Correct()
    {
        var testValues = new float[] { 0f, 1f, -1f, 2f, -2f, 0.5f, -0.5f };
        var input = ReverseGradTensor<float>.FromArray(testValues, requiresGrad: false);

        var output = Activation.Gelu(input);

        Assert.That(output.Length, Is.EqualTo(7));
        for (int i = 0; i < testValues.Length; i++)
        {
            float expected = GeluTanhApprox(testValues[i]);
            TestHelpers.AssertScalarEqual(expected, output[i], label: $"Gelu({testValues[i]})");
        }
    }

    [Test]
    public void Gelu_Zero_IsZero()
    {
        var input = ReverseGradTensor<float>.FromArray(new float[] { 0f }, requiresGrad: false);
        var output = Activation.Gelu(input);

        TestHelpers.AssertScalarEqual(0f, output[0], absTol: 1e-6f, label: "Gelu(0)");
    }

    [Test]
    public void Gelu_LargePositive_ApproachesIdentity()
    {
        var input = ReverseGradTensor<float>.FromArray(new float[] { 10f }, requiresGrad: false);
        var output = Activation.Gelu(input);

        Assert.That(output[0], Is.GreaterThan(9.9f),
            "GELU(10) should be close to 10 (approaches identity for large x)");
    }

    [Test]
    public void Gelu_LargeNegative_ApproachesZero()
    {
        var input = ReverseGradTensor<float>.FromArray(new float[] { -10f }, requiresGrad: false);
        var output = Activation.Gelu(input);

        Assert.That(output[0], Is.InRange(-0.1f, 0f),
            "GELU(-10) should be close to 0 (approaches 0 for large negative x)");
    }

    [Test]
    public void Gelu_GradientFlows()
    {
        var inputData = new float[] { -1f, 0f, 1f, 2f };
        var input = ReverseGradTensor<float>.FromArray(inputData, requiresGrad: true);

        var output = ReverseGradOperations.Gelu(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < inputData.Length; i++)
        {
            float expectedGrad = GeluGradientTanhApprox(inputData[i]);
            TestHelpers.AssertScalarEqual(expectedGrad, input.Grad![i], absTol: 3e-4f,
                label: $"GeluGrad({inputData[i]})");
        }
    }

    [Test]
    public void GeluExact_KnownValues_Correct()
    {
        var testValues = new float[] { 0f, 1f, -1f, 2f, -2f, 0.5f, -0.5f };
        var input = ReverseGradTensor<float>.FromArray(testValues, requiresGrad: false);

        var output = Activation.GeluExact(input);

        Assert.That(output.Length, Is.EqualTo(7));
        for (int i = 0; i < testValues.Length; i++)
        {
            float expected = GeluExact(testValues[i]);
            TestHelpers.AssertScalarEqual(expected, output[i], label: $"GeluExact({testValues[i]})");
        }
    }

    [Test]
    public void GeluExact_GradientFlows()
    {
        var inputData = new float[] { -1f, 0f, 1f, 2f };
        var input = ReverseGradTensor<float>.FromArray(inputData, requiresGrad: true);

        var output = ReverseGradOperations.GeluExact(input);
        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(input.Grad, Is.Not.Null);
        for (int i = 0; i < inputData.Length; i++)
        {
            double x = inputData[i];
            double cdf = 0.5 * (1.0 + Erf(x / Math.Sqrt(2.0)));
            double pdf = Math.Exp(-0.5 * x * x) / Math.Sqrt(2.0 * Math.PI);
            float expectedGrad = (float)(cdf + x * pdf);
            TestHelpers.AssertScalarEqual(expectedGrad, input.Grad![i], absTol: 1e-5f,
                label: $"GeluExactGrad({inputData[i]})");
        }
    }

    [Test]
    public void GeluExact_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("gelu_exact_1d_input.bin");
        var expected = TestHelpers.LoadBin("gelu_exact_1d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.GeluExact(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "GeluExact_1D");
    }

    [Test]
    public void GeluExact_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("gelu_exact_4d_input.bin");
        var expected = TestHelpers.LoadBin("gelu_exact_4d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 4, 4);
        var output = Activation.GeluExact(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "GeluExact_4D");
    }

    [Test]
    public void Gelu_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("gelu_1d_input.bin");
        var expected = TestHelpers.LoadBin("gelu_1d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.Gelu(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Gelu_1D");
    }

    [Test]
    public void Gelu_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("gelu_4d_input.bin");
        var expected = TestHelpers.LoadBin("gelu_4d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 4, 4);
        var output = Activation.Gelu(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Gelu_4D");
    }
}
