using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
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
}
