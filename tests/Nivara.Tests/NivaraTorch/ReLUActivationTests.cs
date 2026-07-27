using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class ReLUActivationTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void ReLU_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_1d_input.bin");
        var expected = TestHelpers.LoadBin("relu_1d_relu_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = Activation.Relu(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "ReLU_1D");
    }

    [Test]
    public void ReLU_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_4d_input.bin");
        var expected = TestHelpers.LoadBin("relu_4d_relu_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 8, 8);
        var output = Activation.Relu(inputTensor);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "ReLU_4D");
    }

    [Test]
    public void ReLU6_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_1d_input.bin");
        var expected = TestHelpers.LoadBin("relu_1d_relu6_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = ReverseGradOperations.Clip(Activation.Relu(inputTensor), 0f, 6f);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "ReLU6_1D");
    }

    [Test]
    public void ReLU6_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("relu_4d_input.bin");
        var expected = TestHelpers.LoadBin("relu_4d_relu6_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 8, 8);
        var output = ReverseGradOperations.Clip(Activation.Relu(inputTensor), 0f, 6f);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "ReLU6_4D");
    }
}
