using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class Conv1dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Conv1d_Kernel3_Stride1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv1d_k3_input.bin");
        var weight = TestHelpers.LoadBin("conv1d_k3_weight.bin");
        var bias = TestHelpers.LoadBin("conv1d_k3_bias.bin");
        var expected = TestHelpers.LoadBin("conv1d_k3_output.bin");

        using var conv = new Conv1d<float>(8, 8, kernelSize: 3, stride: 1, padding: 1, bias: true);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 8, 24, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 16);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv1d_k3");
    }

    [Test]
    public void Conv1d_Kernel5_Stride1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv1d_k5_input.bin");
        var weight = TestHelpers.LoadBin("conv1d_k5_weight.bin");
        var bias = TestHelpers.LoadBin("conv1d_k5_bias.bin");
        var expected = TestHelpers.LoadBin("conv1d_k5_output.bin");

        using var conv = new Conv1d<float>(8, 16, kernelSize: 5, stride: 1, padding: 2, bias: true);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 16, 40, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 16);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv1d_k5");
    }

    [Test]
    public void Conv1d_Kernel7_Stride1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv1d_k7_input.bin");
        var weight = TestHelpers.LoadBin("conv1d_k7_weight.bin");
        var bias = TestHelpers.LoadBin("conv1d_k7_bias.bin");
        var expected = TestHelpers.LoadBin("conv1d_k7_output.bin");

        using var conv = new Conv1d<float>(4, 8, kernelSize: 7, stride: 1, padding: 3, bias: true);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 8, 28, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 4, 32);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 32 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv1d_k7");
    }

    [Test]
    public void Conv1d_Kernel3_Stride2_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv1d_s2_input.bin");
        var weight = TestHelpers.LoadBin("conv1d_s2_weight.bin");
        var bias = TestHelpers.LoadBin("conv1d_s2_bias.bin");
        var expected = TestHelpers.LoadBin("conv1d_s2_output.bin");

        using var conv = new Conv1d<float>(8, 16, kernelSize: 3, stride: 2, padding: 1, bias: true);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 16, 24, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 8, 16);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 8 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv1d_s2");
    }
}
