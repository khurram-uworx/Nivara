using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class Conv2dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Conv2d_3x3Stride1Pad1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_3x3_s1_p1_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_3x3_s1_p1_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_3x3_s1_p1_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_3x3_s1_p1_output.bin");

        using var conv = new Conv2d<float>(3, 16, kernelSize: 3, stride: 1, padding: 1, bias: true, groups: 1);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 16, 27, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 7, 7);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 7, 7 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv2d_3x3_s1_p1");
    }

    [Test]
    public void Conv2d_1x1Stride1Pad0_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_1x1_s1_p0_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_1x1_s1_p0_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_1x1_s1_p0_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_1x1_s1_p0_output.bin");

        using var conv = new Conv2d<float>(3, 32, kernelSize: 1, stride: 1, padding: 0, bias: true, groups: 1);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 32, 3, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 7, 7);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 7, 7 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv2d_1x1_s1_p0");
    }

    [Test]
    public void Conv2d_Depthwise_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_depthwise_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_depthwise_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_depthwise_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_depthwise_output.bin");

        using var conv = new Conv2d<float>(16, 16, kernelSize: 3, stride: 1, padding: 1, bias: true, groups: 16);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 16, 9, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 5, 5);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 5, 5 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv2d_depthwise");
    }

    [Test]
    public void Conv2d_Stride2_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_stride2_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_stride2_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_stride2_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_stride2_output.bin");

        using var conv = new Conv2d<float>(3, 32, kernelSize: 3, stride: 2, padding: 1, bias: true, groups: 1);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 32, 27, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 14, 14);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 7, 7 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv2d_stride2");
    }

    [Test]
    public void Conv2d_WithBias_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("conv2d_with_bias_input.bin");
        var weight = TestHelpers.LoadBin("conv2d_with_bias_weight.bin");
        var bias = TestHelpers.LoadBin("conv2d_with_bias_bias.bin");
        var expected = TestHelpers.LoadBin("conv2d_with_bias_output.bin");

        using var conv = new Conv2d<float>(3, 8, kernelSize: 3, stride: 1, padding: 1, bias: true, groups: 1);
        conv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 8, 27, requiresGrad: false);
        conv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(bias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 4, 4);

        var output = conv.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 4, 4 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Conv2d_with_bias");
    }
}
