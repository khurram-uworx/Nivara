using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class DepthwiseSeparableConv2dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void DepthwiseSeparableConv2d_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("dsc_input.bin");
        var dwWeight = TestHelpers.LoadBin("dsc_dw_weight.bin");
        var pwWeight = TestHelpers.LoadBin("dsc_pw_weight.bin");
        var pwBias = TestHelpers.LoadBin("dsc_pw_bias.bin");
        var expectedOutput = TestHelpers.LoadBin("dsc_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("dsc_input_grad.bin");

        using var dsc = new DepthwiseSeparableConv2d<float>(
            inChannels: 4, outChannels: 8, kernelSize: 3, stride: 1, padding: 1, useBias: true);
        dsc.DepthwiseConv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(dwWeight, 4, 9, requiresGrad: false);
        dsc.PointwiseConv.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(pwWeight, 8, 4, requiresGrad: false);
        dsc.PointwiseConv.Bias!.Tensor = ReverseGradTensor<float>.FromArray(pwBias, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(1, 4, 8, 8);
        var output = dsc.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 8, 8 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "DepthwiseSeparableConv2d_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), label: "DepthwiseSeparableConv2d_input_grad");
    }
}