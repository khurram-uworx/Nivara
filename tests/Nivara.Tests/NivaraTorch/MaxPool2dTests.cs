using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class MaxPool2dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void MaxPool2d_3x3Stride2Pad1_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("maxpool_3x3_s2_p1_input.bin");
        var expected = TestHelpers.LoadBin("maxpool_3x3_s2_p1_output.bin");

        using var pool = new MaxPool2d<float>(kernelSize: 3, stride: 2, padding: 1);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 14, 14);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 7, 7 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "MaxPool2d_3x3_s2_p1");
    }

    [Test]
    public void MaxPool2d_2x2Stride2Pad0_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("maxpool_2x2_s2_p0_input.bin");
        var expected = TestHelpers.LoadBin("maxpool_2x2_s2_p0_output.bin");

        using var pool = new MaxPool2d<float>(kernelSize: 2, stride: 2, padding: 0);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 32, 28, 28);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 14, 14 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "MaxPool2d_2x2_s2_p0");
    }
}
