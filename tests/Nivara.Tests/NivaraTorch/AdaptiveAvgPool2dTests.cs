using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class AdaptiveAvgPool2dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void AdaptiveAvgPool2d_1x1_Large_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("adaptiveavgpool_1x1_input.bin");
        var expected = TestHelpers.LoadBin("adaptiveavgpool_1x1_output.bin");

        using var pool = new AdaptiveAvgPool2d<float>(outputSize: 1);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 512, 7, 7);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 512, 1, 1 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "AdaptiveAvgPool2d_1x1_large");
    }

    [Test]
    public void AdaptiveAvgPool2d_1x1_Small_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("adaptiveavgpool_1x1_sm_input.bin");
        var expected = TestHelpers.LoadBin("adaptiveavgpool_1x1_sm_output.bin");

        using var pool = new AdaptiveAvgPool2d<float>(outputSize: 1);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 32, 14, 14);

        var output = pool.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 32, 1, 1 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "AdaptiveAvgPool2d_1x1_small");
    }
}
