using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class LinearTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Linear_128_64_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("linear_128_64_input.bin");
        var weight = TestHelpers.LoadBin("linear_128_64_weight.bin");
        var biasData = TestHelpers.LoadBin("linear_128_64_bias.bin");
        var expected = TestHelpers.LoadBin("linear_128_64_output.bin");

        using var linear = new Linear<float>(128, 64, bias: true);
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromMatrix(weight, 64, 128, requiresGrad: false),
            ["Bias"] = ReverseGradTensor<float>.FromMatrix(biasData, 1, 64, requiresGrad: false),
        });

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 128);

        var output = linear.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 64 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Linear_128_64");
    }

    [Test]
    public void Linear_512_1000_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("linear_512_1000_input.bin");
        var weight = TestHelpers.LoadBin("linear_512_1000_weight.bin");
        var biasData = TestHelpers.LoadBin("linear_512_1000_bias.bin");
        var expected = TestHelpers.LoadBin("linear_512_1000_output.bin");

        using var linear = new Linear<float>(512, 1000, bias: true);
        linear.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromMatrix(weight, 1000, 512, requiresGrad: false),
            ["Bias"] = ReverseGradTensor<float>.FromMatrix(biasData, 1, 1000, requiresGrad: false),
        });

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 512);

        var output = linear.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 1000 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Linear_512_1000");
    }
}
