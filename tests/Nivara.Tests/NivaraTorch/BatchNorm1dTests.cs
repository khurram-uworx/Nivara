using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class BatchNorm1dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void BatchNorm1d_Eval_2D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn1d_2d_input.bin");
        var gamma = TestHelpers.LoadBin("bn1d_2d_gamma.bin");
        var beta = TestHelpers.LoadBin("bn1d_2d_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn1d_2d_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn1d_2d_running_var.bin");
        var expected = TestHelpers.LoadBin("bn1d_2d_output.bin");

        using var bn = new BatchNorm1d<float>(16);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 16);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "BatchNorm1d_2d");
    }

    [Test]
    public void BatchNorm1d_Eval_3D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn1d_3d_input.bin");
        var gamma = TestHelpers.LoadBin("bn1d_3d_gamma.bin");
        var beta = TestHelpers.LoadBin("bn1d_3d_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn1d_3d_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn1d_3d_running_var.bin");
        var expected = TestHelpers.LoadBin("bn1d_3d_output.bin");

        using var bn = new BatchNorm1d<float>(8);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(2, 8, 20);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 8, 20 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "BatchNorm1d_3d");
    }
}
