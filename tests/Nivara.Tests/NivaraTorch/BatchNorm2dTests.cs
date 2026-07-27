using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class BatchNorm2dTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void BatchNorm2d_Eval_16ch_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn2d_16ch_input.bin");
        var gamma = TestHelpers.LoadBin("bn2d_16ch_gamma.bin");
        var beta = TestHelpers.LoadBin("bn2d_16ch_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn2d_16ch_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn2d_16ch_running_var.bin");
        var expected = TestHelpers.LoadBin("bn2d_16ch_output.bin");

        using var bn = new BatchNorm2d<float>(16);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 16, 5, 5);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 16, 5, 5 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "BatchNorm2d_16ch_eval");
    }

    [Test]
    public void BatchNorm2d_Eval_3ch_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn2d_3ch_input.bin");
        var gamma = TestHelpers.LoadBin("bn2d_3ch_gamma.bin");
        var beta = TestHelpers.LoadBin("bn2d_3ch_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn2d_3ch_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn2d_3ch_running_var.bin");
        var expected = TestHelpers.LoadBin("bn2d_3ch_output.bin");

        using var bn = new BatchNorm2d<float>(3);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(1, 3, 7, 7);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 3, 7, 7 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "BatchNorm2d_3ch_eval");
    }

    [Test]
    public void BatchNorm2d_Eval_Batch4_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("bn2d_batch4_input.bin");
        var gamma = TestHelpers.LoadBin("bn2d_batch4_gamma.bin");
        var beta = TestHelpers.LoadBin("bn2d_batch4_beta.bin");
        var runningMean = TestHelpers.LoadBin("bn2d_batch4_running_mean.bin");
        var runningVar = TestHelpers.LoadBin("bn2d_batch4_running_var.bin");
        var expected = TestHelpers.LoadBin("bn2d_batch4_output.bin");

        using var bn = new BatchNorm2d<float>(16);
        bn.LoadStateDict(new Dictionary<string, ReverseGradTensor<float>>
        {
            ["Weight"] = ReverseGradTensor<float>.FromArray(gamma),
            ["Bias"] = ReverseGradTensor<float>.FromArray(beta),
            ["running_mean"] = ReverseGradTensor<float>.FromArray(runningMean),
            ["running_var"] = ReverseGradTensor<float>.FromArray(runningVar),
        });
        bn.Eval();

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 16, 8, 8);

        var output = bn.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16, 8, 8 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), absTol: 1e-5f, relTol: 1e-4f, label: "BatchNorm2d_batch4_eval");
    }
}
