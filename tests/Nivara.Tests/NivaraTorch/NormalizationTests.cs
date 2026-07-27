using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class NormalizationTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void RMSNorm_PerRow_2D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("rmsnorm_2d_input.bin");
        var expected = TestHelpers.LoadBin("rmsnorm_2d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = ReverseGradOperations.PerRowRMSNorm(inputTensor, rows: 4, cols: 32, eps: 1e-5);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "RMSNorm_PerRow_2D");
    }

    [Test]
    public void RMSNorm_PerRow_3D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("rmsnorm_3d_input.bin");
        var expected = TestHelpers.LoadBin("rmsnorm_3d_output.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        var output = ReverseGradOperations.PerRowRMSNorm(inputTensor, rows: 8, cols: 32, eps: 1e-5);

        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "RMSNorm_PerRow_3D");
    }

    [Test]
    public void LayerNorm_2D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("layernorm_2d_input.bin");
        var gamma = TestHelpers.LoadBin("layernorm_2d_gamma.bin");
        var beta = TestHelpers.LoadBin("layernorm_2d_beta.bin");
        var expected = TestHelpers.LoadBin("layernorm_2d_output.bin");

        using var ln = new LayerNorm<float>(32, eps: 1e-5f, affine: true);
        ln.Weight!.Tensor = ReverseGradTensor<float>.FromArray(gamma, requiresGrad: false);
        ln.Bias!.Tensor = ReverseGradTensor<float>.FromArray(beta, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 32);

        var output = ln.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 32 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "LayerNorm_2D");
    }

    [Test]
    public void LayerNorm_3D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("layernorm_3d_input.bin");
        var gamma = TestHelpers.LoadBin("layernorm_3d_gamma.bin");
        var beta = TestHelpers.LoadBin("layernorm_3d_beta.bin");
        var expected = TestHelpers.LoadBin("layernorm_3d_output.bin");

        using var ln = new LayerNorm<float>(32, eps: 1e-5f, affine: true);
        ln.Weight!.Tensor = ReverseGradTensor<float>.FromArray(gamma, requiresGrad: false);
        ln.Bias!.Tensor = ReverseGradTensor<float>.FromArray(beta, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(2, 4, 32);

        var output = ln.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 2, 4, 32 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "LayerNorm_3D");
    }
}
