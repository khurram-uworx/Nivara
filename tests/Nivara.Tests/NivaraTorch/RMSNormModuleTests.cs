using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class RMSNormModuleTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void RMSNormModule_2D_AffineGamma_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("rmsnorm_module_2d_input.bin");
        var gamma = TestHelpers.LoadBin("rmsnorm_module_2d_gamma.bin");
        var expectedOutput = TestHelpers.LoadBin("rmsnorm_module_2d_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("rmsnorm_module_2d_input_grad.bin");
        var expectedGammaGrad = TestHelpers.LoadBin("rmsnorm_module_2d_gamma_grad.bin");

        using var rms = new RMSNorm<float>(32, eps: 1e-5f);
        rms.Weight!.Tensor = ReverseGradTensor<float>.FromArray(gamma, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(4, 32);
        var output = rms.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 32 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "RMSNormModule_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), label: "RMSNormModule_input_grad");
        TestHelpers.AssertTensorEqual(expectedGammaGrad, TestHelpers.ExtractGrad(rms.Weight!.Tensor), label: "RMSNormModule_gamma_grad");
    }
}