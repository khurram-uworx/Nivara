using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class SiluTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Silu_1D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("silu_1d_input.bin");
        var expectedOutput = TestHelpers.LoadBin("silu_1d_output.bin");
        var expectedGrad = TestHelpers.LoadBin("silu_1d_grad.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        var output = Activation.Silu(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 32 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "Silu_1D");
        TestHelpers.AssertTensorEqual(expectedGrad, TestHelpers.ExtractGrad(inputTensor), label: "Silu_1D_grad");
    }

    [Test]
    public void Silu_4D_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("silu_4d_input.bin");
        var expectedOutput = TestHelpers.LoadBin("silu_4d_output.bin");
        var expectedGrad = TestHelpers.LoadBin("silu_4d_grad.bin");

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(1, 8, 4, 4);
        var output = Activation.Silu(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 1, 8, 4, 4 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "Silu_4D");
        TestHelpers.AssertTensorEqual(expectedGrad, TestHelpers.ExtractGrad(inputTensor), label: "Silu_4D_grad");
    }
}