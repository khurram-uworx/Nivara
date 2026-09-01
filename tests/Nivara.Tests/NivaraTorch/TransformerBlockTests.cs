using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class TransformerBlockTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void TransformerBlock_RmsNorm_MatchesPyTorch() => RunCase("transformer_block_rms", NormType.RMSNorm);

    [Test]
    public void TransformerBlock_LayerNorm_MatchesPyTorch() => RunCase("transformer_block_ln", NormType.LayerNorm);

    void RunCase(string prefix, NormType normType)
    {
        var input = TestHelpers.LoadBin($"{prefix}_input.bin");
        var qw = TestHelpers.LoadBin($"{prefix}_qw.bin");
        var kw = TestHelpers.LoadBin($"{prefix}_kw.bin");
        var vw = TestHelpers.LoadBin($"{prefix}_vw.bin");
        var ow = TestHelpers.LoadBin($"{prefix}_ow.bin");
        var f1w = TestHelpers.LoadBin($"{prefix}_f1w.bin");
        var f2w = TestHelpers.LoadBin($"{prefix}_f2w.bin");
        var expectedOutput = TestHelpers.LoadBin($"{prefix}_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin($"{prefix}_input_grad.bin");

        using var block = new TransformerBlock<float>(
            nEmbd: 32, nHead: 4, dropout: 0.0, maxSeqLen: 16, initStd: 0.02, normType: normType);
        var parameters = block.GetParameters();
        parameters["Module_0.Weight"].Tensor = ReverseGradTensor<float>.FromMatrix(qw, 32, 32, requiresGrad: false);
        parameters["Module_1.Weight"].Tensor = ReverseGradTensor<float>.FromMatrix(kw, 32, 32, requiresGrad: false);
        parameters["Module_2.Weight"].Tensor = ReverseGradTensor<float>.FromMatrix(vw, 32, 32, requiresGrad: false);
        parameters["Module_3.Weight"].Tensor = ReverseGradTensor<float>.FromMatrix(ow, 32, 32, requiresGrad: false);
        parameters["Module_4.Weight"].Tensor = ReverseGradTensor<float>.FromMatrix(f1w, 128, 32, requiresGrad: false);
        parameters["Module_5.Weight"].Tensor = ReverseGradTensor<float>.FromMatrix(f2w, 32, 128, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(6, 32);
        var output = block.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 6, 32 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: $"{prefix}_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), absTol: 1e-4f, relTol: 1e-3f, label: $"{prefix}_input_grad");
    }
}