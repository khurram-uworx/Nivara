using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class LlamaCausalAttentionTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void LlamaCausalAttention_GqaRoPeCausalMask_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("llama_attn_input.bin");
        var qw = TestHelpers.LoadBin("llama_attn_qw.bin");
        var kw = TestHelpers.LoadBin("llama_attn_kw.bin");
        var vw = TestHelpers.LoadBin("llama_attn_vw.bin");
        var ow = TestHelpers.LoadBin("llama_attn_ow.bin");
        var expectedOutput = TestHelpers.LoadBin("llama_attn_output.bin");
        var expectedInputGrad = TestHelpers.LoadBin("llama_attn_input_grad.bin");

        using var attn = new LlamaCausalAttention<float>(
            hiddenSize: 64, numHeads: 4, numKeyValueHeads: 2, maxPositionEmbeddings: 16, ropeTheta: 10000f);
        attn.QProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(qw, 64, 64, requiresGrad: false);
        attn.KProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(kw, 32, 64, requiresGrad: false);
        attn.VProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(vw, 32, 64, requiresGrad: false);
        attn.OProj.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(ow, 64, 64, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(5, 64);
        var output = attn.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 5, 64 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "LlamaCausalAttention_output");
        TestHelpers.AssertTensorEqual(expectedInputGrad, TestHelpers.ExtractGrad(inputTensor), absTol: 1e-4f, relTol: 1e-3f, label: "LlamaCausalAttention_input_grad");
    }
}