using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class RotaryEmbeddingTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void RotaryEmbedding_1Head_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("rope_1head_input.bin");
        var expectedOutput = TestHelpers.LoadBin("rope_1head_output.bin");
        var expectedGrad = TestHelpers.LoadBin("rope_1head_grad.bin");

        using var rope = new RotaryEmbedding<float>(headDim: 8, maxPositionEmbeddings: 8, ropeTheta: 10000f);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(8, 8);
        var output = rope.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 8, 8 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "RotaryEmbedding_1Head");
        TestHelpers.AssertTensorEqual(expectedGrad, TestHelpers.ExtractGrad(inputTensor), label: "RotaryEmbedding_1Head_grad");
    }

    [Test]
    public void RotaryEmbedding_2Heads_MatchesPyTorch()
    {
        var input = TestHelpers.LoadBin("rope_2head_input.bin");
        var expectedOutput = TestHelpers.LoadBin("rope_2head_output.bin");
        var expectedGrad = TestHelpers.LoadBin("rope_2head_grad.bin");

        using var rope = new RotaryEmbedding<float>(headDim: 8, maxPositionEmbeddings: 8, ropeTheta: 10000f);
        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: true);
        inputTensor.Reshape(8, 16);
        var output = rope.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 8, 16 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "RotaryEmbedding_2Heads");
        TestHelpers.AssertTensorEqual(expectedGrad, TestHelpers.ExtractGrad(inputTensor), label: "RotaryEmbedding_2Heads_grad");
    }
}