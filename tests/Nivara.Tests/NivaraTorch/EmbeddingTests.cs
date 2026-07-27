using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class EmbeddingTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void Embedding_SingleToken_MatchesPyTorch()
    {
        var weight = TestHelpers.LoadBin("emb_single_weight.bin");
        var expected = TestHelpers.LoadBin("emb_single_output.bin");

        using var emb = new Embedding<float>(100, 16);
        emb.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 100, 16, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(new float[] { 42 }, requiresGrad: false);

        var output = emb.Forward(inputTensor);

        Assert.That(output.Length, Is.EqualTo(16));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Embedding_single");
    }

    [Test]
    public void Embedding_Batch_MatchesPyTorch()
    {
        var weight = TestHelpers.LoadBin("emb_batch_weight.bin");
        var expected = TestHelpers.LoadBin("emb_batch_output.bin");

        using var emb = new Embedding<float>(100, 16);
        emb.WeightParam.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 100, 16, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(new float[] { 0, 13, 42, 99 }, requiresGrad: false);

        var output = emb.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Embedding_batch");
    }
}
