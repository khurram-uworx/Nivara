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
        emb.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 100, 16, requiresGrad: false);

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
        emb.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 100, 16, requiresGrad: false);

        var inputTensor = ReverseGradTensor<float>.FromArray(new float[] { 0, 13, 42, 99 }, requiresGrad: false);

        var output = emb.Forward(inputTensor);

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 16 }));
        TestHelpers.AssertTensorEqual(expected, TestHelpers.ExtractOutput(output), label: "Embedding_batch");
    }

    [Test]
    public void Embedding_GatherBackward_GradientFlows()
    {
        using var emb = new Embedding<float>(50, 8);

        var inputTensor = ReverseGradTensor<float>.FromArray(new float[] { 5f, 10f, 25f }, requiresGrad: false);

        var output = emb.Forward(inputTensor);

        var gradOutput = new ReverseGradTensor<float>(
            NivaraColumn<float>.Create(Enumerable.Repeat(1f, output.Length).ToArray()),
            requiresGrad: false);
        gradOutput.Reshape(output.Shape);

        output.Backward(gradOutput);

        Assert.That(emb.Weight!.Tensor.Grad, Is.Not.Null, "Embedding weight should have gradients after backward");
        var weightGrad = emb.Weight!.Tensor.Grad!;
        Assert.That(weightGrad.Length, Is.EqualTo(50 * 8));

        Assert.That(weightGrad[5 * 8], Is.EqualTo(1f),
            "Gradient at row 5 should accumulate from first lookup");
        Assert.That(weightGrad[10 * 8], Is.EqualTo(1f),
            "Gradient at row 10 should accumulate from second lookup");
        Assert.That(weightGrad[25 * 8], Is.EqualTo(1f),
            "Gradient at row 25 should accumulate from third lookup");

        int zeroCount = 0;
        for (int i = 0; i < weightGrad.Length; i++)
        {
            bool isRow5 = i >= 5 * 8 && i < 6 * 8;
            bool isRow10 = i >= 10 * 8 && i < 11 * 8;
            bool isRow25 = i >= 25 * 8 && i < 26 * 8;
            if (!isRow5 && !isRow10 && !isRow25 && weightGrad[i] == 0f)
                zeroCount++;
        }
        Assert.That(zeroCount, Is.EqualTo(50 * 8 - 3 * 8),
            "All non-looked-up rows should have zero gradient");
    }

    [Test]
    public void Embedding_SingleTokenBackward_GradientToCorrectRow()
    {
        using var emb = new Embedding<float>(20, 4);

        var inputTensor = ReverseGradTensor<float>.FromArray(new float[] { 7f }, requiresGrad: false);

        var output = emb.Forward(inputTensor);

        var gradData = new float[1 * 4];
        gradData[0] = 1f; gradData[1] = 2f; gradData[2] = 3f; gradData[3] = 4f;
        var gradCol = NivaraColumn<float>.Create(gradData);
        var gradOutput = new ReverseGradTensor<float>(gradCol, requiresGrad: false);
        gradOutput.Reshape(1, 4);

        output.Backward(gradOutput);

        Assert.That(emb.Weight!.Tensor.Grad, Is.Not.Null);
        var weightGrad = emb.Weight!.Tensor.Grad!;

        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                int idx = i * 4 + j;
                if (i == 7)
                    Assert.That(weightGrad[idx], Is.EqualTo(j + 1f),
                        $"Grad at [{i},{j}] should be {j + 1}");
                else
                    Assert.That(weightGrad[idx], Is.EqualTo(0f),
                        $"Grad at [{i},{j}] should be 0 for non-looked-up row");
            }
        }
    }
}
