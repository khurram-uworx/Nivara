using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.NivaraTorch;

[TestFixture]
public class SparseEmbeddingTests
{
    IDisposable? gradScope;

    [SetUp]
    public void SetUp() => gradScope = GradientUtils.Grad();

    [TearDown]
    public void TearDown() => gradScope?.Dispose();

    [Test]
    public void SparseEmbedding_SumBagPaddingIndex_MatchesPyTorch()
    {
        var weight = TestHelpers.LoadBin("sparse_embedding_weight.bin");
        var input = TestHelpers.LoadBin("sparse_embedding_input.bin");
        var expectedOutput = TestHelpers.LoadBin("sparse_embedding_output.bin");
        var expectedWeightGrad = TestHelpers.LoadBin("sparse_embedding_weight_grad.bin");

        using var sparse = new SparseEmbedding<float>(20, 8, paddingIndex: -1);
        sparse.Weight!.Tensor = ReverseGradTensor<float>.FromMatrix(weight, 20, 8, requiresGrad: true);

        var inputTensor = ReverseGradTensor<float>.FromArray(input, requiresGrad: false);
        inputTensor.Reshape(4, 5);
        var output = sparse.Forward(inputTensor);
        ReverseGradOperations.Sum(output).Backward();

        Assert.That(output.Shape, Is.EqualTo(new[] { 4, 8 }));
        TestHelpers.AssertTensorEqual(expectedOutput, TestHelpers.ExtractOutput(output), label: "SparseEmbedding_output");
        TestHelpers.AssertTensorEqual(expectedWeightGrad, TestHelpers.ExtractGrad(sparse.Weight!.Tensor), label: "SparseEmbedding_weight_grad");
    }
}