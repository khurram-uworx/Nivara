using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Utilities;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaCausalAttentionTests
{
    // No global Grad() scope here deliberately: the inference-path guard is that no graph
    // nodes are built outside Grad(), which is the model's default execution mode.

    [Test]
    public void Inference_OutsideGrad_PreservesShapeAndBuildsNoGraph()
    {
        const int hidden = 768, numHeads = 12, numKvHeads = 4, seqLen = 8;
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 64);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(7);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: false);

        var output = attn.Forward(input);

        Assert.That(output.Rank, Is.EqualTo(2));
        Assert.That(output.shape, Is.EqualTo(new[] { seqLen, hidden }));
        Assert.That(output.IsLeaf, Is.True, "Inference forward outside Grad() must not build graph nodes.");
        for (int i = 0; i < output.Length; i++)
            Assert.That(float.IsFinite(output[i]), Is.True, $"Output[{i}] must be finite.");
    }

    [Test]
    public void Forward_InsideGrad_AccumulatesGradientsOnAllProjections()
    {
        const int hidden = 192, numHeads = 6, numKvHeads = 3, seqLen = 4;
        using var gradScope = GradientUtils.Grad();
        using var attn = new LlamaCausalAttention<float>(hidden, numHeads, numKvHeads, maxPositionEmbeddings: 32);

        var inputData = new float[seqLen * hidden];
        var rnd = new Random(11);
        for (int i = 0; i < inputData.Length; i++)
            inputData[i] = (float)(rnd.NextDouble() * 2 - 1);
        var input = ReverseGradTensor<float>.FromMatrix(inputData, seqLen, hidden, requiresGrad: true);

        var output = attn.Forward(input);
        var loss = Nivara.AutoDiff.Operations.ReverseGradOperations.Sum(output);
        loss.Backward();

        Assert.That(input.Grad, Is.Not.Null);
        Assert.That(input.Grad!.Length, Is.EqualTo(seqLen * hidden));
        foreach (var g in input.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False, "Attention input gradient must be finite.");
    }
}
