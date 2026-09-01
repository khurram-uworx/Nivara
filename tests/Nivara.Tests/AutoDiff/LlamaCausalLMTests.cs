using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Utilities;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

[TestFixture]
public class LlamaCausalLMTests
{
    // No global Grad() scope: inference-by-default is the model contract.

    static LlamaForCausalLM<float> TinyModel()
        => new(vocabSize: 128, hiddenSize: 32, numHiddenLayers: 2, numHeads: 4, numKeyValueHeads: 2,
            intermediateSize: 64, rmsNormEps: 1e-5f, maxPositionEmbeddings: 32, ropeTheta: 10000f);

    [Test]
    public void Inference_OutsideGrad_ProducesLogitsAndBuildsNoGraph()
    {
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];

        // Token IDs are exact integers, but the Forward(int[]) overload takes the int array.
        var logits = model.Forward(tokens);

        Assert.That(logits.Rank, Is.EqualTo(2));
        Assert.That(logits.Shape[0], Is.EqualTo(tokens.Length));
        Assert.That(logits.Shape[1], Is.EqualTo(128)); // vocab size
        Assert.That(logits.IsLeaf, Is.True, "Causal LM inference outside Grad() must not build graph nodes.");
        for (int i = 0; i < logits.Length; i++)
            Assert.That(float.IsFinite(logits[i]), Is.True, $"Logit[{i}] must be finite.");
    }

    [Test]
    public void Forward_InsideGrad_AccumulatesGradientsOnEmbedding()
    {
        using var gradScope = GradientUtils.Grad();
        using var model = TinyModel();
        int[] tokens = [1, 12, 45, 78, 99];

        var logits = model.Forward(tokens);
        var loss = ReverseGradOperations.Sum(logits);
        loss.Backward();

        Assert.That(model.Embed.Weight!.Tensor.Grad, Is.Not.Null);
        Assert.That(model.Embed.Weight!.Tensor.Grad!.Length, Is.EqualTo(128 * 32));
        foreach (var g in model.Embed.Weight!.Tensor.Grad!)
            Assert.That(float.IsNaN(g) || float.IsInfinity(g), Is.False, "Embedding gradient must be finite.");
    }

    [Test]
    public void Forward_TiedHeadHasNoSeparateLmHead()
    {
        // The tied LM head must not add a second copy of the vocab weight. The only
        // vocabulary-sized ([vocab, hidden]) parameter is the embedding, which is reused for
        // the output projection. Verify no parameter name references a distinct lm_head and
        // that exactly one embedding-shaped parameter exists.
        using var model = TinyModel();
        var state = model.StateDict();

        Assert.That(state.Any(kv => kv.Key.Contains("lm_head", StringComparison.OrdinalIgnoreCase)), Is.False,
            "A tied LM head must not introduce a separate lm_head parameter.");

        int embeddingSized = state.Values.Count(t => t.Shape.Length == 2 && t.Shape[0] == 128 && t.Shape[1] == 32);
        Assert.That(embeddingSized, Is.EqualTo(1),
            "Exactly one [vocab, hidden]-shaped parameter (the embedding, shared as the LM head) must exist.");
    }
}
