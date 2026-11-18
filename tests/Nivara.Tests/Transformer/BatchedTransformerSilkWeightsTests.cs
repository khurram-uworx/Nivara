using NivaraChat.Transformer;
using NUnit.Framework;

namespace Nivara.Tests.Transformer;

[TestFixture]
public class BatchedTransformerSilkWeightsTests
{
    [Test]
    public void ToSilkWeights_ReturnsExpectedKeysAndShapes()
    {
        const int vocabSize = 7;
        const int nEmbd = 4;
        const int nLayer = 2;
        const int nHead = 2;
        const int maxSeqLen = 8;

        using var model = new BatchedTransformer<float>(vocabSize, nEmbd, nLayer, nHead, maxSeqLen);
        var weights = model.ToSilkWeights();

        Assert.That(weights.Count, Is.EqualTo(3 + 12 * nLayer));

        Assert.That(weights, Contains.Key("tok_emb.weight"));
        Assert.That(weights["tok_emb.weight"].Shape, Is.EqualTo(new[] { vocabSize, nEmbd }));

        Assert.That(weights, Contains.Key("final_norm.weight"));
        Assert.That(weights["final_norm.weight"].Shape, Is.EqualTo(new[] { nEmbd }));
        Assert.That(weights, Contains.Key("final_norm.bias"));
        Assert.That(weights["final_norm.bias"].Shape, Is.EqualTo(new[] { nEmbd }));

        for (int l = 0; l < nLayer; l++)
        {
            string p = $"layers.{l}.";
            Assert.That(weights, Contains.Key(p + "ln1.weight"));
            Assert.That(weights[p + "ln1.weight"].Shape, Is.EqualTo(new[] { nEmbd }));
            Assert.That(weights, Contains.Key(p + "ln1.bias"));
            Assert.That(weights[p + "ln1.bias"].Shape, Is.EqualTo(new[] { nEmbd }));
            Assert.That(weights, Contains.Key(p + "ln2.weight"));
            Assert.That(weights, Contains.Key(p + "ln2.bias"));
            Assert.That(weights, Contains.Key(p + "q_proj.weight"));
            Assert.That(weights[p + "q_proj.weight"].Shape, Is.EqualTo(new[] { nEmbd, nEmbd }));
            Assert.That(weights, Contains.Key(p + "k_proj.weight"));
            Assert.That(weights, Contains.Key(p + "v_proj.weight"));
            Assert.That(weights, Contains.Key(p + "o_proj.weight"));
            Assert.That(weights, Contains.Key(p + "mlp_fc1.weight"));
            Assert.That(weights[p + "mlp_fc1.weight"].Shape, Is.EqualTo(new[] { 4 * nEmbd, nEmbd }));
            Assert.That(weights, Contains.Key(p + "mlp_fc1.bias"));
            Assert.That(weights[p + "mlp_fc1.bias"].Shape, Is.EqualTo(new[] { 1, 4 * nEmbd }));
            Assert.That(weights, Contains.Key(p + "mlp_fc2.weight"));
            Assert.That(weights[p + "mlp_fc2.weight"].Shape, Is.EqualTo(new[] { nEmbd, 4 * nEmbd }));
            Assert.That(weights, Contains.Key(p + "mlp_fc2.bias"));
            Assert.That(weights[p + "mlp_fc2.bias"].Shape, Is.EqualTo(new[] { 1, nEmbd }));
        }
    }

    [Test]
    public void ToSilkWeights_CopiesData_NotAlias()
    {
        using var model = new BatchedTransformer<float>(vocabSize: 5, nEmbd: 3, nLayer: 1, nHead: 1, maxSeqLen: 4);
        var weights = model.ToSilkWeights();
        var arr = weights["tok_emb.weight"].Data;
        arr[0] = float.NaN;
        // Re-export should not reflect mutation (fresh copy).
        var weights2 = model.ToSilkWeights();
        Assert.That(float.IsNaN(weights2["tok_emb.weight"].Data[0]), Is.False);
    }
}
