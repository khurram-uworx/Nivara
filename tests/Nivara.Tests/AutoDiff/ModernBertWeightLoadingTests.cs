using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Covers how ModernBERT's fused weights are read back out of a safetensors dictionary: the
/// <see cref="StateDictLoader.LoadLinearSlice{TModel, TWeight}"/> row-block extraction, and the
/// tensor-prefix parameter that lets one loader serve the stock <c>model.</c> backbone and Laya's
/// nested <c>encoder.</c> copy of the same architecture.
/// </summary>
[TestFixture]
public class ModernBertWeightLoadingTests
{
    const int Hidden = 8;
    const int Heads = 2;
    const int Intermediate = 4;
    const int Layers = 3;
    const int Vocab = 32;

    static ModernBertConfig TinyConfig => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = Heads,
        NumHiddenLayers = Layers,
        IntermediateSize = Intermediate,
        VocabSize = Vocab,
        LocalAttention = 6,
        GlobalAttnEveryNLayers = 2,
    };

    static float[] Filled(int rows, int cols, float start)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = start + i;
        return data;
    }

    /// <summary>
    /// Builds a state dict with recognisable values: the fused matrices are filled row by row
    /// with a stride of 1, so a wrong row block is visible as a wrong offset.
    /// </summary>
    static Dictionary<string, (float[] Data, int[] Shape)> BuildStateDict(string prefix, float scale = 0f)
    {
        var tensors = new Dictionary<string, (float[], int[])>
        {
            [$"{prefix}.embeddings.tok_embeddings.weight"] = (Filled(Vocab, Hidden, scale), [Vocab, Hidden]),
            [$"{prefix}.embeddings.norm.weight"] = (Filled(Hidden, 1, scale), [Hidden]),
            [$"{prefix}.final_norm.weight"] = (Filled(Hidden, 1, scale), [Hidden]),
        };

        for (int i = 0; i < Layers; i++)
        {
            // A distinct scale per layer makes a mis-bound layer index visible too.
            float layerScale = scale + i;
            // Layer 0 has no attn_norm (it is an Identity in ModernBERT).
            if (i > 0)
                tensors[$"{prefix}.layers.{i}.attn_norm.weight"] = (Filled(Hidden, 1, layerScale), [Hidden]);
            tensors[$"{prefix}.layers.{i}.mlp_norm.weight"] = (Filled(Hidden, 1, layerScale), [Hidden]);

            tensors[$"{prefix}.layers.{i}.attn.Wqkv.weight"] =
                (Filled(3 * Hidden, Hidden, layerScale), [3 * Hidden, Hidden]);
            tensors[$"{prefix}.layers.{i}.attn.Wo.weight"] = (Filled(Hidden, Hidden, layerScale), [Hidden, Hidden]);

            tensors[$"{prefix}.layers.{i}.mlp.Wi.weight"] =
                (Filled(2 * Intermediate, Hidden, layerScale), [2 * Intermediate, Hidden]);
            tensors[$"{prefix}.layers.{i}.mlp.Wo.weight"] = (Filled(Hidden, Intermediate, layerScale), [Hidden, Intermediate]);
        }

        return tensors.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static float[] LinearWeight(Linear<float> linear)
    {
        var tensor = linear.Weight!.Tensor;
        Assert.That(tensor.Data.TryGetSpan(out var span), Is.True);
        return span.ToArray();
    }

    static float[] EmbeddingWeight(Embedding<float> embedding)
    {
        var tensor = embedding.Weight!.Tensor;
        Assert.That(tensor.Data.TryGetSpan(out var span), Is.True);
        return span.ToArray();
    }

    static float[] NormWeight(LayerNorm<float> norm)
    {
        var tensor = norm.Weight!.Tensor;
        Assert.That(tensor.Data.TryGetSpan(out var span), Is.True);
        return span.ToArray();
    }

    //  ── LoadLinearSlice ───────────────────────────────────────────────────────

    [Test]
    public void LoadLinearSlice_ExtractsTheRequestedRowBlock()
    {
        const int rows = 6, cols = 3;
        var fused = Filled(rows, cols, 0f);
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (fused, [rows, cols]),
        };

        var linear = new Linear<float>(cols, 3);
        StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", 3, 3);

        var expected = fused.Skip(9).Take(9).ToArray();
        Assert.That(LinearWeight(linear), Is.EqualTo(expected));
    }

    [Test]
    public void LoadLinearSlice_FirstBlockIsTheFirstRows()
    {
        const int rows = 6, cols = 3;
        var fused = Filled(rows, cols, 100f);
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (fused, [rows, cols]),
        };

        var linear = new Linear<float>(cols, 2);
        StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", 0, 2);

        Assert.That(LinearWeight(linear), Is.EqualTo(new[] { 100f, 101f, 102f, 103f, 104f, 105f }));
    }

    [Test]
    public void LoadLinearSlice_MissingTensor_ThrowsWithTheKeyName()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        var linear = new Linear<float>(4, 2);

        var ex = Assert.Throws<KeyNotFoundException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "attn.Wqkv", 0, 2));
        Assert.That(ex!.Message, Does.Contain("attn.Wqkv.weight"));
    }

    [Test]
    public void LoadLinearSlice_RowBlockPastTheEnd_Throws()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(6, 3, 0f), [6, 3]),
        };
        var linear = new Linear<float>(3, 3);

        var ex = Assert.Throws<InvalidOperationException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", 4, 3));
        Assert.That(ex!.Message, Does.Contain("exceeds"));
    }

    [Test]
    public void LoadLinearSlice_InFeaturesMismatch_Throws()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(6, 3, 0f), [6, 3]),
        };
        var linear = new Linear<float>(5, 3);

        Assert.Throws<InvalidOperationException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", 0, 3));
    }

    [Test]
    public void LoadLinearSlice_NonPositiveOrNegativeRanges_Throw()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(6, 3, 0f), [6, 3]),
        };
        var linear = new Linear<float>(3, 3);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", -1, 3));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(linear, tensors, "W", 0, -3));
    }

    [Test]
    public void LoadLinearSlice_NullTarget_Throws()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>
        {
            ["W.weight"] = (Filled(6, 3, 0f), [6, 3]),
        };

        Assert.Throws<ArgumentNullException>(
            () => StateDictLoader.LoadLinearSlice<float, float>(null!, tensors, "W", 0, 3));
    }

    //  ── fused QKV / gate-up binding ───────────────────────────────────────────

    [Test]
    public void LoadWeights_BindsQkvThirdsToConsecutiveRowBlocks()
    {
        var config = TinyConfig;
        var tensors = BuildStateDict("model");
        var encoder = ModernBertEncoder<float>.LoadWeights(tensors, config);

        var layer0 = encoder.layers[0];
        var wqkv = tensors["model.layers.0.attn.Wqkv.weight"];
        var qProj = LinearWeight(layer0.attn.qProj);
        var kProj = LinearWeight(layer0.attn.kProj);
        var vProj = LinearWeight(layer0.attn.vProj);

        Assert.That(qProj, Is.EqualTo(wqkv.Data.Take(Hidden * Hidden).ToArray()));
        Assert.That(kProj, Is.EqualTo(wqkv.Data.Skip(Hidden * Hidden).Take(Hidden * Hidden).ToArray()));
        Assert.That(vProj, Is.EqualTo(wqkv.Data.Skip(2 * Hidden * Hidden).Take(Hidden * Hidden).ToArray()));
    }

    [Test]
    public void LoadWeights_ActivatesTheFirstWiHalf_AndLeavesTheSecondLinear()
    {
        // HuggingFace: input, gate = Wi(x).chunk(2, dim=-1); out = act(input) * gate.
        // The activated projection is therefore the FIRST row block, despite HF naming the
        // unactivated one "gate". Getting this backwards is the one bug that cost cosine 0.82.
        var config = TinyConfig;
        var tensors = BuildStateDict("model");
        var encoder = ModernBertEncoder<float>.LoadWeights(tensors, config);

        var layer0 = encoder.layers[0];
        var wi = tensors["model.layers.0.mlp.Wi.weight"];
        int block = Intermediate * Hidden;

        Assert.That(LinearWeight(layer0.mlp.inputProj), Is.EqualTo(wi.Data.Take(block).ToArray()));
        Assert.That(LinearWeight(layer0.mlp.gateProj), Is.EqualTo(wi.Data.Skip(block).Take(block).ToArray()));
    }

    [Test]
    public void LoadWeights_LoadsEveryTensorAndSkipsLayerZeroAttnNorm()
    {
        var config = TinyConfig;
        var tensors = BuildStateDict("model");
        var encoder = ModernBertEncoder<float>.LoadWeights(tensors, config);

        Assert.That(encoder.layers[0].attnNorm, Is.Null, "layer 0's attn_norm is an Identity in ModernBERT");
        for (int i = 1; i < Layers; i++)
            Assert.That(encoder.layers[i].attnNorm, Is.Not.Null, $"layer {i} attn_norm");

        foreach (var layer in encoder.layers)
            Assert.That(layer.mlpNorm, Is.Not.Null);

        Assert.That(EmbeddingWeight(encoder.tokenEmbedding).Length, Is.EqualTo(Vocab * Hidden));
    }

    [Test]
    public void LoadWeights_NormWeightsCarryTheirOwnLayerScale()
    {
        // Binding is not enough: LoadLayerNorm silently leaves the default gamma (all 1.0) when the
        // key is missing, so a mistyped norm prefix yields a *plausible* untrained model rather than
        // an error. BuildStateDict gives layer i's norms the values [i, i+1, ... i+Hidden-1], which no
        // real checkpoint contains, so the content itself proves which tensor was bound.
        var config = TinyConfig;
        var encoder = ModernBertEncoder<float>.LoadWeights(BuildStateDict("model"), config);

        for (int i = 0; i < Layers; i++)
        {
            var expected = Filled(Hidden, 1, i);
            Assert.That(NormWeight(encoder.layers[i].mlpNorm), Is.EqualTo(expected), $"layer {i} mlp_norm");
            if (i > 0)
                Assert.That(NormWeight(encoder.layers[i].attnNorm!), Is.EqualTo(expected), $"layer {i} attn_norm");
        }

        Assert.That(NormWeight(encoder.embedNorm), Is.EqualTo(Filled(Hidden, 1, 0f)));
        Assert.That(NormWeight(encoder.finalNorm), Is.EqualTo(Filled(Hidden, 1, 0f)));
    }

    //  ── prefix parameter ─────────────────────────────────────────────────────

    [Test]
    public void LoadWeights_ModelAndEncoderPrefixesBindTheSameArchitecture()
    {
        var config = TinyConfig;
        var viaModel = ModernBertEncoder<float>.LoadWeights(BuildStateDict("model"), config, "model");
        var viaEncoder = ModernBertEncoder<float>.LoadWeights(BuildStateDict("encoder"), config, "encoder");

        Assert.That(LinearWeight(viaEncoder.layers[1].attn.qProj),
            Is.EqualTo(LinearWeight(viaModel.layers[1].attn.qProj)));
        Assert.That(LinearWeight(viaEncoder.layers[1].mlp.gateProj),
            Is.EqualTo(LinearWeight(viaModel.layers[1].mlp.gateProj)));
        Assert.That(EmbeddingWeight(viaEncoder.tokenEmbedding),
            Is.EqualTo(EmbeddingWeight(viaModel.tokenEmbedding)));
    }

    [Test]
    public void LoadWeights_UnknownPrefix_ThrowsNamingTheFirstMissingTensor()
    {
        // A mistyped prefix must fail loudly rather than silently yielding an untrained encoder.
        var config = TinyConfig;
        var tensors = BuildStateDict("model");

        var ex = Assert.Throws<KeyNotFoundException>(
            () => ModernBertEncoder<float>.LoadWeights(tensors, config, "encoder"));
        Assert.That(ex!.Message, Does.Contain("encoder.embeddings.tok_embeddings.weight"));
    }

    [Test]
    public void LoadWeights_DistinctPerLayerScales_AreBoundToTheRightLayer()
    {
        var config = TinyConfig;
        var tensors = BuildStateDict("model");
        var encoder = ModernBertEncoder<float>.LoadWeights(tensors, config);

        for (int i = 0; i < Layers; i++)
        {
            var expected = tensors[$"model.layers.{i}.attn.Wqkv.weight"].Data.Take(Hidden * Hidden).ToArray();
            Assert.That(LinearWeight(encoder.layers[i].attn.qProj), Is.EqualTo(expected), $"layer {i}");
        }
    }
}
