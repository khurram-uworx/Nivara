using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public class SilkModernBertNarrowParityTests
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
        MaxPositionEmbeddings = 16,
    };

    static Dictionary<string, (float[] Data, int[] Shape)> BuildStateDict(string prefix, float scale = 0f)
    {
        var tensors = new Dictionary<string, (float[], int[])>
        {
            [$"{prefix}.embeddings.tok_embeddings.weight"] = (Filled(Vocab, Hidden, scale), new[] { Vocab, Hidden }),
            [$"{prefix}.embeddings.norm.weight"] = (Filled(Hidden, 1, scale), new[] { Hidden }),
            [$"{prefix}.final_norm.weight"] = (Filled(Hidden, 1, scale), new[] { Hidden }),
        };

        for (int i = 0; i < Layers; i++)
        {
            float layerScale = scale + i;
            if (i > 0)
                tensors[$"{prefix}.layers.{i}.attn_norm.weight"] = (Filled(Hidden, 1, layerScale), new[] { Hidden });
            tensors[$"{prefix}.layers.{i}.mlp_norm.weight"] = (Filled(Hidden, 1, layerScale), new[] { Hidden });

            tensors[$"{prefix}.layers.{i}.attn.Wqkv.weight"] =
                (Filled(3 * Hidden, Hidden, layerScale), new[] { 3 * Hidden, Hidden });
            tensors[$"{prefix}.layers.{i}.attn.Wo.weight"] = (Filled(Hidden, Hidden, layerScale), new[] { Hidden, Hidden });

            tensors[$"{prefix}.layers.{i}.mlp.Wi.weight"] =
                (Filled(2 * Intermediate, Hidden, layerScale), new[] { 2 * Intermediate, Hidden });
            tensors[$"{prefix}.layers.{i}.mlp.Wo.weight"] = (Filled(Hidden, Intermediate, layerScale), new[] { Hidden, Intermediate });
        }

        return tensors.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static float[] Filled(int rows, int cols, float start)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = start + i;
        return data;
    }

    [Test]
    public void Forward_Bf16_SilkMatchesF32ArgmaxAndBounded()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var config = TinyConfig;
        var tensors = BuildStateDict("model");

        var bf16Tensors = new Dictionary<string, (BFloat16[] Data, int[] Shape)>();
        foreach (var kv in tensors)
        {
            var d = new BFloat16[kv.Value.Data.Length];
            for (int i = 0; i < d.Length; i++) d[i] = (BFloat16)kv.Value.Data[i];
            bf16Tensors[kv.Key] = (d, kv.Value.Shape);
        }

        int[] ids = [0, 1, 2, 3];
        int valid = 4;

        using var f32Session = SilkModernBertSession.Open(config, tensors);
        using var bf16Session = SilkModernBertSession.Open(config, bf16Tensors);

        var f32Out = f32Session.Forward(ids, valid);
        var bf16Out = bf16Session.Forward(ids, valid);

        Assert.That(bf16Out.Length, Is.EqualTo(f32Out.Length));
        Assert.That(ArgMax(bf16Out), Is.EqualTo(ArgMax(f32Out)));

        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < f32Out.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(f32Out[i] - bf16Out[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(f32Out[i]));
        }
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-2f * scale));
    }

    [Test]
    public void Forward_Fp16_SilkMatchesF32ArgmaxAndBounded()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var config = TinyConfig;
        var tensors = BuildStateDict("model");

        var halfTensors = new Dictionary<string, (Half[] Data, int[] Shape)>();
        foreach (var kv in tensors)
        {
            var d = new Half[kv.Value.Data.Length];
            for (int i = 0; i < d.Length; i++) d[i] = (Half)kv.Value.Data[i];
            halfTensors[kv.Key] = (d, kv.Value.Shape);
        }

        int[] ids = [0, 1, 2, 3];
        int valid = 4;

        using var f32Session = SilkModernBertSession.Open(config, tensors);
        using var halfSession = SilkModernBertSession.Open(config, halfTensors);

        var f32Out = f32Session.Forward(ids, valid);
        var halfOut = halfSession.Forward(ids, valid);

        Assert.That(halfOut.Length, Is.EqualTo(f32Out.Length));
        Assert.That(ArgMax(halfOut), Is.EqualTo(ArgMax(f32Out)));

        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < f32Out.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(f32Out[i] - halfOut[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(f32Out[i]));
        }
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-2f * scale));
    }

    static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}