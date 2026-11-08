using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public class SilkLayaNarrowParityTests
{
    const int Hidden = 8;
    const int Intermediate = 4 * Hidden; // 32
    const int EncoderIntermediate = 4;
    const int Layers = 2;
    const int Vocab = 16;
    const int HeadLayers = 2;

    static ModernBertConfig TinyConfig => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = 2,
        NumHiddenLayers = Layers,
        IntermediateSize = EncoderIntermediate,
        VocabSize = Vocab,
        LocalAttention = 6,
        GlobalAttnEveryNLayers = 2,
        MaxPositionEmbeddings = 16,
    };

    static Dictionary<string, (float[] Data, int[] Shape)> BuildEncoderStateDict(string prefix)
    {
        var tensors = new Dictionary<string, (float[], int[])>
        {
            [$"{prefix}.embeddings.tok_embeddings.weight"] = (Filled(Vocab, Hidden, 0f), new[] { Vocab, Hidden }),
            [$"{prefix}.embeddings.norm.weight"] = (Filled(Hidden, 1, 0f), new[] { Hidden }),
            [$"{prefix}.final_norm.weight"] = (Filled(Hidden, 1, 0f), new[] { Hidden }),
        };

        for (int i = 0; i < Layers; i++)
        {
            float s = i;
            if (i > 0)
                tensors[$"{prefix}.layers.{i}.attn_norm.weight"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{prefix}.layers.{i}.mlp_norm.weight"] = (Filled(Hidden, 1, s), new[] { Hidden });

            tensors[$"{prefix}.layers.{i}.attn.Wqkv.weight"] =
                (Filled(3 * Hidden, Hidden, s), new[] { 3 * Hidden, Hidden });
            tensors[$"{prefix}.layers.{i}.attn.Wo.weight"] = (Filled(Hidden, Hidden, s), new[] { Hidden, Hidden });

            tensors[$"{prefix}.layers.{i}.mlp.Wi.weight"] =
                (Filled(2 * EncoderIntermediate, Hidden, s), new[] { 2 * EncoderIntermediate, Hidden });
            tensors[$"{prefix}.layers.{i}.mlp.Wo.weight"] = (Filled(Hidden, EncoderIntermediate, s), new[] { Hidden, EncoderIntermediate });
        }

        return tensors.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static Dictionary<string, (float[] Data, int[] Shape)> BuildHeadTensors()
    {
        int seed = 0;
        var tensors = new Dictionary<string, (float[], int[])>();

        void Add(string key, int rows, int cols)
            => tensors[key] = (Weights(rows * cols, ref seed), new[] { rows, cols });

        void AddVec(string key, int length)
            => tensors[key] = (Weights(length, ref seed), new[] { length });

        tensors["type_emb.weight"] = (Weights(3 * Hidden, ref seed), new[] { 3, Hidden });

        for (int i = 0; i < HeadLayers; i++)
        {
            string p = $"head.layers.{i}";
            AddVec($"{p}.norm1.weight", Hidden);
            AddVec($"{p}.norm1.bias", Hidden);
            AddVec($"{p}.norm2.weight", Hidden);
            AddVec($"{p}.norm2.bias", Hidden);
            Add($"{p}.self_attn.in_proj_weight", 3 * Hidden, Hidden);
            AddVec($"{p}.self_attn.in_proj_bias", 3 * Hidden);
            Add($"{p}.self_attn.out_proj.weight", Hidden, Hidden);
            AddVec($"{p}.self_attn.out_proj.bias", Hidden);
            Add($"{p}.linear1.weight", Intermediate, Hidden);
            AddVec($"{p}.linear1.bias", Intermediate);
            Add($"{p}.linear2.weight", Hidden, Intermediate);
            AddVec($"{p}.linear2.bias", Hidden);
        }

        AddVec("scorer.0.weight", Hidden);
        AddVec("scorer.0.bias", Hidden);
        Add("scorer.1.weight", Hidden, Hidden);
        AddVec("scorer.1.bias", Hidden);
        Add("scorer.3.weight", 1, Hidden);
        AddVec("scorer.3.bias", 1);

        int actHidden = LayaDecisionHead<float, float>.ActHeadHidden;
        Add("act_head.0.weight", actHidden, Hidden + LayaHeadOutput.FeatureCount);
        AddVec("act_head.0.bias", actHidden);
        Add("act_head.2.weight", 2, actHidden);
        AddVec("act_head.2.bias", 2);

        return tensors.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static float[] Filled(int rows, int cols, float start)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
            data[i] = start + i;
        return data;
    }

    static float[] Weights(int count, ref int seed)
    {
        var data = new float[count];
        for (int i = 0; i < count; i++)
        {
            seed = seed * 1664525 + 1013904223;
            data[i] = (seed % 2000 - 1000) / 10000f;
        }
        return data;
    }

    static Dictionary<string, (float[] Data, int[] Shape)> Combine(
        Dictionary<string, (float[] Data, int[] Shape)> a,
        Dictionary<string, (float[] Data, int[] Shape)> b)
    {
        var r = new Dictionary<string, (float[] Data, int[] Shape)>(a);
        foreach (var kv in b) r[kv.Key] = kv.Value;
        return r;
    }

    [Test]
    public void Forward_Bf16_SilkMatchesF32ArgmaxAndBounded()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var config = TinyConfig;
        var tensors = Combine(BuildEncoderStateDict("encoder"), BuildHeadTensors());

        var bf16Tensors = new Dictionary<string, (BFloat16[] Data, int[] Shape)>();
        foreach (var kv in tensors)
        {
            var d = new BFloat16[kv.Value.Data.Length];
            for (int i = 0; i < d.Length; i++) d[i] = (BFloat16)kv.Value.Data[i];
            bf16Tensors[kv.Key] = (d, kv.Value.Shape);
        }

        int[] ids = [0, 1, 2, 3];
        int valid = 4;
        int[] markers = [0];

        using var f32Session = SilkLayaSession.Open(config, tensors, HeadLayers);
        using var bf16Session = SilkLayaSession.Open(config, bf16Tensors, HeadLayers);

        var f32Out = f32Session.Forward(ids, valid, LayaQuestionType.Choice, markers);
        var bf16Out = bf16Session.Forward(ids, valid, LayaQuestionType.Choice, markers);

        Assert.That(ArgMax(bf16Out.Logits), Is.EqualTo(ArgMax(f32Out.Logits)));

        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < f32Out.Logits.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(f32Out.Logits[i] - bf16Out.Logits[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(f32Out.Logits[i]));
        }
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-2f * scale));
    }

    [Test]
    public void Forward_Fp16_SilkMatchesF32ArgmaxAndBounded()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk parity verification."); }

        var config = TinyConfig;
        var tensors = Combine(BuildEncoderStateDict("encoder"), BuildHeadTensors());

        var halfTensors = new Dictionary<string, (Half[] Data, int[] Shape)>();
        foreach (var kv in tensors)
        {
            var d = new Half[kv.Value.Data.Length];
            for (int i = 0; i < d.Length; i++) d[i] = (Half)kv.Value.Data[i];
            halfTensors[kv.Key] = (d, kv.Value.Shape);
        }

        int[] ids = [0, 1, 2, 3];
        int valid = 4;
        int[] markers = [0];

        using var f32Session = SilkLayaSession.Open(config, tensors, HeadLayers);
        using var halfSession = SilkLayaSession.Open(config, halfTensors, HeadLayers);

        var f32Out = f32Session.Forward(ids, valid, LayaQuestionType.Choice, markers);
        var halfOut = halfSession.Forward(ids, valid, LayaQuestionType.Choice, markers);

        Assert.That(ArgMax(halfOut.Logits), Is.EqualTo(ArgMax(f32Out.Logits)));

        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < f32Out.Logits.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(f32Out.Logits[i] - halfOut.Logits[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(f32Out.Logits[i]));
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
