using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
public class SilkBertNarrowParityTests
{
    const int Hidden = 8;
    const int Heads = 2;
    const int Intermediate = 4;
    const int Layers = 2;
    const int Vocab = 32;
    const int MaxPos = 16;

    static BertConfig TinyBertConfig => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = Heads,
        NumHiddenLayers = Layers,
        IntermediateSize = Intermediate,
        VocabSize = Vocab,
        MaxPositionEmbeddings = MaxPos,
    };

    static Dictionary<string, (float[] Data, int[] Shape)> BuildMini()
    {
        var tensors = new Dictionary<string, (float[], int[])>
        {
            ["embeddings.word_embeddings.weight"] = (Filled(Vocab, Hidden, 0f), new[] { Vocab, Hidden }),
            ["embeddings.position_embeddings.weight"] = (Filled(MaxPos, Hidden, 0f), new[] { MaxPos, Hidden }),
            ["embeddings.LayerNorm.weight"] = (Filled(Hidden, 1, 0f), new[] { Hidden }),
            ["embeddings.LayerNorm.bias"] = (Filled(Hidden, 1, 0f), new[] { Hidden }),
        };
        for (int i = 0; i < Layers; i++)
        {
            float s = i;
            string layer = $"encoder.layer.{i}";
            if (i > 0) // mini attention norm only exists? keep simple
            {
            }
            tensors[$"{layer}.attention.self.query.weight"] = (Filled(Hidden, Hidden, s), new[] { Hidden, Hidden });
            tensors[$"{layer}.attention.self.query.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.attention.self.key.weight"] = (Filled(Hidden, Hidden, s + 1), new[] { Hidden, Hidden });
            tensors[$"{layer}.attention.self.key.bias"] = (Filled(Hidden, 1, s + 1), new[] { Hidden });
            tensors[$"{layer}.attention.self.value.weight"] = (Filled(Hidden, Hidden, s + 2), new[] { Hidden, Hidden });
            tensors[$"{layer}.attention.self.value.bias"] = (Filled(Hidden, 1, s + 2), new[] { Hidden });
            tensors[$"{layer}.attention.output.dense.weight"] = (Filled(Hidden, Hidden, s + 3), new[] { Hidden, Hidden });
            tensors[$"{layer}.attention.output.dense.bias"] = (Filled(Hidden, 1, s + 3), new[] { Hidden });
            tensors[$"{layer}.attention.output.LayerNorm.weight"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.attention.output.LayerNorm.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });

            tensors[$"{layer}.intermediate.dense.weight"] = (Filled(Intermediate, Hidden, s), new[] { Intermediate, Hidden });
            tensors[$"{layer}.intermediate.dense.bias"] = (Filled(Intermediate, 1, s), new[] { Intermediate });
            tensors[$"{layer}.output.dense.weight"] = (Filled(Hidden, Intermediate, s), new[] { Hidden, Intermediate });
            tensors[$"{layer}.output.dense.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.output.LayerNorm.weight"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.output.LayerNorm.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
        }
        return tensors.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static Dictionary<string, (float[] Data, int[] Shape)> BuildDistil()
    {
        var tensors = new Dictionary<string, (float[], int[])>
        {
            ["distilbert.embeddings.word_embeddings.weight"] = (Filled(Vocab, Hidden, 0f), new[] { Vocab, Hidden }),
            ["distilbert.embeddings.position_embeddings.weight"] = (Filled(MaxPos, Hidden, 0f), new[] { MaxPos, Hidden }),
            ["distilbert.embeddings.LayerNorm.weight"] = (Filled(Hidden, 1, 0f), new[] { Hidden }),
            ["distilbert.embeddings.LayerNorm.bias"] = (Filled(Hidden, 1, 0f), new[] { Hidden }),
        };
        for (int i = 0; i < Layers; i++)
        {
            float s = i;
            string layer = $"distilbert.transformer.layer.{i}";
            tensors[$"{layer}.sa_layer_norm.weight"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.sa_layer_norm.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
            foreach (var proj in new[] { "q_lin", "k_lin", "v_lin", "out_lin" })
            {
                tensors[$"{layer}.attention.{proj}.weight"] = (Filled(Hidden, Hidden, s), new[] { Hidden, Hidden });
                tensors[$"{layer}.attention.{proj}.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
            }
            tensors[$"{layer}.output_layer_norm.weight"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.output_layer_norm.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
            tensors[$"{layer}.ffn.lin1.weight"] = (Filled(Intermediate, Hidden, s), new[] { Intermediate, Hidden });
            tensors[$"{layer}.ffn.lin1.bias"] = (Filled(Intermediate, 1, s), new[] { Intermediate });
            tensors[$"{layer}.ffn.lin2.weight"] = (Filled(Hidden, Intermediate, s), new[] { Hidden, Intermediate });
            tensors[$"{layer}.ffn.lin2.bias"] = (Filled(Hidden, 1, s), new[] { Hidden });
        }
        tensors["pre_classifier.weight"] = (Filled(Hidden, Hidden, 0f), new[] { Hidden, Hidden });
        tensors["pre_classifier.bias"] = (Filled(Hidden, 1, 0f), new[] { Hidden });
        tensors["classifier.weight"] = (Filled(2, Hidden, 0f), new[] { 2, Hidden });
        tensors["classifier.bias"] = (Filled(2, 1, 0f), new[] { 2 });
        return tensors.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static float[] Filled(int rows, int cols, float start)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = start + i;
        return data;
    }

    [Test]
    public void MiniLM_Bf16_MatchesF32()
    {
        GpuTestGuard.RequireGpu();

        var config = TinyBertConfig;
        var f32 = BuildMini();
        var bf16 = ToBFloat16(f32);

        int[] ids = [0, 1, 2, 3];
        float[] mask = [1f, 1f, 1f, 1f];

        using var rF32 = new SilkRuntime();
        using var runF32 = new SilkBertEncoderRunner(rF32, config, f32, BertGpuNaming.Bert);
        using var rNarrow = new SilkRuntime();
        using var runNarrow = new SilkBertEncoderRunner(rNarrow, config, bf16, BertGpuNaming.Bert);

        var outF32 = runF32.Forward(ids, mask, 1, 4);
        var outNarrow = runNarrow.Forward(ids, mask, 1, 4);

        AssertClose(outNarrow.Hidden, outF32.Hidden);
    }

    [Test]
    public void MiniLM_Fp16_MatchesF32()
    {
        GpuTestGuard.RequireGpu();

        var config = TinyBertConfig;
        var f32 = BuildMini();
        var fp16 = ToHalf(f32);

        int[] ids = [0, 1, 2, 3];
        float[] mask = [1f, 1f, 1f, 1f];

        using var rF32 = new SilkRuntime();
        using var runF32 = new SilkBertEncoderRunner(rF32, config, f32, BertGpuNaming.Bert);
        using var rNarrow = new SilkRuntime();
        using var runNarrow = new SilkBertEncoderRunner(rNarrow, config, fp16, BertGpuNaming.Bert);

        var outF32 = runF32.Forward(ids, mask, 1, 4);
        var outNarrow = runNarrow.Forward(ids, mask, 1, 4);

        AssertClose(outNarrow.Hidden, outF32.Hidden);
    }

    [Test]
    public void DistilBert_Bf16_MatchesF32()
    {
        GpuTestGuard.RequireGpu();

        var config = TinyBertConfig;
        var f32 = BuildDistil();
        var bf16 = ToBFloat16(f32);

        int[] ids = [0, 1, 2, 3];
        float[] mask = [1f, 1f, 1f, 1f];

        using var rF32 = new SilkRuntime();
        using var runF32 = new SilkBertEncoderRunner(rF32, config, f32, BertGpuNaming.DistilBert);
        using var rNarrow = new SilkRuntime();
        using var runNarrow = new SilkBertEncoderRunner(rNarrow, config, bf16, BertGpuNaming.DistilBert);

        var outF32 = runF32.Forward(ids, mask, 1, 4);
        var outNarrow = runNarrow.Forward(ids, mask, 1, 4);

        AssertClose(outNarrow.Hidden, outF32.Hidden);
        if (outF32.Logits != null && outNarrow.Logits != null)
            AssertClose(outNarrow.Logits, outF32.Logits);
    }

    [Test]
    public void DistilBert_Fp16_MatchesF32()
    {
        GpuTestGuard.RequireGpu();

        var config = TinyBertConfig;
        var f32 = BuildDistil();
        var fp16 = ToHalf(f32);

        int[] ids = [0, 1, 2, 3];
        float[] mask = [1f, 1f, 1f, 1f];

        using var rF32 = new SilkRuntime();
        using var runF32 = new SilkBertEncoderRunner(rF32, config, f32, BertGpuNaming.DistilBert);
        using var rNarrow = new SilkRuntime();
        using var runNarrow = new SilkBertEncoderRunner(rNarrow, config, fp16, BertGpuNaming.DistilBert);

        var outF32 = runF32.Forward(ids, mask, 1, 4);
        var outNarrow = runNarrow.Forward(ids, mask, 1, 4);

        AssertClose(outNarrow.Hidden, outF32.Hidden);
        if (outF32.Logits != null && outNarrow.Logits != null)
            AssertClose(outNarrow.Logits, outF32.Logits);
    }

    static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(Dictionary<string, (float[] Data, int[] Shape)> src)
    {
        var result = new Dictionary<string, (BFloat16[], int[])>(src.Count);
        foreach (var kv in src)
        {
            var d = new BFloat16[kv.Value.Data.Length];
            for (int i = 0; i < d.Length; i++) d[i] = (BFloat16)kv.Value.Data[i];
            result[kv.Key] = (d, kv.Value.Shape);
        }
        return result.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static Dictionary<string, (Half[] Data, int[] Shape)> ToHalf(Dictionary<string, (float[] Data, int[] Shape)> src)
    {
        var result = new Dictionary<string, (Half[], int[])>(src.Count);
        foreach (var kv in src)
        {
            var d = new Half[kv.Value.Data.Length];
            for (int i = 0; i < d.Length; i++) d[i] = (Half)kv.Value.Data[i];
            result[kv.Key] = (d, kv.Value.Shape);
        }
        return result.ToDictionary(kv => kv.Key, kv => (kv.Value.Item1, kv.Value.Item2));
    }

    static void AssertClose(float[] actual, float[] expected)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length));
        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[i] - actual[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(expected[i]));
        }
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-2f * scale));
    }
}
