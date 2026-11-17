using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

internal static class SilkLayaHeadBf16Fixtures
{
    internal const int HeadLayers = 1;
    internal const int Hidden = 8;
    internal const int Intermediate = 32;
    internal const int SeqLen = 6;
    internal const int ActClasses = 2;
    internal const int ValidLength = 6;

    internal static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk Laya head verification."); throw; }
    }

    internal static Dictionary<string, (float[] Data, int[] Shape)> Checkpoint(int seed = 1)
    {
        var rng = new Random(seed);
        var t = new Dictionary<string, (float[] Data, int[] Shape)>();

        t["type_emb.weight"] = (Matrix(3, Hidden, rng), [3, Hidden]);

        for (int i = 0; i < HeadLayers; i++)
        {
            string prefix = $"head.layers.{i}";
            string attn = $"{prefix}.self_attn";
            t[$"{prefix}.norm1.weight"] = (Ones(Hidden), [Hidden]);
            t[$"{prefix}.norm1.bias"] = (new float[Hidden], [Hidden]);
            // fused in_proj
            t[$"{attn}.in_proj_weight"] = (Matrix(3 * Hidden, Hidden, rng), [3 * Hidden, Hidden]);
            t[$"{attn}.in_proj_bias"] = (new float[3 * Hidden], [3 * Hidden]);
            t[$"{attn}.out_proj.weight"] = (Matrix(Hidden, Hidden, rng), [Hidden, Hidden]);
            t[$"{attn}.out_proj.bias"] = (new float[Hidden], [Hidden]);
            t[$"{prefix}.norm2.weight"] = (Ones(Hidden), [Hidden]);
            t[$"{prefix}.norm2.bias"] = (new float[Hidden], [Hidden]);
            t[$"{prefix}.linear1.weight"] = (Matrix(Intermediate, Hidden, rng), [Intermediate, Hidden]);
            t[$"{prefix}.linear1.bias"] = (new float[Intermediate], [Intermediate]);
            t[$"{prefix}.linear2.weight"] = (Matrix(Hidden, Intermediate, rng), [Hidden, Intermediate]);
            t[$"{prefix}.linear2.bias"] = (new float[Hidden], [Hidden]);
        }

        t["scorer.0.weight"] = (Ones(Hidden), [Hidden]);
        t["scorer.0.bias"] = (new float[Hidden], [Hidden]);
        t["scorer.1.weight"] = (Matrix(Hidden, Hidden, rng), [Hidden, Hidden]);
        t["scorer.1.bias"] = (new float[Hidden], [Hidden]);
        t["scorer.3.weight"] = (Matrix(1, Hidden, rng), [1, Hidden]);
        t["scorer.3.bias"] = (new float[1], [1]);

        t["act_head.0.weight"] = (Matrix(256, Hidden + LayaHeadOutput.FeatureCount, rng), [256, Hidden + LayaHeadOutput.FeatureCount]);
        t["act_head.0.bias"] = (new float[256], [256]);
        t["act_head.2.weight"] = (Matrix(ActClasses, 256, rng), [ActClasses, 256]);
        t["act_head.2.bias"] = (new float[ActClasses], [ActClasses]);

        return t;
    }

    internal static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBf16(Dictionary<string, (float[] Data, int[] Shape)> src)
    {
        var result = new Dictionary<string, (BFloat16[] Data, int[] Shape)>(src.Count);
        foreach (var kv in src)
        {
            var data = new BFloat16[kv.Value.Data.Length];
            for (int i = 0; i < data.Length; i++) data[i] = BFloat16.CreateChecked(kv.Value.Data[i]);
            result[kv.Key] = (data, kv.Value.Shape);
        }
        return result;
    }

    internal static float[] HiddenState(int seed = 2)
    {
        var rng = new Random(seed);
        var data = new float[SeqLen * Hidden];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.5f;
        return data;
    }

    internal static IReadOnlyList<int> Markers() => new[] { 2, 4 };

    internal static (float[] ScorerGrad, float[] ActGrad) Grads(int seed = 3)
    {
        var rng = new Random(seed);
        var sg = new float[Markers().Count];
        var ag = new float[ActClasses];
        for (int i = 0; i < sg.Length; i++) sg[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        for (int i = 0; i < ag.Length; i++) ag[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return (sg, ag);
    }

    static float[] Matrix(int rows, int cols, Random rng)
    {
        float bound = MathF.Sqrt(6f / (rows + cols));
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1) * bound;
        return data;
    }

    static float[] Ones(int n)
    {
        var data = new float[n];
        Array.Fill(data, 1f);
        return data;
    }
}
