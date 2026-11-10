using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Diagnostics;
using System.Numerics;

namespace Nivara.Tests.Gpu;

[TestFixture]
[Category("Gpu")]
[Category("Performance")]
public class SilkLayaNarrowPerfTests
{
    const int Hidden = 32;
    const int Heads = 4;
    const int Intermediate = 4 * Hidden; // 128 (head)
    const int EncoderIntermediate = 8;
    const int Layers = 2;
    const int Vocab = 64;
    const int HeadLayers = 2;
    const int TokenCount = 64;

    static ModernBertConfig TinyConfig => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = Heads,
        NumHiddenLayers = Layers,
        IntermediateSize = EncoderIntermediate,
        VocabSize = Vocab,
        LocalAttention = 8,
        GlobalAttnEveryNLayers = 2,
        MaxPositionEmbeddings = 64,
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

    static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBFloat16(
        Dictionary<string, (float[] Data, int[] Shape)> src)
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

    [Test]
    public void Forward_NarrowVsF32_InterleavedTiming()
    {
        try { using var probe = new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk perf verification."); }

        TimingGuards.RequireOptimizedBuildForTiming();

        var config = TinyConfig;
        var tensors = Combine(BuildEncoderStateDict("encoder"), BuildHeadTensors());
        var bf16Tensors = ToBFloat16(tensors);

        int[] ids = new int[TokenCount];
        for (int i = 0; i < ids.Length; i++) ids[i] = i % Vocab;
        int valid = TokenCount;
        int[] markers = [0];

        using var f32Session = SilkLayaSession.Open(config, tensors, HeadLayers);
        using var narrowSession = SilkLayaSession.Open(config, bf16Tensors, HeadLayers);

        f32Session.Forward(ids, valid, LayaQuestionType.Choice, markers);
        narrowSession.Forward(ids, valid, LayaQuestionType.Choice, markers);

        const int Warmup = 3;
        const int Iterations = 7;
        var f32Ms = new double[Iterations];
        var narrowMs = new double[Iterations];

        for (int w = 0; w < Warmup; w++)
        {
            f32Session.Forward(ids, valid, LayaQuestionType.Choice, markers);
            narrowSession.Forward(ids, valid, LayaQuestionType.Choice, markers);
        }

        var sw = new Stopwatch();
        for (int i = 0; i < Iterations; i++)
        {
            if ((i & 1) == 0)
            {
                sw.Restart();
                f32Session.Forward(ids, valid, LayaQuestionType.Choice, markers);
                sw.Stop();
                f32Ms[i] = sw.Elapsed.TotalMilliseconds;

                sw.Restart();
                narrowSession.Forward(ids, valid, LayaQuestionType.Choice, markers);
                sw.Stop();
                narrowMs[i] = sw.Elapsed.TotalMilliseconds;
            }
            else
            {
                sw.Restart();
                narrowSession.Forward(ids, valid, LayaQuestionType.Choice, markers);
                sw.Stop();
                narrowMs[i] = sw.Elapsed.TotalMilliseconds;

                sw.Restart();
                f32Session.Forward(ids, valid, LayaQuestionType.Choice, markers);
                sw.Stop();
                f32Ms[i] = sw.Elapsed.TotalMilliseconds;
            }
        }

        static double Median(double[] values)
        {
            var sorted = (double[])values.Clone();
            Array.Sort(sorted);
            int middle = sorted.Length / 2;
            return sorted.Length % 2 == 0
                ? (sorted[middle - 1] + sorted[middle]) / 2.0
                : sorted[middle];
        }

        double f32Median = Median(f32Ms);
        double narrowMedian = Median(narrowMs);
        double ratio = f32Median > 0 ? narrowMedian / f32Median : double.NaN;

        TestContext.Out.WriteLine(
            $"Silk Laya narrow-vs-F32 perf gate: tokens={TokenCount}, hidden={Hidden}, layers={Layers}, headLayers={HeadLayers}");
        TestContext.Out.WriteLine($"  F32 median:    {f32Median:F3} ms");
        TestContext.Out.WriteLine($"  Narrow median: {narrowMedian:F3} ms");
        TestContext.Out.WriteLine($"  Narrow/F32 ratio: {ratio:F3} ({Iterations} interleaved rounds, median of each route)");
        TestContext.Out.WriteLine("  Phase 1 narrows matmul weights only; activations remain F32, so the expected win is small.");

        Assert.That(!double.IsNaN(ratio) && !double.IsInfinity(ratio));
        Assert.That(f32Median, Is.GreaterThan(0));
        Assert.That(narrowMedian, Is.GreaterThan(0));
    }
}
