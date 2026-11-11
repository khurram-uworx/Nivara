using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Shared deterministic fixture for the BF16 trainer gates. Every gate runs on the same tiny
/// shape so a divergence is attributable to the narrow lane and not to a shape difference.
///
/// <para>
/// The matrix weights are seeded Xavier-uniform, not the <c>0..N</c> ramp the first version of
/// these tests used. BF16 carries an 8-bit mantissa, so its spacing at 1024 is 8: a ramp that
/// reaches 2047 is destroyed by the conversion and the F32 and BF16 trainers then start from
/// materially different models. Seeding the generator keeps each gate reproducible while
/// keeping the weights inside BF16's well-resolved range.
/// </para>
/// </summary>
internal static class SilkBertBf16Fixtures
{
    internal const int Layers = 2;
    internal const int Hidden = 32;
    internal const int Heads = 4;
    internal const int Intermediate = 64;
    internal const int Vocab = 64;
    internal const int MaxPos = 16;
    internal const int SeqLen = 8;
    internal const int NumClasses = 2;

    internal static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk BF16 verification."); throw; }
    }

    internal static BertConfig Config() => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = Heads,
        NumHiddenLayers = Layers,
        IntermediateSize = Intermediate,
        VocabSize = Vocab,
        MaxPositionEmbeddings = MaxPos,
        LayerNormEps = 1e-4f,
    };

    /// <summary>
    /// A complete BERT-family checkpoint in the key space <see cref="BertGpuKeys"/> resolves for
    /// <paramref name="naming"/>, including the sequence-classification head the trainer requires.
    /// </summary>
    internal static Dictionary<string, (float[] Data, int[] Shape)> Checkpoint(
        BertGpuNaming naming, int seed = 1)
    {
        var rng = new Random(seed);
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();

        string emb = BertGpuKeys.Embeddings(naming);
        tensors[$"{emb}.word_embeddings.weight"] = (Matrix(Vocab, Hidden, rng), [Vocab, Hidden]);
        tensors[$"{emb}.position_embeddings.weight"] = (Matrix(MaxPos, Hidden, rng), [MaxPos, Hidden]);
        tensors[$"{emb}.LayerNorm.weight"] = (Ones(Hidden), [Hidden]);
        tensors[$"{emb}.LayerNorm.bias"] = (new float[Hidden], [Hidden]);

        for (int i = 0; i < Layers; i++)
        {
            foreach (char role in "qkvo")
            {
                string attention = BertGpuKeys.Attention(naming, i, role);
                tensors[$"{attention}.weight"] = (Matrix(Hidden, Hidden, rng), [Hidden, Hidden]);
                tensors[$"{attention}.bias"] = (new float[Hidden], [Hidden]);
            }

            string ffn1 = BertGpuKeys.Ffn(naming, i, 1);
            string ffn2 = BertGpuKeys.Ffn(naming, i, 2);
            tensors[$"{ffn1}.weight"] = (Matrix(Intermediate, Hidden, rng), [Intermediate, Hidden]);
            tensors[$"{ffn1}.bias"] = (new float[Intermediate], [Intermediate]);
            tensors[$"{ffn2}.weight"] = (Matrix(Hidden, Intermediate, rng), [Hidden, Intermediate]);
            tensors[$"{ffn2}.bias"] = (new float[Hidden], [Hidden]);

            foreach (int which in new[] { 1, 2 })
            {
                string ln = BertGpuKeys.LayerNorm(naming, i, which);
                tensors[$"{ln}.weight"] = (Ones(Hidden), [Hidden]);
                tensors[$"{ln}.bias"] = (new float[Hidden], [Hidden]);
            }
        }

        tensors["pre_classifier.weight"] = (Matrix(Hidden, Hidden, rng), [Hidden, Hidden]);
        tensors["pre_classifier.bias"] = (new float[Hidden], [Hidden]);
        tensors["classifier.weight"] = (Matrix(NumClasses, Hidden, rng), [NumClasses, Hidden]);
        tensors["classifier.bias"] = (new float[NumClasses], [NumClasses]);
        return tensors;
    }

    internal static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBf16(
        Dictionary<string, (float[] Data, int[] Shape)> src)
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

    /// <summary>Random tokens with a full attention mask; label count comes from the seed.</summary>
    internal static (int[] Tokens, float[] Mask, int[] Labels) Batch(int batch, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[batch * SeqLen];
        var mask = new float[batch * SeqLen];
        var labels = new int[batch];
        for (int i = 0; i < tokens.Length; i++) { tokens[i] = rng.Next(Vocab); mask[i] = 1f; }
        for (int i = 0; i < batch; i++) labels[i] = rng.Next(NumClasses);
        return (tokens, mask, labels);
    }

    /// <summary>
    /// A learnable smoke batch: the first token is a deterministic class marker and the rest is
    /// noise, so a decreasing loss measures the optimizer rather than memorised label noise.
    /// </summary>
    internal static (int[] Tokens, float[] Mask, int[] Labels) TrainingBatch(int samples, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[samples * SeqLen];
        var mask = new float[samples * SeqLen];
        var labels = new int[samples];
        for (int i = 0; i < samples; i++)
        {
            labels[i] = i % NumClasses;
            tokens[i * SeqLen] = labels[i] == 0 ? 2 : 5;
            for (int s = 1; s < SeqLen; s++) tokens[i * SeqLen + s] = rng.Next(1, Vocab);
            for (int s = 0; s < SeqLen; s++) mask[i * SeqLen + s] = 1f;
        }
        return (tokens, mask, labels);
    }

    internal static void AssertNarrow(float[] actual, float[] expected, string context)
    {
        Assert.That(actual.Length, Is.EqualTo(expected.Length), context);
        float maxAbs = 0f;
        float scale = 1f;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(expected[i] - actual[i]));
            scale = MathF.Max(scale, 1f + MathF.Abs(expected[i]));
        }
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-2f * scale), $"{context}: maxAbs {maxAbs} > 1e-2*{scale}");
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
