using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity gate for true BF16 BERT/MiniLM training through <see cref="SilkDistilBertTrainer"/>
/// with <see cref="BertGpuNaming.Bert"/>: forward logits and parameter gradients must agree
/// with the F32 trainer within the documented narrow bound.
/// </summary>
[Category("Gpu")]
public sealed class SilkBertBf16ParityTests
{
    const int Layers = 2;
    const int Hidden = 32;
    const int Heads = 4;
    const int Intermediate = 64;
    const int Vocab = 64;
    const int MaxPos = 16;
    const int SeqLen = 8;
    const int NumClasses = 2;

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk BF16 parity verification."); throw; }
    }

    static BertConfig TinyConfig() => new()
    {
        HiddenSize = Hidden,
        NumAttentionHeads = Heads,
        NumHiddenLayers = Layers,
        IntermediateSize = Intermediate,
        VocabSize = Vocab,
        MaxPositionEmbeddings = MaxPos,
        LayerNormEps = 1e-4f,
    };

    static Dictionary<string, (float[] Data, int[] Shape)> BuildBert()
    {
        var tensors = new Dictionary<string, (float[] Data, int[] Shape)>();
        tensors["embeddings.word_embeddings.weight"] = (Filled(Vocab, Hidden, 0f), new[] { Vocab, Hidden });
        tensors["embeddings.position_embeddings.weight"] = (Filled(MaxPos, Hidden, 0f), new[] { MaxPos, Hidden });
        tensors["embeddings.LayerNorm.weight"] = (Filled(Hidden, 1, 0f), new[] { Hidden });
        tensors["embeddings.LayerNorm.bias"] = (Filled(Hidden, 1, 0f), new[] { Hidden });
        for (int i = 0; i < Layers; i++)
        {
            float s = i;
            string layer = $"encoder.layer.{i}";
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
        tensors["pre_classifier.weight"] = (Filled(Hidden, Hidden, 0f), new[] { Hidden, Hidden });
        tensors["pre_classifier.bias"] = (Filled(Hidden, 1, 0f), new[] { Hidden });
        tensors["classifier.weight"] = (Filled(NumClasses, Hidden, 0f), new[] { NumClasses, Hidden });
        tensors["classifier.bias"] = (Filled(NumClasses, 1, 0f), new[] { NumClasses });
        return tensors;
    }

    static Dictionary<string, (BFloat16[] Data, int[] Shape)> ToBf16(
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

    static (int[] tokens, float[] mask, int[] labels) MakeBatch(int batch, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[batch * SeqLen];
        var mask = new float[batch * SeqLen];
        var labels = new int[batch];
        for (int i = 0; i < tokens.Length; i++) { tokens[i] = rng.Next(Vocab); mask[i] = 1f; }
        for (int i = 0; i < batch; i++) labels[i] = rng.Next(NumClasses);
        return (tokens, mask, labels);
    }

    static void AssertNarrow(float[] actual, float[] expected, string context)
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

    static float[] Filled(int rows, int cols, float start)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = start + i;
        return data;
    }

    [Test]
    public void Forward_TrueBf16_MatchesF32Trainer()
    {
        var config = TinyConfig();
        var floatTensors = BuildBert();
        var bf16Tensors = ToBf16(floatTensors);

        var (tokens, mask, _) = MakeBatch(2, 7);

        using var rtF = Device();
        using var rtB = Device();
        using var f32Trainer = new SilkDistilBertTrainer(rtF, config, floatTensors, BertGpuNaming.Bert, maxBatch: 4, maxSeqLen: 16);
        using var bf16Trainer = SilkDistilBertTrainer.CreateTrueBf16(rtB, config, bf16Tensors, BertGpuNaming.Bert, maxBatch: 4, maxSeqLen: 16);

        var fLogits = f32Trainer.Forward(tokens, mask, 2, SeqLen);
        var bLogits = bf16Trainer.Forward(tokens, mask, 2, SeqLen);

        AssertNarrow(bLogits, fLogits, "logits");
    }

    [Test]
    public void Gradients_TrueBf16_MatchF32Trainer()
    {
        var config = TinyConfig();
        var floatTensors = BuildBert();
        var bf16Tensors = ToBf16(floatTensors);

        var (tokens, mask, labels) = MakeBatch(2, 11);

        using var rtF = Device();
        using var rtB = Device();
        using var f32Trainer = new SilkDistilBertTrainer(rtF, config, floatTensors, BertGpuNaming.Bert, maxBatch: 4, maxSeqLen: 16);
        using var bf16Trainer = SilkDistilBertTrainer.CreateTrueBf16(rtB, config, bf16Tensors, BertGpuNaming.Bert, maxBatch: 4, maxSeqLen: 16);

        var fLogits = f32Trainer.Forward(tokens, mask, 2, SeqLen);
        var fGrad = new float[2 * NumClasses];
        SilkLoss.InputGradient(fLogits, labels, 2, NumClasses, 1f / 2, fGrad);
        f32Trainer.Backward(2, SeqLen, fGrad);
        var fGrads = f32Trainer.ExportGradients();

        var bLogits = bf16Trainer.Forward(tokens, mask, 2, SeqLen);
        var bGrad = new float[2 * NumClasses];
        SilkLoss.InputGradient(bLogits, labels, 2, NumClasses, 1f / 2, bGrad);
        bf16Trainer.Backward(2, SeqLen, bGrad);
        var bGrads = bf16Trainer.ExportGradients();

        Assert.That(bGrads.Keys.Count, Is.EqualTo(fGrads.Keys.Count));
        foreach (var key in fGrads.Keys)
        {
            Assert.That(bGrads.ContainsKey(key), Is.True, $"missing device grad for {key}");
            var f = fGrads[key].Data;
            var b = bGrads[key].Data;
            Assert.That(b.Length, Is.EqualTo(f.Length), key);
            float localWorst = 0f;
            float scale = 1f;
            for (int i = 0; i < f.Length; i++)
            {
                localWorst = MathF.Max(localWorst, MathF.Abs(b[i] - f[i]));
                scale = MathF.Max(scale, 1f + MathF.Abs(f[i]));
            }
            Assert.That(localWorst, Is.LessThanOrEqualTo(1e-2f * scale), $"{key}: {localWorst}");
        }
    }
}
