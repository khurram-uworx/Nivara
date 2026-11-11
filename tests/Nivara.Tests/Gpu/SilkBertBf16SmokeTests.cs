using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkBertBf16SmokeTests
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
        catch { Assert.Ignore("No OpenCL GPU available; skipping."); throw; }
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

    static float[] Filled(int rows, int cols, float start)
    {
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = start + i;
        return data;
    }

    [Test]
    public void Train_TrueBf16_LossDecreases_AndExportF32()
    {
        var runtime = Device();
        using (runtime)
        {
            var config = TinyConfig();
            var floatTensors = BuildBert();
            var bf16Tensors = ToBf16(floatTensors);

            using var trainer = SilkDistilBertTrainer.CreateTrueBf16(runtime, config, bf16Tensors, BertGpuNaming.Bert,
                maxBatch: 4, maxSeqLen: 16);

            const int samples = 64;
            var rng = new Random(37);
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

            var epochs = trainer.Train(tokens, mask, labels, batchSize: 4, seqLen: SeqLen, epochs: 4,
                learningRate: 0.1f, weightDecay: 0.01f, seed: 41);

            Assert.That(epochs.Count, Is.EqualTo(4));
            foreach (var e in epochs) Assert.That(float.IsFinite(e.Loss), Is.True);
            Assert.That(epochs[^1].Loss, Is.LessThan(0.75f * epochs[0].Loss));

            var exported = trainer.ExportCheckpoint();
            Assert.That(exported.Count, Is.GreaterThan(0));
        }
    }
}
