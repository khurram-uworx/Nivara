using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.Samples;
using Nivara.Silk;
using System.Numerics;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkDistilBertTrainerBf16SmokeTests
{
    const int Layers = 2;
    const int Hidden = 32;
    const int Heads = 4;
    const int Intermediate = 64;
    const int Vocab = 64;
    const int MaxPos = 16;
    const int SeqLen = 8;

    static SilkRuntime Device()
    {
        try { return new SilkRuntime(); }
        catch { Assert.Ignore("No OpenCL GPU available; skipping."); throw; }
    }

    static DistilBertConfig TinyDistilConfig() => new()
    {
        Dim = Hidden,
        NLayers = Layers,
        NHeads = Heads,
        HiddenDim = Intermediate,
        MaxPosition = MaxPos,
        Eps = 1e-4f,
        VocabSize = Vocab,
    };

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

    [Test]
    public void Train_Bf16Checkpoint_LossDecreases()
    {
        var runtime = Device();
        using (runtime)
        {
            var distilConfig = TinyDistilConfig();
            var config = distilConfig.ToBertConfig();
            var model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
            var floatTensors = model.ToSilkTensors();
            var bf16Tensors = ToBf16(floatTensors);

            using var trainer = new SilkDistilBertTrainer(runtime, config, bf16Tensors, BertGpuNaming.DistilBert,
                maxBatch: 4, maxSeqLen: 16);

            const int samples = 64;
            var rng = new Random(37);
            var tokens = new int[samples * SeqLen];
            var mask = new float[samples * SeqLen];
            var labels = new int[samples];
            for (int i = 0; i < samples; i++)
            {
                labels[i] = i % 2;
                tokens[i * SeqLen] = labels[i] == 0 ? 2 : 5;
                for (int s = 1; s < SeqLen; s++) tokens[i * SeqLen + s] = rng.Next(1, Vocab);
                for (int s = 0; s < SeqLen; s++) mask[i * SeqLen + s] = 1f;
            }

            var epochs = trainer.Train(tokens, mask, labels, batchSize: 4, seqLen: SeqLen, epochs: 4,
                learningRate: 0.1f, weightDecay: 0.01f, seed: 41);

            Assert.That(epochs.Count, Is.EqualTo(4));
            foreach (var e in epochs) Assert.That(float.IsFinite(e.Loss), Is.True);
            Assert.That(epochs[^1].Loss, Is.LessThan(0.75f * epochs[0].Loss));
        }
    }
}
