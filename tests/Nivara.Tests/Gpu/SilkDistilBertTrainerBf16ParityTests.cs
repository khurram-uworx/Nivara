using Nivara.AI;
using Nivara.Samples;
using Nivara.Silk;
using NUnit.Framework;
using System.Numerics;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity gate for the true BF16 DistilBERT trainer (#1135): forward logits and parameter
/// gradients must agree with the F32 trainer on the tiny DistilBERT shape within the documented
/// narrow bound (maxAbs <= 1e-2 * (1+|reference|)).
/// </summary>
[Category("Gpu")]
public sealed class SilkDistilBertTrainerBf16ParityTests
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
        catch { Assert.Ignore("No OpenCL GPU available; skipping Silk BF16 parity verification."); throw; }
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

    static (int[] tokens, float[] mask, int[] labels) MakeBatch(int batch, int seed)
    {
        var rng = new Random(seed);
        var tokens = new int[batch * SeqLen];
        var mask = new float[batch * SeqLen];
        var labels = new int[batch];
        for (int i = 0; i < tokens.Length; i++) { tokens[i] = rng.Next(Vocab); mask[i] = 1f; }
        for (int i = 0; i < batch; i++) labels[i] = rng.Next(2);
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

    [Test]
    public void Forward_TrueBf16_MatchesF32Trainer()
    {
        var distilConfig = TinyDistilConfig();
        var config = distilConfig.ToBertConfig();
        var model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        var floatTensors = model.ToSilkTensors();
        var bf16Tensors = ToBf16(floatTensors);

        var (tokens, mask, _) = MakeBatch(2, 7);

        using var rtF = Device();
        using var rtB = Device();
        using var f32Trainer = new SilkDistilBertTrainer(rtF, config, floatTensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);
        using var bf16Trainer = SilkDistilBertTrainer.CreateTrueBf16(rtB, config, bf16Tensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);

        var fLogits = f32Trainer.Forward(tokens, mask, 2, SeqLen);
        var bLogits = bf16Trainer.Forward(tokens, mask, 2, SeqLen);

        AssertNarrow(bLogits, fLogits, "logits");
    }

    [Test]
    public void Gradients_TrueBf16_MatchF32Trainer()
    {
        var distilConfig = TinyDistilConfig();
        var config = distilConfig.ToBertConfig();
        var model = new DistilBertForSequenceClassification<float>(config, numClasses: 2);
        var floatTensors = model.ToSilkTensors();
        var bf16Tensors = ToBf16(floatTensors);

        var (tokens, mask, labels) = MakeBatch(2, 11);

        using var rtF = Device();
        using var rtB = Device();
        using var f32Trainer = new SilkDistilBertTrainer(rtF, config, floatTensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);
        using var bf16Trainer = SilkDistilBertTrainer.CreateTrueBf16(rtB, config, bf16Tensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);

        // F32 path.
        var fLogits = f32Trainer.Forward(tokens, mask, 2, SeqLen);
        var fGrad = new float[2 * 2];
        SilkLoss.InputGradient(fLogits, labels, 2, 2, 1f / 2, fGrad);
        f32Trainer.Backward(2, SeqLen, fGrad);
        var fGrads = f32Trainer.ExportGradients();

        // True BF16 path.
        var bLogits = bf16Trainer.Forward(tokens, mask, 2, SeqLen);
        var bGrad = new float[2 * 2];
        SilkLoss.InputGradient(bLogits, labels, 2, 2, 1f / 2, bGrad);
        bf16Trainer.Backward(2, SeqLen, bGrad);
        var bGrads = bf16Trainer.ExportGradients();

        Assert.That(bGrads.Keys.Count, Is.EqualTo(fGrads.Keys.Count));
        float worst = 0f;
        string worstKey = "";
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
            if (localWorst > worst) { worst = localWorst; worstKey = key; }
            Assert.That(localWorst, Is.LessThanOrEqualTo(1e-2f * scale), $"{key}: {localWorst}");
        }
        TestContext.Out.WriteLine($"worst gradient diff {worst} at {worstKey}");
    }
}
