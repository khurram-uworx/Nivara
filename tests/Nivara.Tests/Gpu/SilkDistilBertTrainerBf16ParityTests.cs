using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity gate for the true BF16 DistilBERT trainer (#1135): forward logits and parameter
/// gradients must agree with the F32 trainer on the tiny DistilBERT shape within the documented
/// narrow bound (maxAbs <= 1e-2 * (1+|reference|)).
/// </summary>
[Category("Gpu")]
public sealed class SilkDistilBertTrainerBf16ParityTests
{
    const int Batch = 2;

    [Test]
    public void Forward_TrueBf16_MatchesF32Trainer()
    {
        var config = SilkBertBf16Fixtures.Config();
        var floatTensors = SilkBertBf16Fixtures.Checkpoint(BertGpuNaming.DistilBert);
        var bf16Tensors = SilkBertBf16Fixtures.ToBf16(floatTensors);

        var (tokens, mask, _) = SilkBertBf16Fixtures.Batch(Batch, 7);

        using var rtF = SilkBertBf16Fixtures.Device();
        using var rtB = SilkBertBf16Fixtures.Device();
        using var f32Trainer = new SilkDistilBertTrainer(rtF, config, floatTensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);
        using var bf16Trainer = SilkDistilBertTrainer.CreateTrueBf16(rtB, config, bf16Tensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);

        var fLogits = f32Trainer.Forward(tokens, mask, Batch, SilkBertBf16Fixtures.SeqLen);
        var bLogits = bf16Trainer.Forward(tokens, mask, Batch, SilkBertBf16Fixtures.SeqLen);

        SilkBertBf16Fixtures.AssertNarrow(bLogits, fLogits, "logits");
    }

    [Test]
    public void Gradients_TrueBf16_MatchF32Trainer()
    {
        var config = SilkBertBf16Fixtures.Config();
        var floatTensors = SilkBertBf16Fixtures.Checkpoint(BertGpuNaming.DistilBert);
        var bf16Tensors = SilkBertBf16Fixtures.ToBf16(floatTensors);

        var (tokens, mask, labels) = SilkBertBf16Fixtures.Batch(Batch, 11);

        using var rtF = SilkBertBf16Fixtures.Device();
        using var rtB = SilkBertBf16Fixtures.Device();
        using var f32Trainer = new SilkDistilBertTrainer(rtF, config, floatTensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);
        using var bf16Trainer = SilkDistilBertTrainer.CreateTrueBf16(rtB, config, bf16Tensors, BertGpuNaming.DistilBert, maxBatch: 4, maxSeqLen: 16);

        var fLogits = f32Trainer.Forward(tokens, mask, Batch, SilkBertBf16Fixtures.SeqLen);
        var fGrad = new float[Batch * SilkBertBf16Fixtures.NumClasses];
        SilkLoss.InputGradient(fLogits, labels, Batch, SilkBertBf16Fixtures.NumClasses, 1f / Batch, fGrad);
        f32Trainer.Backward(Batch, SilkBertBf16Fixtures.SeqLen, fGrad);
        var fGrads = f32Trainer.ExportGradients();

        var bLogits = bf16Trainer.Forward(tokens, mask, Batch, SilkBertBf16Fixtures.SeqLen);
        var bGrad = new float[Batch * SilkBertBf16Fixtures.NumClasses];
        SilkLoss.InputGradient(bLogits, labels, Batch, SilkBertBf16Fixtures.NumClasses, 1f / Batch, bGrad);
        bf16Trainer.Backward(Batch, SilkBertBf16Fixtures.SeqLen, bGrad);
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
