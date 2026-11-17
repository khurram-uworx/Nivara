using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity gate for the true BF16 Laya head trainer: forward logits must agree with the F32 trainer
/// on the tiny seeded fixture within the documented narrow bound.
/// </summary>
[Category("Gpu")]
public sealed class SilkLayaHeadTrainerBf16ParityTests
{
    [Test]
    public void Forward_TrueBf16_MatchesF32Trainer()
    {
        var floatTensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        var bf16Tensors = SilkLayaHeadBf16Fixtures.ToBf16(floatTensors);

        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();

        using var rtF = SilkLayaHeadBf16Fixtures.Device();
        using var rtB = SilkLayaHeadBf16Fixtures.Device();
        using var f32 = new SilkLayaHeadTrainer(rtF, floatTensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, maxSeqLen: SilkLayaHeadBf16Fixtures.SeqLen + 2);
        using var bf16 = SilkLayaHeadTrainer.CreateTrueBf16(rtB, bf16Tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, SilkLayaHeadBf16Fixtures.SeqLen + 2);

        var fOut = f32.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
        var bOut = bf16.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);

        Assert.That(bOut.Logits.Length, Is.EqualTo(fOut.Logits.Length));
        for (int i = 0; i < fOut.Logits.Length; i++)
        {
            float scale = 1f + MathF.Abs(fOut.Logits[i]);
            Assert.That(MathF.Abs(bOut.Logits[i] - fOut.Logits[i]), Is.LessThanOrEqualTo(1e-2f * scale));
        }
    }

    [Test]
    public void Checkpoint_TrueBf16_Reloads()
    {
        var floatTensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        var bf16Tensors = SilkLayaHeadBf16Fixtures.ToBf16(floatTensors);

        using var rt = SilkLayaHeadBf16Fixtures.Device();
        using var trainer = SilkLayaHeadTrainer.CreateTrueBf16(rt, bf16Tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, SilkLayaHeadBf16Fixtures.SeqLen + 2);

        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();
        var (sg, ag) = SilkLayaHeadBf16Fixtures.Grads();

        trainer.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
        trainer.Backward(sg, ag);
        trainer.AdamUpdateAll(1e-3f, 0f);

        var exported = trainer.ExportCheckpoint();
        LayaDecisionHead<float, float>.LoadWeights(exported, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);
    }
}
