using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Parity gate for <see cref="SilkLayaHeadTrainer"/> (F32): forward logits must agree with the
/// CPU AutoDiff <see cref="LayaDecisionHead{TModel,TWeight}"/> on the seeded fixture, and the
/// exported checkpoint must round-trip through <see cref="LayaDecisionHead{TModel,TWeight}.LoadWeights"/>.
/// </summary>
[Category("Gpu")]
public sealed class SilkLayaHeadTrainerTests
{
    [Test]
    public void Forward_LogitsMatchCpuHead()
    {
        using var rt = SilkLayaHeadBf16Fixtures.Device();
        var tensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        using var trainer = new SilkLayaHeadTrainer(rt, tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, maxSeqLen: SilkLayaHeadBf16Fixtures.SeqLen + 2);

        var cpuHead = LayaDecisionHead<float, float>.LoadWeights(tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);

        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();

        var deviceOut = trainer.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);

        var hiddenTensor = ReverseGradTensor<float>.FromMatrix((float[])hidden.Clone(), SilkLayaHeadBf16Fixtures.SeqLen, SilkLayaHeadBf16Fixtures.Hidden);
        var cpuOut = cpuHead.Forward(hiddenTensor, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);

        Assert.That(deviceOut.Logits.Length, Is.EqualTo(cpuOut.Logits.Length));
        float worst = 0f;
        for (int i = 0; i < deviceOut.Logits.Length; i++)
            worst = MathF.Max(worst, MathF.Abs(deviceOut.Logits[i] - cpuOut.Logits[i]));
        Assert.That(worst, Is.LessThan(1e-4f), $"worst logit diff {worst}");
    }

    [Test]
    public void BackwardAndUpdate_RunAndCheckpointReloads()
    {
        using var rt = SilkLayaHeadBf16Fixtures.Device();
        var tensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        using var trainer = new SilkLayaHeadTrainer(rt, tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, maxSeqLen: SilkLayaHeadBf16Fixtures.SeqLen + 2);

        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();
        var (sg, ag) = SilkLayaHeadBf16Fixtures.Grads();

        trainer.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
        trainer.Backward(sg, ag);
        trainer.AdamUpdateAll(1e-3f, 0f);

        var exported = trainer.ExportCheckpoint();
        Assert.That(exported.ContainsKey("type_emb.weight"));
        LayaDecisionHead<float, float>.LoadWeights(exported, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);
    }
}
