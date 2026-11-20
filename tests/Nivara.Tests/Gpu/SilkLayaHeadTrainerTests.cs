using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;
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

    [Test]
    public void TrainingStep_CpuWeightsMatchSilk()
    {
        using var rt = SilkLayaHeadBf16Fixtures.Device();
        var tensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();
        var (sg, ag) = SilkLayaHeadBf16Fixtures.Grads();

        // Silk path
        using var trainer = new SilkLayaHeadTrainer(rt, tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, maxSeqLen: SilkLayaHeadBf16Fixtures.SeqLen + 2);
        trainer.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
        trainer.Backward(sg, ag);
        trainer.AdamUpdateAll(1e-3f, 0f);
        var silkExported = trainer.ExportCheckpoint();

        // CPU path
        var cpuHead = LayaDecisionHead<float, float>.LoadWeights(tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);
        using var optimizer = new Adam<float>(1e-3f);
        optimizer.AddParameterGroup(cpuHead.GetParameters().Values);
        optimizer.ZeroGrad();
        using (GradientUtils.Grad())
        {
            var hiddenT = ReverseGradTensor<float>.FromMatrix((float[])hidden.Clone(), SilkLayaHeadBf16Fixtures.SeqLen, SilkLayaHeadBf16Fixtures.Hidden);
            var choiceLogits = cpuHead.ForwardChoiceLogits(hiddenT, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
            var actLogits = cpuHead.ForwardActLogits(hiddenT, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
            int k = markers.Count;
            var gradChoice = ReverseGradTensor<float>.FromMatrix((float[])sg.Clone(), 1, k, requiresGrad: false);
            var gradAct = ReverseGradTensor<float>.FromMatrix((float[])ag.Clone(), 1, 2, requiresGrad: false);
            var total = ReverseGradOperations.Add(
                ReverseGradOperations.Sum(ReverseGradOperations.Multiply(choiceLogits, gradChoice)),
                ReverseGradOperations.Sum(ReverseGradOperations.Multiply(actLogits, gradAct)));
            total.Backward();
        }
        optimizer.Step();

        var silkHead = LayaDecisionHead<float, float>.LoadWeights(silkExported, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);

        float worst = 0f;
        string worstName = "";
        foreach (var (name, cp) in cpuHead.GetParameters())
        {
            if (!silkHead.GetParameters().TryGetValue(name, out var sp)) continue;
            if (cp.Tensor.Data.TryGetSpan(out var cspan) && sp.Tensor.Data.TryGetSpan(out var sspan))
            {
                for (int i = 0; i < cspan.Length && i < sspan.Length; i++)
                {
                    var d = MathF.Abs(cspan[i] - sspan[i]);
                    if (d > worst) { worst = d; worstName = name; }
                }
            }
        }
        Assert.That(worst, Is.LessThan(1e-2f), $"worst param diff {worst} at {worstName}");
    }
}