using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkLayaHeadTrainerSmokeTests
{
    [Test]
    public void ForwardBackwardExport_Runs()
    {
        using var rt = SilkLayaHeadBf16Fixtures.Device();
        var tensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        using var trainer = new SilkLayaHeadTrainer(rt, tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, maxSeqLen: SilkLayaHeadBf16Fixtures.SeqLen + 2);

        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();
        var (sg, ag) = SilkLayaHeadBf16Fixtures.Grads();

        var output = trainer.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
        Assert.That(output.Logits.Length, Is.EqualTo(markers.Count));

        trainer.Backward(sg, ag);
        trainer.AdamUpdateAll(1e-3f, 0f);

        var exported = trainer.ExportCheckpoint();
        Assert.That(exported.ContainsKey("type_emb.weight"));
        LayaDecisionHead<float, float>.LoadWeights(exported, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);
    }
}
