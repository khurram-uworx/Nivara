using Nivara.AI;
using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

[Category("Gpu")]
public sealed class SilkLayaHeadTrainerBf16SmokeTests
{
    [Test]
    public void TrueBf16_ForwardBackwardExport_Runs()
    {
        using var rt = SilkLayaHeadBf16Fixtures.Device();
        var floatTensors = SilkLayaHeadBf16Fixtures.Checkpoint();
        var bf16Tensors = SilkLayaHeadBf16Fixtures.ToBf16(floatTensors);
        using var trainer = SilkLayaHeadTrainer.CreateTrueBf16(rt, bf16Tensors, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses, SilkLayaHeadBf16Fixtures.SeqLen + 2);

        var hidden = SilkLayaHeadBf16Fixtures.HiddenState();
        var markers = SilkLayaHeadBf16Fixtures.Markers();
        var (sg, ag) = SilkLayaHeadBf16Fixtures.Grads();

        var output = trainer.Forward(hidden, LayaQuestionType.Choice, markers, SilkLayaHeadBf16Fixtures.ValidLength);
        Assert.That(output.Logits.Length, Is.EqualTo(markers.Count));

        trainer.Backward(sg, ag);
        trainer.AdamUpdateAll(1e-3f, 0f);

        var exported = trainer.ExportCheckpoint();
        LayaDecisionHead<float, float>.LoadWeights(exported, SilkLayaHeadBf16Fixtures.HeadLayers, SilkLayaHeadBf16Fixtures.ActClasses);
    }
}
