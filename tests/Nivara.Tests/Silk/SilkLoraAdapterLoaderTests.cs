using Nivara.Silk;
using NUnit.Framework;
using System.Runtime.InteropServices;

namespace Nivara.Tests.Silk;

[TestFixture]
public sealed class SilkLoraAdapterLoaderTests
{
    static LoraSftAdapterBundle MakeBundle(int rank = 2, float alpha = 4f, string targets = "Q,V", int hidden = 8, int qDim = 8, int kvDim = 8, int intermediate = 16, int layers = 1)
    {
        var targetFlags = SilkLoraAdapterLoader.ParseTargets(targets);
        var slots = SilkLoraAdapterLoader.BuildExpectedSlots(targetFlags, layers, hidden, qDim, kvDim, intermediate);
        var entries = new LoraSftAdapterEntry[slots.Count];
        for (int i = 0; i < slots.Count; i++)
        {
            var s = slots[i];
            var a = new float[s.InFeatures * rank];
            var b = new float[rank * s.OutFeatures];
            for (int j = 0; j < a.Length; j++) a[j] = 0.01f * (j + 1);
            for (int j = 0; j < b.Length; j++) b[j] = 0.02f * (j + 1);
            entries[i] = new LoraSftAdapterEntry
            {
                Index = i,
                Layer = s.Layer,
                Target = s.Target.ToString(),
                AShape = [s.InFeatures, rank],
                BShape = [rank, s.OutFeatures],
                A = Convert.ToBase64String(MemoryMarshal.AsBytes(a.AsSpan())),
                B = Convert.ToBase64String(MemoryMarshal.AsBytes(b.AsSpan())),
            };
        }

        return new LoraSftAdapterBundle
        {
            BaseModelId = "test",
            Template = "qwen",
            Rank = rank,
            Alpha = alpha,
            Targets = targets,
            TrainableParams = entries.Sum(e => e.AShape[0] * e.AShape[1] + e.BShape[0] * e.BShape[1]),
            Adapters = entries,
        };
    }

    [Test]
    public void ValidateAndParse_ValidBundle_ReturnsAdapters()
    {
        var bundle = MakeBundle();
        var slots = SilkLoraAdapterLoader.BuildExpectedSlots(SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V, 1, 8, 8, 8, 16);
        var parsed = SilkLoraAdapterLoader.ValidateAndParse(bundle, slots);

        Assert.That(parsed, Has.Length.EqualTo(2));
        Assert.That(parsed[0].A, Has.Length.EqualTo(8 * 2));
        Assert.That(parsed[0].B, Has.Length.EqualTo(2 * 8));
    }

    [Test]
    public void ValidateAndParse_RankMismatch_Throws()
    {
        var bundle = MakeBundle(rank: 2);
        var wrongSlots = SilkLoraAdapterLoader.BuildExpectedSlots(SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V, 1, 8, 8, 8, 16);
        bundle.Rank = 4; // mismatch: data/shape expect rank 2

        var ex = Assert.Throws<InvalidOperationException>(() => SilkLoraAdapterLoader.ValidateAndParse(bundle, wrongSlots));
        Assert.That(ex!.Message, Does.Contain("shape mismatch").IgnoreCase.Or.Contain("data length mismatch").IgnoreCase.Or.Contain("rank").IgnoreCase);
    }

    [Test]
    public void ValidateAndParse_CountMismatch_Throws()
    {
        var bundle = MakeBundle();
        bundle.Adapters = bundle.Adapters.Take(1).ToArray();

        var slots = SilkLoraAdapterLoader.BuildExpectedSlots(SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V, 1, 8, 8, 8, 16);
        var ex = Assert.Throws<InvalidOperationException>(() => SilkLoraAdapterLoader.ValidateAndParse(bundle, slots));
        Assert.That(ex!.Message, Does.Contain("count mismatch").IgnoreCase);
    }

    [Test]
    public void ValidateAndParse_ShapeMismatch_Throws()
    {
        var bundle = MakeBundle();
        bundle.Adapters[0].AShape = [999, 999];

        var slots = SilkLoraAdapterLoader.BuildExpectedSlots(SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V, 1, 8, 8, 8, 16);
        var ex = Assert.Throws<InvalidOperationException>(() => SilkLoraAdapterLoader.ValidateAndParse(bundle, slots));
        Assert.That(ex!.Message, Does.Contain("shape mismatch").IgnoreCase);
    }

    [Test]
    public void ValidateAndParse_DataLengthMismatch_Throws()
    {
        var bundle = MakeBundle();
        bundle.Adapters[0].A = Convert.ToBase64String(new byte[4]); // too small
        bundle.Adapters[0].AShape = [8, 2];

        var slots = SilkLoraAdapterLoader.BuildExpectedSlots(SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V, 1, 8, 8, 8, 16);
        var ex = Assert.Throws<InvalidOperationException>(() => SilkLoraAdapterLoader.ValidateAndParse(bundle, slots));
        Assert.That(ex!.Message, Does.Contain("data length mismatch").IgnoreCase);
    }

    [Test]
    public void ParseTargets_SupportsCommonForms()
    {
        var targets = SilkLoraAdapterLoader.ParseTargets("Q, V, FFN");
        Assert.That(targets.HasFlag(SilkLlamaLoRATarget.Q), Is.True);
        Assert.That(targets.HasFlag(SilkLlamaLoRATarget.V), Is.True);
        Assert.That(targets.HasFlag(SilkLlamaLoRATarget.Gate), Is.True);
        Assert.That(targets.HasFlag(SilkLlamaLoRATarget.Up), Is.True);
        Assert.That(targets.HasFlag(SilkLlamaLoRATarget.Down), Is.True);
    }

    [Test]
    public void ParseTargets_Unknown_Throws()
    {
        var ex = Assert.Throws<ArgumentException>(() => SilkLoraAdapterLoader.ParseTargets("Q,banana"));
        Assert.That(ex!.Message, Does.Contain("Unknown LoRA target").IgnoreCase);
    }

    [Test]
    public void ValidateAndParse_InvalidAlpha_Throws()
    {
        var bundle = MakeBundle();
        bundle.Alpha = 0f;
        var slots = SilkLoraAdapterLoader.BuildExpectedSlots(SilkLlamaLoRATarget.Q | SilkLlamaLoRATarget.V, 1, 8, 8, 8, 16);
        var ex = Assert.Throws<InvalidOperationException>(() => SilkLoraAdapterLoader.ValidateAndParse(bundle, slots));
        Assert.That(ex!.Message, Does.Contain("alpha").IgnoreCase);
    }
}
