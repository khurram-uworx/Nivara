using Nivara.Silk;
using NUnit.Framework;

namespace Nivara.Tests.Gpu;

/// <summary>
/// Pins the reusable partial/frozen training policy that every Silk trainer must share.
/// </summary>
public sealed class SilkPartialTrainingPolicyTests
{
    [Test]
    public void Normalize_ClampsNegativeAndOversized()
    {
        Assert.That(SilkPartialTrainingPolicy.Normalize(-1, 2, true), Is.EqualTo(2));
        Assert.That(SilkPartialTrainingPolicy.Normalize(-100, 2, true), Is.EqualTo(0));
        Assert.That(SilkPartialTrainingPolicy.Normalize(0, 2, true), Is.EqualTo(0));
        Assert.That(SilkPartialTrainingPolicy.Normalize(5, 2, true), Is.EqualTo(2));
        Assert.That(SilkPartialTrainingPolicy.Normalize(int.MaxValue, 2, true), Is.EqualTo(2));
        Assert.That(SilkPartialTrainingPolicy.Normalize(3, 2, false), Is.EqualTo(0));
    }

    [Test]
    public void EmbeddingsTrainable_OnlyWhenAllLayersTrainable()
    {
        Assert.That(SilkPartialTrainingPolicy.EmbeddingsTrainable(2, 2, true), Is.True);
        Assert.That(SilkPartialTrainingPolicy.EmbeddingsTrainable(1, 2, true), Is.False);
        Assert.That(SilkPartialTrainingPolicy.EmbeddingsTrainable(2, 2, false), Is.False);
    }

    [Test]
    public void LayerTrainable_SelectsTopSuffix()
    {
        // 2 layers, 1 trainable => only layer index 1.
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(0, 2, 1, true), Is.False);
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(1, 2, 1, true), Is.True);
        // 0 trainable => none.
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(0, 2, 0, true), Is.False);
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(1, 2, 0, true), Is.False);
        // all trainable => both.
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(0, 2, 2, true), Is.True);
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(1, 2, 2, true), Is.True);
        // frozen globally => none.
        Assert.That(SilkPartialTrainingPolicy.LayerTrainable(1, 2, 2, false), Is.False);
    }

    [Test]
    public void HeadTrainable_FollowsTrainableFlag()
    {
        Assert.That(SilkPartialTrainingPolicy.HeadTrainable(true), Is.True);
        Assert.That(SilkPartialTrainingPolicy.HeadTrainable(false), Is.False);
    }
}
