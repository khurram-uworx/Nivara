using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public class RetrievalLabelsTests
{
    [Test]
    public void SelectSource_NoRequestAndJudgeAbsent_RefusesOrFallsBackToAuto()
    {
        Assert.Throws<ArgumentException>(() => RetrievalLabels.SelectSource(null, false));
    }

    [Test]
    public void SelectSource_NoRequestAndJudgePresent_ReturnsJudge()
    {
        Assert.That(RetrievalLabels.SelectSource(null, true), Is.EqualTo(LabelKind.Judge));
        Assert.That(RetrievalLabels.SelectSource("  ", true), Is.EqualTo(LabelKind.Judge));
    }

    [Test]
    public void SelectSource_ExplicitValue_OverridesJudgePresence()
    {
        Assert.That(RetrievalLabels.SelectSource("judge", false), Is.EqualTo(LabelKind.Judge));
        Assert.That(RetrievalLabels.SelectSource("auto", true), Is.EqualTo(LabelKind.Auto));
        Assert.That(RetrievalLabels.SelectSource("fixture", false), Is.EqualTo(LabelKind.Judge));
    }

    [Test]
    public void SelectSource_UnknownValue_Throws()
    {
        Assert.Throws<ArgumentException>(() => RetrievalLabels.SelectSource("nope", true));
    }

    [Test]
    public void CoverageGate_BelowMinimum_Refuses()
    {
        var labels = new ResolvedLabels(LabelKind.Judge, "d", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 49, 100);
        Assert.That(RetrievalLabels.Accepts(labels), Is.False);
    }

    [Test]
    public void CoverageGate_AtMinimum_Accepts()
    {
        var labels = new ResolvedLabels(LabelKind.Judge, "d", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 50, 100);
        Assert.That(RetrievalLabels.Accepts(labels), Is.True);
    }

    [Test]
    public void CoverageGate_NoRows_Refuses()
    {
        var labels = new ResolvedLabels(LabelKind.Judge, "d", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 0, 0);
        Assert.That(RetrievalLabels.Accepts(labels), Is.False);
    }

    [Test]
    public void IsComplete_RequiresEveryRow()
    {
        var labels = new ResolvedLabels(LabelKind.Judge, "d", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 5, 5);
        Assert.That(labels.IsComplete, Is.True);
        var partial = new ResolvedLabels(LabelKind.Judge, "d", new Dictionary<string, IReadOnlyDictionary<int, int>>(), 4, 5);
        Assert.That(partial.IsComplete, Is.False);
    }
}
