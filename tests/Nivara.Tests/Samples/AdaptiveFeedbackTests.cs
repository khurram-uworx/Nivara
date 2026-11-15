using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveFeedbackTests
{
    static readonly MultiViewChunk[] Chunks =
    [
        new("a#0", "a.md", "a.md", "A", "A", "public", "first", null),
        new("b#0", "b.md", "b.md", "B", "B", "public", "second", null),
        new("c#0", "c.md", "c.md", "C", "C", "public", "third", null)
    ];

    static AdaptiveFrozenFeatures Feat() => new([1f, 2f, 3f, 4f], 2, 2, [0, 1]);

    [Test]
    public void Build_GradeZeroIsNotRelevant()
    {
        var grades = new Dictionary<string, IReadOnlyDictionary<int, int>>(StringComparer.Ordinal)
        {
            ["q"] = new Dictionary<int, int> { [0] = 0, [1] = 2 }
        };
        var labels = new ResolvedLabels(LabelKind.Judge, "judge", grades, 2, 2);
        var retrieved = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal)
        {
            ["q"] = new[] { 0, 1, 2 }
        };

        var result = AdaptiveFeedback.Build(
            labels, Chunks, ["q"],
            q => retrieved[q],
            (q, chunk) => Feat(),
            new AdaptiveBuildLimits(10, 10));

        Assert.That(result.Examples.Count, Is.EqualTo(2));
        Assert.That(result.Examples[0].Target, Is.EqualTo(1)); // grade 0 -> not relevant
        Assert.That(result.Examples[1].Target, Is.EqualTo(0)); // grade 2 -> relevant
    }

    [Test]
    public void Build_SkipsUnlabelledCandidates()
    {
        var grades = new Dictionary<string, IReadOnlyDictionary<int, int>>(StringComparer.Ordinal)
        {
            ["q"] = new Dictionary<int, int> { [0] = 1 }
        };
        var labels = new ResolvedLabels(LabelKind.Judge, "judge", grades, 1, 1);
        var retrieved = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal)
        {
            ["q"] = new[] { 0, 1, 2 }
        };

        var result = AdaptiveFeedback.Build(
            labels, Chunks, ["q"],
            q => retrieved[q],
            (q, chunk) => Feat(),
            new AdaptiveBuildLimits(10, 10));

        Assert.That(result.Examples.Count, Is.EqualTo(1));
        Assert.That(result.SkippedUnlabelled, Is.EqualTo(2));
    }

    [Test]
    public void Build_CapsQueriesAndExamplesDeterministically()
    {
        var grades = new Dictionary<string, IReadOnlyDictionary<int, int>>(StringComparer.Ordinal)
        {
            ["q1"] = new Dictionary<int, int> { [0] = 1, [1] = 1 },
            ["q2"] = new Dictionary<int, int> { [0] = 1 },
            ["q3"] = new Dictionary<int, int> { [0] = 1 },
        };
        var labels = new ResolvedLabels(LabelKind.Judge, "judge", grades, 4, 4);
        var retrieved = new Dictionary<string, IReadOnlyList<int>>(StringComparer.Ordinal)
        {
            ["q1"] = new[] { 0, 1 },
            ["q2"] = new[] { 0 },
            ["q3"] = new[] { 0 },
        };

        var result = AdaptiveFeedback.Build(
            labels, Chunks, ["q1", "q2", "q3"],
            q => retrieved[q],
            (q, chunk) => Feat(),
            new AdaptiveBuildLimits(2, 2));

        Assert.That(result.QueriesConsidered, Is.EqualTo(2));
        Assert.That(result.Examples.Count, Is.EqualTo(2));
        Assert.That(result.Truncated, Is.True);
        Assert.That(result.SkippedByCap, Is.GreaterThan(0));
    }

    [Test]
    public void Build_ReportsTruncationAndSkips()
    {
        var grades = new Dictionary<string, IReadOnlyDictionary<int, int>>(StringComparer.Ordinal)
        {
            ["q"] = new Dictionary<int, int> { [0] = 1 }
        };
        var labels = new ResolvedLabels(LabelKind.Judge, "judge", grades, 1, 1);
        var result = AdaptiveFeedback.Build(
            labels, Chunks, ["q"],
            q => new[] { 0 },
            (q, chunk) => Feat(),
            new AdaptiveBuildLimits(10, 10));

        Assert.That(result.CandidatesConsidered, Is.EqualTo(1));
        Assert.That(result.SkippedUnlabelled, Is.EqualTo(0));
        Assert.That(result.SkippedByCap, Is.EqualTo(0));
        Assert.That(result.Truncated, Is.False);
    }

    [Test]
    public void AllowsAutoLabels()
    {
        Assert.That(AdaptiveFeedback.AllowsAutoLabels(LabelKind.Judge), Is.True);
        Assert.That(AdaptiveFeedback.AllowsAutoLabels(LabelKind.Auto), Is.True);
        Assert.That(AdaptiveFeedback.AllowsAutoLabels(LabelKind.Fixture), Is.False);
    }
}
