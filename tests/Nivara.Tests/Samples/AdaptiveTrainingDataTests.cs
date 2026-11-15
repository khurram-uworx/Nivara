using Nivara.AutoDiff.Utilities;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveTrainingDataTests
{
    static readonly MultiViewChunk[] Chunks =
    [
        new("a#0", "a.md", "a.md", "A", "A", "public", "first", null),
        new("b#0", "b.md", "b.md", "B", "B", "public", "second", null),
        new("c#0", "c.md", "c.md", "C", "C", "public", "third", null)
    ];

    [Test]
    public void Build_UsesJudgeGradesAndExtractsInStableOrder()
    {
        var grades = new Dictionary<string, IReadOnlyDictionary<int, int>>(StringComparer.Ordinal)
        {
            ["question"] = new Dictionary<int, int> { [2] = 2, [0] = 0, [1] = 1 }
        };
        var labels = new ResolvedLabels(LabelKind.Judge, "judge", grades, 3, 3);
        var calls = new List<string>();

        var examples = AdaptiveTrainingData.Build(labels, Chunks, (query, chunk) =>
        {
            calls.Add($"{query}:{chunk.ChunkId}");
            return new AdaptiveFrozenFeatures([1f, 2f, 3f, 4f], 2, 2, [0, 1]);
        });

        Assert.Multiple(() =>
        {
            Assert.That(calls, Is.EqualTo(new[] { "question:a#0", "question:b#0", "question:c#0" }));
            Assert.That(examples.Select(example => example.Target), Is.EqualTo(new[] { 1, 0, 0 }));
            Assert.That(examples.Select(example => example.ChunkId), Is.EqualTo(new[] { "a#0", "b#0", "c#0" }));
        });
    }

    [Test]
    public void Build_RejectsAutoLabels()
    {
        var labels = new ResolvedLabels(
            LabelKind.Auto,
            "auto",
            new Dictionary<string, IReadOnlyDictionary<int, int>>
            {
                ["question"] = new Dictionary<int, int> { [0] = 1 }
            },
            1,
            1);

        Assert.Throws<ArgumentException>(() => AdaptiveTrainingData.Build(
            labels,
            Chunks,
            (_, _) => new AdaptiveFrozenFeatures([1f, 2f], 1, 2, [0, 0])));
    }

    [Test]
    public void Build_RejectsMalformedFeatureShapesAndUnmappedGrades()
    {
        var labels = new ResolvedLabels(
            LabelKind.Judge,
            "judge",
            new Dictionary<string, IReadOnlyDictionary<int, int>>
            {
                ["question"] = new Dictionary<int, int> { [0] = 3 }
            },
            1,
            1);

        Assert.Throws<ArgumentOutOfRangeException>(() => AdaptiveTrainingData.Build(
            labels,
            Chunks,
            (_, _) => new AdaptiveFrozenFeatures([1f, 2f], 1, 2, [0, 0])));

        var validLabels = labels with
        {
            Grades = new Dictionary<string, IReadOnlyDictionary<int, int>>
            {
                ["question"] = new Dictionary<int, int> { [0] = 1 }
            }
        };
        Assert.Throws<ArgumentException>(() => AdaptiveTrainingData.Build(
            validLabels,
            Chunks,
            (_, _) => new AdaptiveFrozenFeatures([1f], 1, 2, [0, 0])));
    }

    [Test]
    public void Build_RejectsFeatureExtractionInsideGradientScope()
    {
        var labels = new ResolvedLabels(
            LabelKind.Judge,
            "judge",
            new Dictionary<string, IReadOnlyDictionary<int, int>>
            {
                ["question"] = new Dictionary<int, int> { [0] = 1 }
            },
            1,
            1);

        using (GradientUtils.Grad())
            Assert.Throws<InvalidOperationException>(() => AdaptiveTrainingData.Build(
                labels,
                Chunks,
                (_, _) => new AdaptiveFrozenFeatures([1f, 2f], 1, 2, [0, 0])));
    }

    [Test]
    public void Split_IsDeterministicAndKeepsQueriesDisjoint()
    {
        var examples = Enumerable.Range(0, 100)
            .Select(index => new AdaptiveTrainingExample(
                $"query-{index}", $"chunk-{index}", [1f, 2f], 1, 2, [0, 0], index % 2))
            .ToArray();

        var first = AdaptiveTrainingData.Split(examples, seed: 42);
        var second = AdaptiveTrainingData.Split(examples, seed: 42);

        Assert.Multiple(() =>
        {
            Assert.That(first.Train.Select(example => example.Query), Is.EqualTo(second.Train.Select(example => example.Query)));
            Assert.That(first.Validation.Select(example => example.Query), Is.EqualTo(second.Validation.Select(example => example.Query)));
            Assert.That(first.Test.Select(example => example.Query), Is.EqualTo(second.Test.Select(example => example.Query)));
            Assert.That(first.Train.Select(example => example.Query).Intersect(first.Test.Select(example => example.Query)), Is.Empty);
            Assert.That(first.Train.Select(example => example.Query).Intersect(first.Validation.Select(example => example.Query)), Is.Empty);
            Assert.That(first.Validation.Select(example => example.Query).Intersect(first.Test.Select(example => example.Query)), Is.Empty);
            Assert.That(first.Train.Count + first.Validation.Count + first.Test.Count, Is.EqualTo(examples.Length));
        });
    }
}
