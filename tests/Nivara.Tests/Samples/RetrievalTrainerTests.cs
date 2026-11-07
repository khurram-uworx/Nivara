using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers #915: the trainable fusion must be a strict superset of the #914 fixed fusion at
/// initialisation (so a measured delta is attributable to learning), the candidate pool must not let
/// the baseline pick its own competitors, labels must survive docs edits, and the split must be
/// reproducible across processes.
/// </summary>
[TestFixture]
public class RetrievalTrainerTests
{
    const int ViewCount = TrainingSetBuilder.ViewCount;

    static (float[] Scores, bool[] Present) Block(
        params (float? Content, float? Title, float? Section, float? Code)[] rows)
    {
        var scores = new float[rows.Length * ViewCount];
        var present = new bool[rows.Length * ViewCount];
        for (var row = 0; row < rows.Length; row++)
        {
            float?[] values = [rows[row].Content, rows[row].Title, rows[row].Section, rows[row].Code];
            for (var view = 0; view < ViewCount; view++)
            {
                scores[row * ViewCount + view] = values[view] ?? 0f;
                present[row * ViewCount + view] = values[view] is not null;
            }
        }

        return (scores, present);
    }

    static float[] Target(params int[] grades)
    {
        var weights = grades.Select(grade => MathF.Pow(TrainingSetBuilder.GradeBase, grade)).ToArray();
        var total = weights.Sum();
        return [.. weights.Select(weight => weight / total)];
    }

    static TrainingQuery Query(
        string name,
        (float? Content, float? Title, float? Section, float? Code)[] rows,
        int[] grades,
        float[]? vector = null)
    {
        var (scores, present) = Block(rows);
        var candidates = Enumerable.Range(0, grades.Length)
            .Select(row => new RankedCandidate(row, $"{name}#{row}", grades[row]))
            .ToList();

        return new TrainingQuery(
            name,
            vector ?? [0f, 0f],
            candidates,
            scores,
            present,
            Target(grades),
            0,
            "test");
    }

    /// <summary>
    /// Four rows with deliberately unequal view coverage: full, content+title, code-only, section-only.
    /// A fusion that ignores the presence mask cannot reproduce the fixed baseline on this shape.
    /// </summary>
    static TrainingQuery MixedPresenceQuery() => Query(
        "mixed presence",
        [
            (0.90f, 0.40f, 0.20f, 0.10f),
            (0.30f, 0.80f, null, null),
            (null, null, null, 0.95f),
            (null, null, 0.60f, null)
        ],
        [2, 1, 0, 0]);

    /// <summary>
    /// The fixed fusion ranks row 1 first (Content dominates at 0.5) even though row 0 is the graded
    /// relevant result, so training has something real to correct rather than a ranking already right.
    /// </summary>
    static TrainingQuery MisrankedQuery(string name = "misranked") => Query(
        name,
        [
            (0.30f, 0.90f, 0.50f, 0.50f),
            (0.90f, 0.10f, 0.50f, 0.50f),
            (0.60f, 0.20f, 0.50f, 0.50f)
        ],
        [2, 0, 1]);

    static MultiViewChunk Chunk(string path, int index) => new(
        $"{path}#{index}",
        Path.GetFileName(path),
        path,
        "Doc",
        $"Doc > S{index}",
        "docs",
        $"content {index}",
        null);

    [Test]
    public void FusionModel_BeforeTraining_MatchesTheFixedFusion()
    {
        var query = MixedPresenceQuery();
        using var model = new FusionModel();

        var learned = RetrievalEvaluation.Scores(model, query);
        var baseline = RetrievalEvaluation.FixedScores(query, FusionWeights.Default);

        Assert.Multiple(() =>
        {
            Assert.That(learned, Has.Length.EqualTo(baseline.Length));
            for (var row = 0; row < baseline.Length; row++)
                Assert.That(learned[row], Is.EqualTo(baseline[row]).Within(1e-5), $"row {row}");
        });
    }

    [Test]
    public void FusionModel_WeightsBeforeTraining_AreTheDefaultFusionWeights()
    {
        using var model = new FusionModel();

        var weights = model.Weights;

        Assert.Multiple(() =>
        {
            for (var view = 0; view < ViewCount; view++)
                Assert.That(weights[view], Is.EqualTo(FusionWeights.Default[(ViewSpace)view]).Within(1e-6));
        });
    }

    [Test]
    public void GatedFusionModel_BeforeTraining_IsInvariantToItsGateInput()
    {
        using var model = new GatedFusionModel(GateFeatureKind.ScoreStats, queryDimension: 2);

        var first = model.WeightsFor(MixedPresenceQuery());
        var second = model.WeightsFor(MisrankedQuery());

        Assert.Multiple(() =>
        {
            for (var view = 0; view < ViewCount; view++)
            {
                Assert.That(first[view], Is.EqualTo(second[view]).Within(1e-6));
                Assert.That(first[view], Is.EqualTo(FusionWeights.Default[(ViewSpace)view]).Within(1e-6));
            }
        });
    }

    [Test]
    public void GatedFusionModel_BeforeTraining_MatchesTheFixedFusion()
    {
        var query = MixedPresenceQuery();
        using var model = new GatedFusionModel(GateFeatureKind.ScoreStats, queryDimension: 2);

        var learned = RetrievalEvaluation.Scores(model, query);
        var baseline = RetrievalEvaluation.FixedScores(query, FusionWeights.Default);

        Assert.Multiple(() =>
        {
            for (var row = 0; row < baseline.Length; row++)
                Assert.That(learned[row], Is.EqualTo(baseline[row]).Within(1e-5), $"row {row}");
        });
    }

    [Test]
    public void GatedFusionModel_FeatureVector_IsPerViewMeanAndMaxOverPresentRows()
    {
        var query = Query(
            "stats",
            [
                (0.20f, 0.40f, null, 0.80f),
                (0.60f, null, null, null)
            ],
            [2, 0]);
        using var model = new GatedFusionModel(GateFeatureKind.ScoreStats, queryDimension: 2);

        var features = model.FeatureVector(query);

        Assert.Multiple(() =>
        {
            Assert.That(features, Has.Length.EqualTo(ViewCount * 2));
            Assert.That(features[(int)ViewSpace.Content], Is.EqualTo(0.40f).Within(1e-6));
            Assert.That(features[(int)ViewSpace.Title], Is.EqualTo(0.40f).Within(1e-6));
            Assert.That(features[(int)ViewSpace.Section], Is.EqualTo(0f));
            Assert.That(features[(int)ViewSpace.Code], Is.EqualTo(0.80f).Within(1e-6));
            Assert.That(features[ViewCount + (int)ViewSpace.Content], Is.EqualTo(0.60f).Within(1e-6));
            Assert.That(features[ViewCount + (int)ViewSpace.Title], Is.EqualTo(0.40f).Within(1e-6));
            Assert.That(features[ViewCount + (int)ViewSpace.Section], Is.EqualTo(0f));
            Assert.That(features[ViewCount + (int)ViewSpace.Code], Is.EqualTo(0.80f).Within(1e-6));
        });
    }

    [Test]
    public void Train_SameSeedAndData_ProducesIdenticalWeights()
    {
        var options = new TrainingOptions(Epochs: 10, LearningRate: 0.1f, WeightDecay: 0.01f, Seed: 7);
        var queries = new[] { MisrankedQuery("a"), MisrankedQuery("b") };

        using var first = new FusionModel();
        using var second = new FusionModel();
        RetrievalTrainer.Train(first, queries, [], options);
        RetrievalTrainer.Train(second, queries, [], options);

        Assert.That(second.Weights, Is.EqualTo(first.Weights).Within(1e-7));
    }

    [Test]
    public void Train_OneQuery_MovesWeightsAwayFromTheDefaultAndKeepsThemNormalized()
    {
        using var model = new FusionModel();

        var report = RetrievalTrainer.Train(
            model,
            [MisrankedQuery()],
            [],
            new TrainingOptions(Epochs: 25, LearningRate: 0.2f, WeightDecay: 0f));

        var weights = model.Weights;
        var moved = Enumerable.Range(0, ViewCount)
            .Max(view => Math.Abs(weights[view] - FusionWeights.Default[(ViewSpace)view]));

        Assert.Multiple(() =>
        {
            Assert.That(report.OptimizerSteps, Is.EqualTo(25));
            Assert.That(weights.Sum(), Is.EqualTo(1f).Within(1e-5));
            Assert.That(moved, Is.GreaterThan(1e-3), "a nonzero step must move at least one weight");
        });
    }

    [Test]
    public void Train_ListwiseLoss_FallsBelowItsStartingValue()
    {
        using var model = new FusionModel();

        var report = RetrievalTrainer.Train(
            model,
            [MisrankedQuery()],
            [],
            new TrainingOptions(Epochs: 100, LearningRate: 0.2f, WeightDecay: 0f));

        Assert.Multiple(() =>
        {
            Assert.That(report.History, Has.Count.EqualTo(100));
            Assert.That(report.FinalTrainLoss, Is.LessThan(report.History[0].TrainLoss));
        });
    }

    [Test]
    public void Train_ListwiseLoss_PromotesTheRelevantCandidateThatTheBaselineRankedLast()
    {
        var query = MisrankedQuery();
        var before = RetrievalEvaluation.Rank(
            RetrievalEvaluation.FixedScores(query, FusionWeights.Default), topK: 3);

        using var model = new FusionModel();
        RetrievalTrainer.Train(
            model,
            [query],
            [],
            new TrainingOptions(Epochs: 100, LearningRate: 0.2f, WeightDecay: 0f));

        var after = RetrievalEvaluation.Rank(RetrievalEvaluation.Scores(model, query), topK: 3);

        Assert.Multiple(() =>
        {
            Assert.That(before[0], Is.EqualTo(1), "fixture premise: the baseline ranks the wrong row first");
            Assert.That(after[0], Is.EqualTo(0), "training must promote the graded relevant row");
        });
    }

    [Test]
    public void Train_PairwiseLoss_PromotesTheRelevantCandidateThatTheBaselineRankedLast()
    {
        var query = MisrankedQuery();
        using var model = new FusionModel();
        RetrievalTrainer.Train(
            model,
            [query],
            [],
            new TrainingOptions(Epochs: 100, LearningRate: 0.2f, WeightDecay: 0f, Loss: LossKind.Pairwise));

        var scores = RetrievalEvaluation.Scores(model, query);

        Assert.Multiple(() =>
        {
            Assert.That(scores[0], Is.GreaterThan(scores[1]));
            Assert.That(scores[2], Is.GreaterThan(scores[1]));
        });
    }

    [Test]
    public void Train_WithValidation_RestoresTheBestCheckpointAndStopsEarly()
    {
        var validation = MisrankedQuery("validation");
        using var model = new FusionModel();

        var report = RetrievalTrainer.Train(
            model,
            [MisrankedQuery()],
            [validation],
            new TrainingOptions(Epochs: 100, LearningRate: 0.2f, WeightDecay: 0f, Patience: 3));

        Assert.Multiple(() =>
        {
            Assert.That(report.StoppedEarly, Is.True);
            Assert.That(report.BestEpoch, Is.LessThanOrEqualTo(report.History.Count));
            Assert.That(
                RetrievalEvaluation.MeanNdcg(model, [validation]),
                Is.EqualTo(report.BestValidationNdcg).Within(1e-9),
                "the restored parameters must be the ones that scored the reported validation nDCG");
        });
    }

    [Test]
    public void Train_WithoutValidation_UsesEveryEpoch()
    {
        using var model = new FusionModel();

        var report = RetrievalTrainer.Train(
            model,
            [MisrankedQuery()],
            [],
            new TrainingOptions(Epochs: 7, LearningRate: 0.05f, WeightDecay: 0f, Patience: 2));

        Assert.Multiple(() =>
        {
            Assert.That(report.StoppedEarly, Is.False);
            Assert.That(report.BestEpoch, Is.EqualTo(7));
            Assert.That(report.History, Has.Count.EqualTo(7));
        });
    }

    [Test]
    public void Train_QueryWithNoPositive_IsSkippedRatherThanTrainedOn()
    {
        var negative = Query("negative", [(0.5f, 0.5f, 0.5f, 0.5f), (0.1f, 0.1f, 0.1f, 0.1f)], [0, 0]);
        using var model = new FusionModel();

        var report = RetrievalTrainer.Train(model, [negative], [], new TrainingOptions(Epochs: 5));

        Assert.Multiple(() =>
        {
            Assert.That(negative.IsUsable, Is.False);
            Assert.That(report.QueriesUsed, Is.EqualTo(0));
            Assert.That(report.History, Is.Empty);
        });
    }

    [Test]
    public void CandidatePool_UnionsEachViewTopKWithEveryLabeledPositive()
    {
        ViewScores[] scores =
        [
            new(0.90f, null, null, null),
            new(0.80f, null, null, null),
            new(null, 0.90f, null, null),
            new(null, null, null, 0.95f),
            new(0.10f, null, 0.70f, null),
            new(null, null, null, null)
        ];

        var selected = CandidatePool.Select(
            scores,
            new Dictionary<int, int> { [5] = 2, [1] = 0 },
            perView: 1);

        Assert.That(selected, Is.EqualTo(new[] { 0, 2, 3, 4, 5 }));
    }

    [Test]
    public void TrainingSetBuilder_RowWithNoScoredView_IsDroppedAndCounted()
    {
        var chunks = new List<MultiViewChunk> { Chunk("docs/a.md", 0), Chunk("docs/a.md", 1), Chunk("docs/a.md", 2) };
        var scores = new[]
        {
            new ViewScores(null, null, null, null),
            new ViewScores(0.8f, 0.5f, 0.4f, 0.3f),
            new ViewScores(0.2f, 0.1f, 0.1f, 0.1f)
        };

        var training = TrainingSetBuilder.Create(
            "q",
            [0f, 0f],
            chunks,
            scores,
            new Dictionary<int, int> { [0] = 2 },
            "test");

        Assert.Multiple(() =>
        {
            Assert.That(training.DroppedCandidates, Is.EqualTo(1));
            Assert.That(training.RowCount, Is.EqualTo(2));
            Assert.That(training.Candidates.Select(candidate => candidate.Row), Is.EqualTo(new[] { 1, 2 }));
        });
    }

    [Test]
    public void TrainingSetBuilder_Target_WeightsGradesByPowersOfTwo()
    {
        var chunks = new List<MultiViewChunk> { Chunk("docs/a.md", 0), Chunk("docs/a.md", 1), Chunk("docs/a.md", 2) };
        var scores = new[]
        {
            new ViewScores(0.9f, null, null, null),
            new ViewScores(0.8f, null, null, null),
            new ViewScores(0.7f, null, null, null)
        };

        var training = TrainingSetBuilder.Create(
            "q",
            [0f, 0f],
            chunks,
            scores,
            new Dictionary<int, int> { [0] = 2, [1] = 1, [2] = 0 },
            "test");

        Assert.That(training.LossTarget, Is.EqualTo(new[] { 4f / 7f, 2f / 7f, 1f / 7f }).Within(1e-6));
    }

    [Test]
    public void QuerySplit_IsReproducibleAndPartitionsEveryQuery()
    {
        var queries = Enumerable.Range(0, 200).Select(index => $"query {index}").ToArray();

        var first = QuerySplit.Create(queries, seed: 42);
        var second = QuerySplit.Create(queries, seed: 42);
        var other = QuerySplit.Create(queries, seed: 43);

        Assert.Multiple(() =>
        {
            Assert.That(second.Train, Is.EqualTo(first.Train));
            Assert.That(second.Validation, Is.EqualTo(first.Validation));
            Assert.That(second.Test, Is.EqualTo(first.Test));

            var all = first.Train.Concat(first.Validation).Concat(first.Test).ToArray();
            Assert.That(all, Has.Length.EqualTo(200));
            Assert.That(all.Distinct(StringComparer.Ordinal).Count(), Is.EqualTo(200));
            Assert.That(first.Train.Intersect(first.Test, StringComparer.Ordinal), Is.Empty);

            Assert.That(
                other.Train.SequenceEqual(first.Train) && other.Test.SequenceEqual(first.Test),
                Is.False,
                "a different seed must move at least some queries");
        });
    }

    [Test]
    public void QuerySplit_FractionsThatCoverEverything_AreRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            QuerySplit.Create(["a", "b"], validationFraction: 0.6, testFraction: 0.4));
    }

    [Test]
    public void RetrievalLabels_LoadFixture_IgnoresCommentsAndKeepsTheHighestGrade()
    {
        var path = Path.Combine(Path.GetTempPath(), $"nivara-fixture-{Guid.NewGuid():N}.tsv");
        File.WriteAllLines(path,
        [
            "# comment",
            "",
            "how do I stream\tdocs/a.md\t1",
            "how do I stream\tdocs/a.md\t2",
            "how do I stream\tdocs/b.md\t0",
            "malformed line without tabs"
        ]);

        try
        {
            var fixture = RetrievalLabels.LoadFixture(path);

            Assert.Multiple(() =>
            {
                Assert.That(fixture, Has.Count.EqualTo(1));
                Assert.That(fixture["how do I stream"]["docs/a.md"], Is.EqualTo(2));
                Assert.That(fixture["how do I stream"]["docs/b.md"], Is.EqualTo(0));
            });
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Test]
    public void RetrievalLabels_ResolveFixture_ExpandsDocumentPathToEveryChunkOfThatDocument()
    {
        var chunks = new List<MultiViewChunk>
        {
            Chunk("docs/a.md", 0),
            Chunk("docs/a.md", 1),
            Chunk("docs/b.md", 0)
        };
        var labels = new[]
        {
            new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                "q",
                new Dictionary<string, int> { ["docs/a.md"] = 2 })
        };

        var resolved = RetrievalLabels.Resolve(LabelKind.Fixture, chunks, labels);

        Assert.Multiple(() =>
        {
            Assert.That(resolved.MatchedLabelRows, Is.EqualTo(1));
            Assert.That(resolved.TotalLabelRows, Is.EqualTo(1));
            Assert.That(resolved.Coverage, Is.EqualTo(1d));
            Assert.That(resolved.Grades["q"].Keys, Is.EquivalentTo(new[] { 0, 1 }));
            Assert.That(resolved.Grades["q"].Values, Is.All.EqualTo(2));
        });
    }

    [Test]
    public void RetrievalLabels_ResolveJudge_MatchesChunkIdsOnly()
    {
        var chunks = new List<MultiViewChunk> { Chunk("docs/a.md", 0), Chunk("docs/a.md", 1) };
        var labels = new[]
        {
            new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                "q",
                new Dictionary<string, int> { ["docs/a.md#1"] = 1 })
        };

        var resolved = RetrievalLabels.Resolve(LabelKind.Judge, chunks, labels);

        Assert.Multiple(() =>
        {
            Assert.That(resolved.Coverage, Is.EqualTo(1d));
            Assert.That(resolved.Grades["q"].Keys, Is.EqualTo(new[] { 1 }));
        });
    }

    [Test]
    public void RetrievalLabels_ResolveUnmatchedRows_ReportsReducedCoverage()
    {
        var chunks = new List<MultiViewChunk> { Chunk("docs/a.md", 0) };
        var labels = new[]
        {
            new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                "q",
                new Dictionary<string, int> { ["docs/a.md"] = 2, ["docs/gone.md"] = 1 })
        };

        var resolved = RetrievalLabels.Resolve(LabelKind.Fixture, chunks, labels);

        Assert.Multiple(() =>
        {
            Assert.That(resolved.MatchedLabelRows, Is.EqualTo(1));
            Assert.That(resolved.TotalLabelRows, Is.EqualTo(2));
            Assert.That(resolved.Coverage, Is.EqualTo(0.5d).Within(1e-9));
            Assert.That(resolved.IsComplete, Is.False);
        });
    }

    [Test]
    public void RetrievalEvaluation_Rank_BreaksScoreTiesByCandidateIndex()
    {
        var ranked = RetrievalEvaluation.Rank([0.5f, 0.9f, 0.9f, 0.1f], topK: 3);

        Assert.That(ranked, Is.EqualTo(new[] { 1, 2, 0 }));
    }

    [Test]
    public void RetrievalEvaluation_Compare_RejectsDifferentQueryCounts()
    {
        var metrics = new RankerMetrics(0d, 0d, 0d, 0d);
        var one = new RankerResult("one", metrics, [metrics], [0.1d]);
        var two = new RankerResult("two", metrics, [metrics, metrics], [0.1d, 0.2d]);

        Assert.Throws<ArgumentException>(() => RetrievalEvaluation.Compare(one, two));
    }
}
