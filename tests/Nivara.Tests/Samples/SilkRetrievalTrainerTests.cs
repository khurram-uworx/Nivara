using Nivara.Silk;
using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

/// <summary>
/// Covers the #915 Silk retrieval trainer: its scores and fusion must agree with the CPU
/// FusionModel/GatedFusionModel paths at initialisation, and training must run on the device
/// without falling back. Requires an OpenCL GPU; otherwise ignored.
/// </summary>
[TestFixture]
public class SilkRetrievalTrainerTests
{
    const int ViewCount = TrainingSetBuilder.ViewCount;

    static SilkRuntime? TryDevice()
    {
        try { return new SilkRuntime(); }
        catch { return null; }
    }

    static TrainingQuery MakeQuery()
    {
        float[] scores =
        [
            0.90f, 0.40f, 0.20f, 0.10f,
            0.30f, 0.80f, 0.00f, 0.00f,
            0.00f, 0.00f, 0.00f, 0.95f,
            0.00f, 0.00f, 0.60f, 0.00f
        ];
        bool[] present =
        [
            true, true, true, true,
            true, true, false, false,
            false, false, false, true,
            false, false, true, false
        ];
        var candidates = new[]
        {
            new RankedCandidate(0, "q#0", 2),
            new RankedCandidate(1, "q#1", 1),
            new RankedCandidate(2, "q#2", 0),
            new RankedCandidate(3, "q#3", 0)
        };
        var target = new[] { 0.6666667f, 0.2222222f, 0.05555556f, 0.05555556f };
        return new TrainingQuery("q", [0f, 0f], candidates, scores, present, target, 0, "test");
    }

    [Test]
    public void Global_Scores_MatchCpuFusionModel_AtInit()
    {
        using var device = TryDevice();
        if (device is null) Assert.Ignore("No OpenCL GPU available; skipping Silk retrieval verification.");

        var query = MakeQuery();
        var cpu = new FusionModel();
        using var silk = SilkRetrievalTrainer.CreateGlobal(0.05f, 0.01f, query.RowCount);
        var silkRanker = new SilkFusionRanker(silk);

        var cpuScores = RetrievalEvaluation.Scores(cpu, query);
        var silkScores = RetrievalEvaluation.Scores(silkRanker, query);

        Assert.That(silkScores.Length, Is.EqualTo(cpuScores.Length));
        for (var i = 0; i < cpuScores.Length; i++)
            Assert.That(silkScores[i], Is.EqualTo(cpuScores[i]).Within(1e-5f));
    }

    [Test]
    public void GatedStats_Scores_MatchCpuGatedFusionModel_AtInit()
    {
        using var device = TryDevice();
        if (device is null) Assert.Ignore("No OpenCL GPU available; skipping Silk retrieval verification.");

        var query = MakeQuery();
        var cpu = new GatedFusionModel(GateFeatureKind.ScoreStats, TrainingSetBuilder.ViewCount * 2);
        using var silk = SilkRetrievalTrainer.CreateForStats(0.05f, 0.01f, query.RowCount);
        var silkRanker = new SilkFusionRanker(silk);

        var cpuScores = RetrievalEvaluation.Scores(cpu, query);
        var silkScores = RetrievalEvaluation.Scores(silkRanker, query);

        Assert.That(silkScores.Length, Is.EqualTo(cpuScores.Length));
        for (var i = 0; i < cpuScores.Length; i++)
            Assert.That(silkScores[i], Is.EqualTo(cpuScores[i]).Within(1e-5f));
    }

    [Test]
    public void Global_Trains_WithoutFallback()
    {
        using var device = TryDevice();
        if (device is null) Assert.Ignore("No OpenCL GPU available; skipping Silk retrieval verification.");

        var query = MakeQuery();
        using var silk = SilkRetrievalTrainer.CreateGlobal(0.05f, 0.01f, query.RowCount);
        var report = silk.Train([query], [], new TrainingOptions(Epochs: 2, LearningRate: 0.05f), topK: 2);

        Assert.That(report.History.Count, Is.EqualTo(2));
        foreach (var epoch in report.History)
            Assert.That(double.IsFinite(epoch.TrainLoss), Is.True);
    }

    [Test]
    public void GatedStats_Trains_WithoutFallback()
    {
        using var device = TryDevice();
        if (device is null) Assert.Ignore("No OpenCL GPU available; skipping Silk retrieval verification.");

        var query = MakeQuery();
        using var silk = SilkRetrievalTrainer.CreateForStats(0.05f, 0.01f, query.RowCount);
        var report = silk.Train([query], [], new TrainingOptions(Epochs: 2, LearningRate: 0.05f), topK: 2);

        Assert.That(report.History.Count, Is.EqualTo(2));
        foreach (var epoch in report.History)
            Assert.That(double.IsFinite(epoch.TrainLoss), Is.True);
    }
}
