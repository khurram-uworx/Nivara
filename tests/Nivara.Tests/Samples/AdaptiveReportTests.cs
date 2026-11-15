using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveReportTests
{
    static AdaptiveReportInput MakeInput() => new()
    {
        Mode = "adaptive-eval (cpu)",
        LabelKind = "judge",
        LabelDescription = "LLM judge",
        LabelMatched = 5,
        LabelTotal = 5,
        LabelCoverage = 1.0,
        LabelQueries = 3,
        Seed = 42,
        K = 10,
        Threshold = 0.8,
        MaxQueries = 40,
        MaxExamples = 128,
        CorpusFiles = 12,
        CorpusChunks = 100,
        CorpusContentHash = "abc",
        CorpusChunkerVersion = "v1",
        CorpusChunkSize = 500,
        CorpusBackend = "cpu",
        CorpusDimension = 384,
        TrainQueries = 2,
        TestQueries = 1,
        TrainExamples = 4,
        TestExamples = 2,
        TrainFeatureBytes = 1000,
        TestFeatureBytes = 500,
        Leakage = false,
        Resume = false,
        UpdateBatch = 32,
        HeadEpochs = 5,
        HeadLearningRate = 0.0005,
        RetrievalRows =
        [
            new AdaptiveReportRow("fixed", 0.5, 0.4, 0.3, 0.6, 1),
            new AdaptiveReportRow("adaptive-rerank", 0.5, 0.4, 0.35, 0.62, 1),
        ],
        Comparisons = [],
        DecisionBefore = new AdaptiveReportDecision(2, 1, 0.5, 0.5, 0.5, 0.5),
        DecisionAfter = new AdaptiveReportDecision(2, 2, 1.0, 1.0, 1.0, 1.0),
        CalibrationBefore = new AdaptiveReportCalibration(2, 0.1, 0.05),
        CalibrationAfter = new AdaptiveReportCalibration(2, 0.05, 0.02),
        ReliabilityBefore = [],
        ReliabilityAfter = [],
        Timings = [new AdaptiveReportTiming("x", 1, 5)],
        Silk = new AdaptiveReportSilk("cpu", 0.0, 0, true, 0, 0, false, false, true),
        Persistence = new AdaptiveReportPersistence("/tmp/h.json", false, 1, 100, 100, true),
        InSampleAgreement = [],
        DecisiveNote = "note",
    };

    [Test]
    public void Markdown_IsByteStableForIdenticalInput()
    {
        var a = AdaptiveReport.Markdown(MakeInput());
        var b = AdaptiveReport.Markdown(MakeInput());
        Assert.That(a, Is.EqualTo(b));
    }

    [Test]
    public void Markdown_StatesAutoLabelLeakage()
    {
        var input = MakeInput() with { Leakage = true };
        var md = AdaptiveReport.Markdown(input);
        Assert.That(md, Does.Contain("leakage diagnostic"));
    }

    [Test]
    public void Markdown_StatesRerankInvariance()
    {
        var md = AdaptiveReport.Markdown(MakeInput());
        Assert.That(md, Does.Contain("invariant by construction"));
    }

    [Test]
    public void Markdown_ContainsNoClockOrDate()
    {
        var md = AdaptiveReport.Markdown(MakeInput());
        Assert.That(md, Does.Not.Match(@"\d{4}-\d{2}-\d{2}"));
        Assert.That(md, Does.Not.Contain("DateTime"));
    }

    [Test]
    public void MetricsJson_RoundTrips()
    {
        var json = AdaptiveReport.MetricsJson(MakeInput());
        using var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.That(doc.RootElement.GetProperty("schemaVersion").GetInt32(), Is.EqualTo(1));
        Assert.That(doc.RootElement.GetProperty("seed").GetInt32(), Is.EqualTo(42));
    }

    [Test]
    public void ResolvePath_RelativeLandsUnderAdaptive()
    {
        var path = AdaptiveReport.ResolvePath("x.md", "E:/repo");
        Assert.That(path, Does.EndWith(Path.Combine("adaptive", "x.md")));
    }
}
