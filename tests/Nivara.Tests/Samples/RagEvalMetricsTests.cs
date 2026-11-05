using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public class RagEvalMetricsTests
{
    [Test]
    public void PrecisionAtK_CountsHitsOverK()
    {
        var retrieved = new[] { "a", "b", "c" };
        var relevant = new HashSet<string> { "a", "c" };
        Assert.That(RagEvalMetrics.PrecisionAtK(retrieved, relevant, 2), Is.EqualTo(0.5));
    }

    [Test]
    public void RecallAtK_UsesRelevantCount()
    {
        var retrieved = new[] { "a", "b" };
        var relevant = new HashSet<string> { "a", "c" };
        Assert.That(RagEvalMetrics.RecallAtK(retrieved, relevant, 2), Is.EqualTo(0.5));
    }

    [Test]
    public void Mrr_ReturnsReciprocalRank()
    {
        var retrieved = new[] { "x", "a" };
        var relevant = new HashSet<string> { "a" };
        Assert.That(RagEvalMetrics.Mrr(retrieved, relevant), Is.EqualTo(0.5));
    }

    [Test]
    public void NdcgAtK_PerfectOrdering_IsOne()
    {
        var retrieved = new[] { "a", "b" };
        var grades = new Dictionary<string, int> { ["a"] = 2, ["b"] = 1 };
        Assert.That(RagEvalMetrics.NdcgAtK(retrieved, grades, 2), Is.EqualTo(1.0).Within(1e-9));
    }

    [Test]
    public void SignTestPValue_NoDifference_IsOne()
    {
        Assert.That(RagEvalMetrics.SignTestPValue(0, 0), Is.EqualTo(1.0));
    }

    [Test]
    public void EvalGroundTruth_Load_ParsesGrades()
    {
        var dir = Path.Combine(Path.GetTempPath(), "rag-eval-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, "queries.tsv");
        File.WriteAllText(path, "q1\tc1:2,c2\tnote\n# comment\n\nq2\tc3\n");

        var queries = EvalGroundTruth.Load(path);
        Assert.That(queries.Count, Is.EqualTo(2));
        Assert.That(queries[0].Relevant["c1"], Is.EqualTo(2));
        Assert.That(queries[0].Relevant["c2"], Is.EqualTo(1));
        Assert.That(queries[0].Notes, Is.EqualTo("note"));

        Directory.Delete(dir, true);
    }
}
