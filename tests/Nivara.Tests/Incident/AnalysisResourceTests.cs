using Nivara.Samples.Incident;
using Nivara.Tests.IO;
using NUnit.Framework;

namespace Nivara.Tests.Incident;

/// <summary>
/// Gates that every <see cref="Analysis"/> entry point releases the Parquet file
/// handle it opens.
/// <para>
/// <c>ParquetLazySource</c> opens the file with <c>FileShare.Read</c> and no
/// <c>FileShare.Delete</c>, so a handle that outlives the query blocks
/// <c>Directory.Delete</c> with <see cref="IOException"/> — the fixture teardown
/// failure behind issue #496. Probing with <c>FileShare.None</c> fails while any
/// other handle is open, which detects the leak deterministically instead of relying
/// on finalizer timing.
/// </para>
/// <para>
/// The probe itself moved to <see cref="FileHandleProbe"/> in #498, which also covers the
/// consumers outside this sample. Unlike those, every entry point here reads to EOF through
/// <c>ParquetLazySource</c>, which does not release at EOF — so a plain open/dispose/probe
/// cycle is load-bearing here and needs no midpoint assertion.
/// </para>
/// </summary>
[TestFixture]
public class AnalysisResourceTests
{
    const int TotalRecords = 10_000;

    string tempDir = null!;
    string requestsPath = null!;

    [OneTimeSetUp]
    public void OneTimeSetUp()
    {
        tempDir = Path.Combine(Path.GetTempPath(), $"inc-analysis-resources-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        DatasetGenerator.GenerateFromRecordCount(tempDir, "A", TotalRecords);
        requestsPath = Path.Combine(tempDir, "requests.parquet");
    }

    [OneTimeTearDown]
    public void OneTimeTearDown()
    {
        if (tempDir is not null && Directory.Exists(tempDir))
            Directory.Delete(tempDir, true);
    }

    [Test]
    public void AnalyzeDegradationOrdering_ReleasesFileHandle()
    {
        var scenario = Scenarios.Get("A");
        AssertReleasesFileHandle(nameof(Analysis.AnalyzeDegradationOrdering),
            () => Analysis.AnalyzeDegradationOrdering(tempDir, scenario));
    }

    [Test]
    public void AnalyzeDeploymentCorrelation_ReleasesFileHandle()
    {
        var scenario = Scenarios.Get("A");
        AssertReleasesFileHandle(nameof(Analysis.AnalyzeDeploymentCorrelation),
            () => Analysis.AnalyzeDeploymentCorrelation(tempDir, scenario));
    }

    [Test]
    public void AnalyzeSaturationOrdering_ReleasesFileHandle()
    {
        var scenario = Scenarios.Get("A");
        AssertReleasesFileHandle(nameof(Analysis.AnalyzeSaturationOrdering),
            () => Analysis.AnalyzeSaturationOrdering(tempDir, scenario));
    }

    [Test]
    public void AnalyzeRegionalPartitioning_ReleasesFileHandle()
    {
        var scenario = Scenarios.Get("A");
        AssertReleasesFileHandle(nameof(Analysis.AnalyzeRegionalPartitioning),
            () => Analysis.AnalyzeRegionalPartitioning(tempDir, scenario));
    }

    [Test]
    public void AnalyzeGroupedAggregation_ReleasesFileHandle()
    {
        var scenario = Scenarios.Get("A");
        AssertReleasesFileHandle(nameof(Analysis.AnalyzeGroupedAggregation),
            () => Analysis.AnalyzeGroupedAggregation(tempDir, scenario));
    }

    [Test]
    public void AnalyzeGroupedAggregationWithTypedLinq_ReleasesFileHandle()
    {
        var scenario = Scenarios.Get("A");
        AssertReleasesFileHandle(nameof(Analysis.AnalyzeGroupedAggregationWithTypedLinq),
            () => Analysis.AnalyzeGroupedAggregationWithTypedLinq(tempDir, scenario));
    }

    void AssertReleasesFileHandle(string analysisName, Func<IDisposable?> analysis)
        => FileHandleProbe.AssertReleasesAfter(
            requestsPath,
            analysisName,
            analysis,
            result => result!.Dispose());
}