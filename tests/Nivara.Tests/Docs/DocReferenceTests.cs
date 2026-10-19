using NUnit.Framework;

namespace Nivara.Tests.Docs;

/// <summary>
/// Checks that every <c>File.cs:NN</c> citation and every <c>docs/…md</c> reference in the product
/// trees resolves. Motivation is #625: a deleted <c>docs/BERT-GPU.md</c> was still cited from
/// <c>samples/</c> and <c>tests/</c>, which <see cref="DocCitationTests"/> never reads; and #549, the
/// same blind spot for a deleted <c>docs/TODO.md</c>. The gate now states its own coverage: the files
/// it reads and the reference forms it recognises are pinned below, so a narrowing scope fails rather
/// than passes quietly.
/// </summary>
[TestFixture]
public class DocReferenceTests
{
    const string NegativeControl = "tests/Nivara.Tests/Docs/Fixtures/BrokenReferences.md";

    /// <summary>
    /// Files read by the widened scan. Pinned deliberately: a drop is reduced coverage.
    /// 696 = 691 + the six new Silk/model files (<c>WeightLayout</c>, <c>BertEncoderGpuTypes</c>,
    /// <c>GpuSeam</c>, <c>SilkLauncher</c>, <c>SilkModernBertSession</c>, <c>SilkLayaSession</c>)
    /// minus the deleted samples <c>IBertEncoderRunner.cs</c>. 698 = 696 +
    /// <c>docs/BLAS-MKL.md</c> and <c>tests/Nivara.PerformanceTests/LayaForwardProbe.cs</c>
    /// added by the #492 reconciliation. 700 = 698 + the two SmolLM Silk files
    /// (<c>SilkSmolLMGpuRunner.cs</c>, <c>SilkSmolLMSession.cs</c>) added by #628. 710 = 709 +
    /// <c>docs/DECISION-MODELS.md</c>, the decision-model ecosystem doc split out of the
    /// inference sample's README and <c>docs/LAYA.md</c>. 711 = 710 + the #670 fused-decode
    /// fixture, <c>tests/Nivara.Tests/Gpu/SilkRmsNormGemvTests.cs</c>. 712 = 711 + the #670
    /// device-free geometry fixture, <c>tests/Nivara.Tests/Gpu/SilkGemvGeometryTests.cs</c>, which
    /// was split out of that one so the bit-identity fixture stays byte-identical while the
    /// geometry it relies on is reshaped. 715 = 714 + the #681 bit-identity fixture,
    /// <c>tests/Nivara.Tests/Gpu/SilkGemmGemvAddTests.cs</c>. 722 = 721 + <c>docs/443-PLAN.md</c>.
    /// The merge of PR #726 added three counted files without moving the pin —
    /// <c>samples/Nivara.Samples/Incident/GpuOffload.cs</c>,
    /// <c>src/Nivara.Extensions/Silk/SilkAnalytics.cs</c>, and
    /// <c>tests/Nivara.Tests/Qwen/SynthesizeTensorsBiasGatingTests.cs</c> — so the merged tree was
    /// actually 725 and this gate was red on <c>main</c>. 729 = 725 + the four files the #443
    /// Phase 2/3 work adds: <c>samples/Nivara.Samples/Incident/GpuProfitability.cs</c>,
    /// <c>tests/Nivara.Tests/Gpu/SilkGroupedHashReduceTests.cs</c>,
    /// <c>tests/Nivara.Tests/Incident/GpuIncidentTests.cs</c>, and
    /// <c>tests/Nivara.Tests/Incident/GpuProfitabilityTests.cs</c>.
    /// </summary>
    const int ExpectedFileCount = 730;

    /// <summary><c>File.cs:NN</c> citations discovered across the widened scope. Pinned deliberately.</summary>
    const int ExpectedCodeCitationCount = 127;

    [Test]
    public void EveryReferenceInTheProductTrees_Resolves()
    {
        var faults = DocReferenceExtractor.AllFaults();

        Assert.That(faults, Is.Empty, Environment.NewLine + string.Join(Environment.NewLine, faults.Select(f => f.ToString())));
    }

    [Test]
    public void Coverage_IsAccountedForAcrossTheWidenedScope()
    {
        Assert.Multiple(() =>
        {
            Assert.That(DocReferenceExtractor.FileCount(), Is.EqualTo(ExpectedFileCount),
                "the widened scan file count moved; update this number deliberately. A drop means files "
                + "stopped being read, which is reduced coverage, not a neutral event");

            Assert.That(DocReferenceExtractor.CodeCitationCount(), Is.EqualTo(ExpectedCodeCitationCount),
                "the code-citation count across the widened scope moved; update this number deliberately");
        });
    }

    [Test]
    public void NegativeControl_RejectsMissingDocuments_AndAcceptsResolvableOnes()
    {
        var faults = DocReferenceExtractor.Faults(NegativeControl);

        Assert.Multiple(() =>
        {
            Assert.That(faults.Where(f => f.Kind == DocCitationFaultKind.MissingDocument).Select(f => f.Reference),
                Is.EquivalentTo(new[]
                {
                    "docs/NoSuchDocumentAnywhere.md",
                    "docs/gone/AlsoMissing.md",
                    "docs/also/Gone.md",
                }));

            Assert.That(faults.Select(f => f.Reference), Does.Not.Contain("docs/ILGPU.md"),
                "the negative control rejects a reference that resolves; that would prove nothing");
        });
    }

    [Test]
    public void NegativeControl_StillRejectsMissingCodeCitations()
    {
        var faults = DocReferenceExtractor.Faults(NegativeControl);

        Assert.That(faults.Where(f => f.Kind == DocCitationFaultKind.MissingFile).Select(f => f.Reference),
            Does.Contain("NoSuchFileAnywhere.cs:3"),
            "the widened scan must still check File.cs:NN, not only document references");
    }

    [Test]
    public void ToleratedHistoricalMentions_ArePresent_AndOnlyTheAllowlistKeepsThemGreen()
    {
        foreach (var (file, target) in DocReferenceExtractor.ToleratedReferences)
        {
            // The transient plan document's absence is what makes the counterfactual observable, so
            // while it exists (an iterative-work session mid-plan) its three entries cannot be checked
            // and are skipped; CI runs on a plan-less tree, where every entry is checked in full.
            if (DocSnippetExtractor.ScanExcludedDocuments.Contains(target, StringComparer.Ordinal)
                && File.Exists(Path.Combine(DocSnippetExtractor.RepoRoot, target)))
                continue;

            var raw = DocReferenceExtractor.DocumentFaults(file, tolerate: false)
                .Where(f => f.Reference == target);

            Assert.That(raw, Is.Not.Empty,
                $"{file} must actually name {target}, or the allowlist entry proves nothing");

            Assert.That(DocReferenceExtractor.DocumentFaults(file).Where(f => f.Reference == target), Is.Empty,
                $"{file} -> {target} is tolerated history and must not surface as a fault");
        }
    }
}
