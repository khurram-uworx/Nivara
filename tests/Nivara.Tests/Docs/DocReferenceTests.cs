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
    /// (<c>SilkSmolLMGpuRunner.cs</c>, <c>SilkSmolLMSession.cs</c>) added by #628.
    /// </summary>
    const int ExpectedFileCount = 709;

    /// <summary>
    /// <c>docs/…md</c> references discovered. Pinned deliberately.
    /// 357 = 358 - one <c>docs/LAYA.md</c> reference removed from
    /// <c>tests/Nivara.PerformanceTests/ApplyMaskProbe.cs</c> and one from
    /// <c>tests/Nivara.PerformanceTests/README.md</c>, plus one added in
    /// <c>docs/ACCELERATION.md</c>, by the #492 reconciliation.
    /// </summary>
    const int ExpectedDocumentReferenceCount = 358;

    /// <summary><c>File.cs:NN</c> citations discovered across the widened scope. Pinned deliberately.</summary>
    const int ExpectedCodeCitationCount = 126;

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

            Assert.That(DocReferenceExtractor.DocumentReferenceCount(), Is.EqualTo(ExpectedDocumentReferenceCount),
                "the markdown-reference count moved; update this number deliberately");

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
            var raw = DocReferenceExtractor.DocumentFaults(file, tolerate: false)
                .Where(f => f.Reference == target);

            Assert.That(raw, Is.Not.Empty,
                $"{file} must actually name {target}, or the allowlist entry proves nothing");

            Assert.That(DocReferenceExtractor.DocumentFaults(file).Where(f => f.Reference == target), Is.Empty,
                $"{file} -> {target} is tolerated history and must not surface as a fault");
        }
    }
}
