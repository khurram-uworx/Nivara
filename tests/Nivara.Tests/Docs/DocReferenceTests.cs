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

    [Test]
    [Category("ReleaseReadiness")]
    public void EveryReferenceInTheProductTrees_Resolves()
    {
        var faults = DocReferenceExtractor.AllFaults();

        Assert.That(faults, Is.Empty, Environment.NewLine + string.Join(Environment.NewLine, faults.Select(f => f.ToString())));
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
