using NUnit.Framework;

namespace Nivara.Tests.Docs;

/// <summary>
/// Checks that every <c>File.cs:NN</c> citation in the documentation resolves to exactly one
/// source file and a line inside it. Motivation is #516: a doc stated a memory budget the code
/// did not use, and nothing detected it. The snippet gate added by #515 catches drift in code
/// shown inline; this catches drift in the file references pointing at that code.
/// </summary>
[TestFixture]
public class DocCitationTests
{
    const string NegativeControl = "tests/Nivara.Tests/Docs/Fixtures/BrokenCitations.md";

    /// <summary>Count of citations found repository-wide. Pinned deliberately: see the coverage test.</summary>
    const int ExpectedCitationCount = 105;

    /// <summary>Count of documents carrying at least one citation. Pinned deliberately.</summary>
    const int ExpectedCitedDocumentCount = 16;

    /// <summary>Count of documents in scope. Pinned so a shrinking scan cannot look like a pass.</summary>
    const int ExpectedDocumentCount = 46;

    [Test]
    public void EveryCitationInTheDocumentation_Resolves()
    {
        var faults = DocCitationExtractor.AllFaults();

        Assert.That(faults, Is.Empty, Environment.NewLine + string.Join(Environment.NewLine, faults.Select(f => f.ToString())));
    }

    [Test]
    public void Coverage_IsAccountedForAcrossTheWholeRepository()
    {
        Assert.Multiple(() =>
        {
            Assert.That(DocCitationExtractor.CitationCount(), Is.EqualTo(ExpectedCitationCount),
                "the repository-wide citation count moved; update this number deliberately. Unlike the "
                + "snippet gate there is no opt-out and no ungated list — every citation in every scanned "
                + "document is checked, so a drop here is reduced coverage, not a neutral event");

            Assert.That(DocSnippetExtractor.MarkdownFiles(), Has.Count.EqualTo(ExpectedDocumentCount),
                "the document scope moved; update this number deliberately. A shrinking scope means "
                + "documents stopped being checked, which is reduced coverage, not a neutral event");

            Assert.That(DocCitationExtractor.CitedDocumentCount(), Is.EqualTo(ExpectedCitedDocumentCount),
                "the number of documents carrying citations moved; update this number deliberately");
        });
    }

    [Test]
    public void ScanScope_ExcludesTheTransientPlanDocument()
    {
        Assert.That(DocSnippetExtractor.MarkdownFiles(), Does.Not.Contain("docs/TODO.md"),
            "the gate reuses the snippet gate's document scope, which leaves the transient plan out");
    }

    [Test]
    public void RangeCitations_AreJudgedByTheirLastLine()
    {
        // The negative control carries 1-100000: the first line is real, the last is not. A gate that
        // only checked the start would pass it, which is exactly how a 1366-1388 range survived in a
        // file that is now 187 lines.
        var faults = DocCitationExtractor.Faults(NegativeControl);
        var range = faults.Single(f => f.Reference.EndsWith("NivaraExecutionContext.cs:1-100000", StringComparison.Ordinal));

        Assert.That(range.Kind, Is.EqualTo(DocCitationFaultKind.PastEndOfFile));
        Assert.That(range.Detail, Does.Contain("NivaraExecutionContext.cs"));
    }

    [Test]
    public void AmbiguousBasenames_AreRejectedRatherThanGuessed()
    {
        var faults = DocCitationExtractor.Faults(NegativeControl);
        var ambiguous = faults.Single(f => f.Reference == "Program.cs:1");

        Assert.That(ambiguous.Kind, Is.EqualTo(DocCitationFaultKind.AmbiguousFile));
        Assert.That(ambiguous.Detail, Does.Contain("qualify the path"));
    }

    [Test]
    public void MissingFiles_AreReportedByQualifiedAndBareFormAlike()
    {
        var faults = DocCitationExtractor.Faults(NegativeControl);

        Assert.Multiple(() =>
        {
            Assert.That(faults.Where(f => f.Kind == DocCitationFaultKind.MissingFile).Select(f => f.Reference),
                Is.EquivalentTo(new[]
                {
                    "src/Nivara/Storage/NoSuchFileDoesNotExist.cs:12",
                    "NoSuchFileAnywhere.cs:3",
                }));
        });
    }

    [Test]
    public void NegativeControl_TheGateStillRejectsBrokenCitations()
    {
        var faults = DocCitationExtractor.Faults(NegativeControl);

        Assert.That(faults, Has.Count.EqualTo(5),
            "the gate stopped rejecting its negative control, so a green run no longer means anything. "
            + Environment.NewLine + string.Join(Environment.NewLine, faults.Select(f => f.ToString())));
    }

    [Test]
    public void NegativeControl_StillAcceptsTheResolvableCitation()
    {
        var faults = DocCitationExtractor.Faults(NegativeControl);

        Assert.That(faults.Select(f => f.Reference),
            Does.Not.Contain("src/Nivara/Execution/NivaraExecutionContext.cs:1"),
            "the negative control rejects everything, including a citation that is correct; that would "
            + "prove nothing about discriminating real faults from valid references");
    }
}
