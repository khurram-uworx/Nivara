using NUnit.Framework;

namespace Nivara.Tests.Docs;

[TestFixture]
public class DocsCoverageTests
{
    [Test]
    public void Coverage_IsAccountedForAcrossTheRepositoryAndWidenedScope()
    {
        Assert.Multiple(() =>
        {
            Assert.That(DocCitationExtractor.CitationCount(), Is.EqualTo(DocsCoveragePins.ExpectedCitationCount),
                "the repository-wide citation count moved; update this number deliberately. Unlike the "
                + "snippet gate there is no opt-out and no ungated list - every citation in every scanned "
                + "document is checked, so a drop here is reduced coverage, not a neutral event");

            Assert.That(DocSnippetExtractor.MarkdownFiles(), Has.Count.EqualTo(DocsCoveragePins.ExpectedDocumentCount),
                "the document scope moved; update this number deliberately. A shrinking scope means "
                + "documents stopped being checked, which is reduced coverage, not a neutral event");

            Assert.That(DocCitationExtractor.CitedDocumentCount(), Is.EqualTo(DocsCoveragePins.ExpectedCitedDocumentCount),
                "the number of documents carrying citations moved; update this number deliberately");

            Assert.That(DocReferenceExtractor.CodeCitationCount(), Is.EqualTo(DocsCoveragePins.ExpectedCodeCitationCount),
                "the code-citation count across the widened scope moved; update this number deliberately");

            Assert.That(DocReferenceExtractor.DocumentReferenceCount(), Is.EqualTo(DocsCoveragePins.ExpectedDocumentReferenceCount),
                "the markdown-reference count moved; update this number deliberately");
        });
    }
}
