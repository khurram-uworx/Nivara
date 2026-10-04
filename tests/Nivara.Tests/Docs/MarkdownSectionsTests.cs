using NUnit.Framework;

namespace Nivara.Tests.Docs;

/// <summary>
/// Tests for the section reader the context-file gates are built on. Every case is driven by
/// lines supplied here rather than by a real document, so the reader can be shown rejecting
/// input instead of only ever being shown accepting it.
/// </summary>
[TestFixture]
public class MarkdownSectionsTests
{
    [Test]
    public void AHeadingIsOneToSixHashesFollowedByWhitespace()
    {
        var sections = MarkdownSections.ExtractLines(
        [
            "# Title",
            "## Section",
            "#### Subsection",
            "",
            "#noSpaceBefore is text",
            "####### tooDeep is text",
            "#",
        ]);

        Assert.Multiple(() =>
        {
            Assert.That(sections.Select(s => s.Heading),
                Is.EqualTo(new[] { "Title", "Section", "Subsection" }),
                "a bare '#' carries no heading text, a '#' with no space after it is text, and more "
                + "than six hashes is not a heading either. Note '# text' IS a heading, which is why "
                + "this line is the bare '#' and not a sentence");

            Assert.That(sections.Select(s => s.Level), Is.EqualTo(new[] { 1, 2, 4 }));
        });
    }

    [Test]
    public void ASectionRunsUntilTheNextHeadingAtItsOwnLevel()
    {
        var sections = MarkdownSections.ExtractLines(
        [
            "# Top",
            "top body",
            "## Left",
            "left body",
            "### Nested",
            "nested body",
            "## Right",
            "right body",
            "# Second Top",
            "second top body",
        ]);

        var left = sections.Single(s => s.Heading == "Left");

        Assert.Multiple(() =>
        {
            Assert.That(left.Body, Is.EqualTo(new[] { "left body", "### Nested", "nested body" }),
                "a deeper heading is inside the section, not the end of it");

            Assert.That(left.EndLine, Is.EqualTo(6), "the section ends at the line before the next heading of its level");
            Assert.That(sections.Single(s => s.Heading == "Top").Body,
                Is.EqualTo(new[] { "top body", "## Left", "left body", "### Nested", "nested body", "## Right", "right body" }));
        });
    }

    [Test]
    public void SubsectionsAttachToTheNearestEnclosingSection()
    {
        var sections = MarkdownSections.ExtractLines(
        [
            "# Top",
            "## Outer",
            "### Inner",
            "## Sibling",
            "#### Deep",
        ]);

        var top = sections.Single(s => s.Heading == "Top");
        var outer = sections.Single(s => s.Heading == "Outer");
        var inner = sections.Single(s => s.Heading == "Inner");

        Assert.Multiple(() =>
        {
            Assert.That(top.Children.Select(c => c.Heading), Is.EqualTo(new[] { "Outer", "Sibling" }));
            Assert.That(outer.Children.Select(c => c.Heading), Is.EqualTo(new[] { "Inner" }),
                "'Deep' is four levels down, so its nearest enclosing section is Sibling, not Top");
            Assert.That(sections.Single(s => s.Heading == "Sibling").Children.Select(c => c.Heading),
                Is.EqualTo(new[] { "Deep" }));
            Assert.That(inner.Children, Is.Empty);
        });
    }

    [Test]
    public void ABulletNeedsADashFollowedByASpace()
    {
        var section = MarkdownSections.ExtractLines(["# S", "- first", "-second", "  - nested", "a - b"]).Single();

        Assert.That(section.Bullets(), Is.EqualTo(new[] { "first", "nested" }),
            "a dash with no space is hyphenated prose, and a dash mid-line is not a list marker");
    }

    [Test]
    public void ACodeSpanNeedsTwoBackticks()
    {
        Assert.Multiple(() =>
        {
            Assert.That(MarkdownSections.FirstCodeSpan("`docs/adr/001-a.md`: decides a thing."),
                Is.EqualTo("docs/adr/001-a.md"), "only the first span is returned, so trailing spans stay in the summary");
            Assert.That(MarkdownSections.FirstCodeSpan("`<!-- gate -->` keys are mode, locals"),
                Is.EqualTo("<!-- gate -->"));
            Assert.That(MarkdownSections.FirstCodeSpan("an unclosed ` backtick"), Is.Null);
            Assert.That(MarkdownSections.FirstCodeSpan("no backtick at all"), Is.Null);
        });
    }

    [Test]
    public void AnOrdinalIsTheDigitsBeforeTheFirstDash()
    {
        Assert.Multiple(() =>
        {
            Assert.That(MarkdownSections.OrdinalOf("005-snippet-gate-authoring-contract.md"), Is.EqualTo("005"));
            Assert.That(MarkdownSections.OrdinalOf("README.md"), Is.Null, "no leading number, so no ordinal");
            Assert.That(MarkdownSections.OrdinalOf("adr-001-x.md"), Is.Null, "leading characters are not digits");
            Assert.That(MarkdownSections.OrdinalOf("no-dash.md"), Is.Null);
        });
    }

    [Test]
    public void RequireReportsWhatIsPresentWhenTheHeadingIsMissing()
    {
        var error = Assert.Throws<InvalidOperationException>(
            () => MarkdownSections.Require("AGENTS.md", "A Section That Does Not Exist"));

        Assert.That(error!.Message, Does.Contain("no section has that heading").And.Contain("Architectural Decisions (ADRs)"),
            "a failure that does not list what is available leaves the reader to go looking for it");
    }

    [Test]
    public void RequireFindsTheRealAdrSectionOfTheAgentIndex()
    {
        var section = MarkdownSections.Require("AGENTS.md", "Architectural Decisions (ADRs)");

        Assert.Multiple(() =>
        {
            Assert.That(section.Bullets(), Is.Not.Empty);
            Assert.That(section.Text, Does.Contain("ADR-001"), "sanity: the section read is the one asked for");
        });
    }
}