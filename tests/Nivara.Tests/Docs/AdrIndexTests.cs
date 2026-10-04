using NUnit.Framework;

namespace Nivara.Tests.Docs;

/// <param name="Label">The record's ordinal as the index writes it, e.g. <c>ADR-001</c>.</param>
/// <param name="Path">Repo-relative path to the ADR file.</param>
/// <param name="Summary">The sentence after the path, saying what the decision constrains.</param>
sealed record AdrIndexEntry(string Label, string Path, string Summary);

/// <param name="Unlisted">ADR files the index never mentions.</param>
/// <param name="Dangling">Index entries whose file is gone.</param>
/// <param name="Duplicate">Index entries naming the same file more than once.</param>
/// <param name="Summaryless">Index entries with nothing after the path.</param>
sealed record AdrIndexDrift(
    string[] Unlisted,
    string[] Dangling,
    string[] Duplicate,
    string[] Summaryless);

/// <summary>
/// The ADR index in <c>AGENTS.md</c>, read as a section and checked against the folder it
/// summarises.
/// </summary>
/// <remarks>
/// Kept beside its test because it exists only to serve that test. The reusable part — reading a
/// document's section structure — is <see cref="MarkdownSections"/>.
/// </remarks>
static class AdrIndex
{
    internal const string AdrDirectory = "docs/adr";
    internal const string AgentsIndexPath = "AGENTS.md";
    internal const string SectionHeading = "Architectural Decisions (ADRs)";

    /// <summary>The ADR files on disk, in folder order.</summary>
    internal static IReadOnlyList<string> ReadFileNames()
    {
        var directory = Path.Combine(DocSnippetExtractor.RepoRoot, AdrDirectory);

        return Directory.Exists(directory)
            ? [.. Directory.GetFiles(directory, "*.md").Select(Path.GetFileName).Select(n => n!)]
            : [];
    }

    /// <summary>The index entries, in the order the section lists them.</summary>
    internal static IReadOnlyList<AdrIndexEntry> ReadEntries()
    {
        var section = MarkdownSections.Require(AgentsIndexPath, SectionHeading);
        var entries = new List<AdrIndexEntry>();

        foreach (var bullet in section.Bullets())
            if (TryParseEntry(bullet) is { } entry)
                entries.Add(entry);

        return entries;
    }

    /// <summary>
    /// The named ADR section of the agent index, so a test can assert its own contract on it.
    /// </summary>
    internal static MarkdownSection ReadSection() => MarkdownSections.Require(AgentsIndexPath, SectionHeading);

    /// <summary>
    /// Reads one index bullet, or null when the bullet carries no code span at all.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A bullet reads <c>**ADR-001** (<c>docs/adr/001-name.md</c>): what it decides.</c> — the
    /// path is the text inside the backticks, the label is what precedes them, and the summary is
    /// what follows. The trailing <c>):</c> and the label's bold markers are stripped so the two
    /// halves compare equal to what they are meant to be.
    /// </para>
    /// <para>
    /// The bare <c>See docs/adr/</c> pointer is *not* kept out by returning null — it does have
    /// backticks, so it would parse as a path of <c>docs/adr/</c>. It is kept out by not being a
    /// list item. Were it ever turned into one, it would surface as a dangling entry naming
    /// <c>adr/</c>, which fails loudly rather than silently.
    /// </para>
    /// </remarks>
    internal static AdrIndexEntry? TryParseEntry(string bullet)
    {
        var path = MarkdownSections.FirstCodeSpan(bullet);
        if (path is null)
            return null;

        var open = bullet.IndexOf('`');
        var close = bullet.IndexOf('`', open + 1);

        var label = bullet[..open].Trim().TrimEnd('(').Trim().Trim('*').Trim();
        var summary = bullet[(close + 1)..].Trim().TrimStart(')', ':').Trim();

        return new AdrIndexEntry(label, path, summary);
    }

    /// <summary>How an index and a folder disagree. Both sides are compared, not just one.</summary>
    internal static AdrIndexDrift Compare(IReadOnlyList<string> adrFileNames, IReadOnlyList<AdrIndexEntry> entries)
    {
        var referenced = entries.Select(e => Path.GetFileName(e.Path)).ToArray();

        return new AdrIndexDrift(
            Unlisted: [.. adrFileNames.Except(referenced, StringComparer.Ordinal).Order(StringComparer.Ordinal)],
            Dangling: [.. referenced.Except(adrFileNames, StringComparer.Ordinal).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal)],
            Duplicate: [.. referenced.GroupBy(r => r, StringComparer.Ordinal)
                                        .Where(g => g.Count() > 1)
                                        .Select(g => g.Key)
                                        .Order(StringComparer.Ordinal)],
            Summaryless: [.. entries.Where(e => e.Summary.Length == 0)
                                    .Select(e => e.Label)
                                    .Order(StringComparer.Ordinal)]);
    }

    /// <summary>The label the index must use for a file, e.g. <c>001-anything.md</c> is <c>ADR-001</c>.</summary>
    internal static string? ExpectedLabelFor(string fileName) =>
        MarkdownSections.OrdinalOf(fileName) is { } ordinal ? $"ADR-{ordinal}" : null;
}

/// <summary>
/// Guards against an ADR landing in <c>docs/adr/</c> without being listed in the <c>AGENTS.md</c>
/// index, which is the one place an agent is told to look.
/// </summary>
/// <remarks>
/// <para>
/// ADR-004 was recorded in <c>660fc229</c> (2026-08-15) and its index line did not appear until
/// <c>5563c412</c> (2026-09-27). Six weeks of the index understating what the folder held, with
/// nothing to catch it, because an index maintained by remembering is not a gate. This makes the
/// omission a test failure instead.
/// </para>
/// <para>
/// <b>Coverage.</b> The <c>Architectural Decisions (ADRs)</c> section of <c>AGENTS.md</c> against
/// the files in <c>docs/adr/</c>, both directions. <b>Not covered:</b> prose links to an ADR from
/// another document, and the bare <c>See docs/adr/</c> pointer line, which is a pointer rather
/// than a per-record reference and so is not an entry.
/// </para>
/// </remarks>
[TestFixture]
public class AdrIndexTests
{
    [Test]
    public void TheAgentsIndexHasOneAdrSectionAndItListsRecords()
    {
        var section = AdrIndex.ReadSection();

        Assert.Multiple(() =>
        {
            Assert.That(section.Level, Is.EqualTo(2),
                "the index is a subsection of AGENTS.md, so renaming or demoting it changes what loads");
            Assert.That(section.Bullets(), Is.Not.Empty,
                $"'{AdrIndex.SectionHeading}' exists but lists nothing, so an agent following it is told nothing");
        });
    }

    [Test]
    public void EveryAdrIsListedInTheAgentsIndex()
    {
        var drift = AdrIndex.Compare(AdrIndex.ReadFileNames(), AdrIndex.ReadEntries());

        Assert.Multiple(() =>
        {
            Assert.That(drift.Unlisted, Is.Empty,
                "these ADRs exist but the index does not mention them, so the one place an agent is "
                + "told to look understates the folder. Add a one-line entry: "
                + string.Join(", ", drift.Unlisted));

            Assert.That(drift.Dangling, Is.Empty,
                "the index points at these files but none exists, so a reader following it is sent "
                + "after a decision that was withdrawn: " + string.Join(", ", drift.Dangling));
        });
    }

    [Test]
    public void NoAdrIsListedTwice()
    {
        var drift = AdrIndex.Compare(AdrIndex.ReadFileNames(), AdrIndex.ReadEntries());

        Assert.That(drift.Duplicate, Is.Empty,
            "the same record is listed more than once, which reads as two decisions and makes the "
            + "index disagree with the folder's count: " + string.Join(", ", drift.Duplicate));
    }

    [Test]
    public void EveryIndexEntryStatesWhatTheDecisionConstrains()
    {
        var summaryless = AdrIndex.ReadEntries().Where(e => e.Summary.Length == 0).ToArray();

        Assert.That(summaryless, Is.Empty,
            "an index line with a path but no summary is a bare link. Its purpose is to let an agent "
            + "decide relevance without opening the file, so the summary is the whole value: "
            + string.Join(", ", summaryless.Select(e => e.Label)));
    }

    [Test]
    public void EveryIndexLabelAgreesWithTheFileItNames()
    {
        var mismatched = AdrIndex.ReadEntries()
            .Where(e => AdrIndex.ExpectedLabelFor(e.Path) is { } expected && expected != e.Label)
            .Select(e => $"{e.Label} -> {e.Path}")
            .ToArray();

        Assert.That(mismatched, Is.Empty,
            "the label and the file name disagree, so the index reads as recording a decision the "
            + "folder does not hold: " + string.Join(", ", mismatched));
    }

    [Test]
    public void NegativeControl_TheScanReportsAnAdrMissingFromTheIndex()
    {
        var files = new[] { "001-real.md", "002-real.md" };

        // Precondition: the index genuinely omits one of the two files the folder declares.
        // Without it the assertion below could pass because the comparison found nothing to do.
        var drift = AdrIndex.Compare(files, [new AdrIndexEntry("ADR-001", "docs/adr/001-real.md", "decides a thing")]);

        Assert.Multiple(() =>
        {
            Assert.That(drift.Unlisted, Is.EqualTo(new[] { "002-real.md" }));
            Assert.That(drift.Dangling, Is.Empty, "this control is about an omission, so the reverse direction stays clean");
        });
    }

    [Test]
    public void NegativeControl_TheScanReportsAnEntryWithNoFile()
    {
        var drift = AdrIndex.Compare(
            ["001-real.md"],
            [new AdrIndexEntry("ADR-001", "docs/adr/001-real.md", "decides a thing"),
             new AdrIndexEntry("ADR-009", "docs/adr/009-withdrawn.md", "was withdrawn")]);

        Assert.Multiple(() =>
        {
            Assert.That(drift.Unlisted, Is.Empty);
            Assert.That(drift.Dangling, Is.EqualTo(new[] { "009-withdrawn.md" }),
                "a dangling entry sends a reader after a withdrawn decision, which an "
                + "omission-only check would never notice");
        });
    }

    [Test]
    public void NegativeControl_TheScanReportsARepeatedEntry()
    {
        var drift = AdrIndex.Compare(
            ["001-real.md"],
            [new AdrIndexEntry("ADR-001", "docs/adr/001-real.md", "decides a thing"),
             new AdrIndexEntry("ADR-001", "docs/adr/001-real.md", "decides a thing")]);

        Assert.That(drift.Duplicate, Is.EqualTo(new[] { "001-real.md" }));
    }

    [Test]
    public void NegativeControl_TheScanAcceptsAFullyListedSet()
    {
        var files = new[] { "001-a.md", "002-b.md", "003-c.md" };
        var entries = files.Select(f => new AdrIndexEntry(AdrIndex.ExpectedLabelFor(f)!, $"docs/adr/{f}", "decides a thing")).ToArray();

        var drift = AdrIndex.Compare(files, entries);

        Assert.Multiple(() =>
        {
            Assert.That(drift.Unlisted, Is.Empty);
            Assert.That(drift.Dangling, Is.Empty);
            Assert.That(drift.Duplicate, Is.Empty);
            Assert.That(drift.Summaryless, Is.Empty);
        });
    }

    [Test]
    public void NegativeControl_ABulletWithoutAPathIsNotAnEntry()
    {
        // The bare 'See docs/adr/' pointer is protected by not being a bullet, not by its shape.
        // Were it ever turned into one, it would parse as an entry and the dangling check above
        // would name 'adr/' — so the safety net is covered either way.
        Assert.That(AdrIndex.TryParseEntry("a plain bullet with no path"), Is.Null);

        var entry = AdrIndex.TryParseEntry("**ADR-007** (`docs/adr/007-name.md`): decides a thing.");

        Assert.That(entry, Is.EqualTo(new AdrIndexEntry("ADR-007", "docs/adr/007-name.md", "decides a thing.")));
    }
}