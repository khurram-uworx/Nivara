using NUnit.Framework;

namespace Nivara.Tests.Docs;

/// <summary>
/// Compiles every fenced ```csharp block in the gated documents. Reports how many blocks it
/// checked, how many it skipped and why, and asserts the three numbers sum to the total it
/// discovers by scanning the repository, so a green run cannot hide reduced coverage.
/// </summary>
[TestFixture]
public class DocumentationSnippetTests
{
    const string ExcludedSlug = "streamix-aspnetcore";

    static IReadOnlyList<DocBlock> GatedBlocks() =>
        [.. DocSnippetExtractor.GatedDocuments.SelectMany(DocSnippetExtractor.Extract)];

    [Test]
    public void GatedDocuments_ContainBlocksToGate()
    {
        Assert.That(GatedBlocks(), Is.Not.Empty, "the allowlist named no snippets to gate");
    }

    [Test]
    public void PreambleNames_ResolveInBothDirections()
    {
        var used = GatedBlocks()
            .Select(b => b.PreambleName)
            .Where(name => name is not null)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
        var known = DocSnippetExtractor.KnownPreambles.OrderBy(n => n, StringComparer.Ordinal).ToArray();

        Assert.That(used, Is.EqualTo(known),
            "every preamble name in a gate comment must be declared, and every declared name must be used");
    }

    [Test]
    public void ScanExclusions_AreTheSingleTransientPlanDocument()
    {
        Assert.That(DocSnippetExtractor.ScanExcludedDocuments, Is.EqualTo(new[] { "docs/TODO.md" }),
            "scan exclusions are a named list, never a glob; extending it needs a reason and a decision");
    }

    [Test]
    public void EveryGatedBlock_CompilesWithoutErrors()
    {
        var failures = new List<string>();

        foreach (var block in GatedBlocks().Where(b => !b.IsExcluded))
            failures.AddRange(DocSnippetCompiler.Compile(block).Select(e => e.ToString()));

        Assert.That(failures, Is.Empty,
            Environment.NewLine + string.Join(Environment.NewLine, failures));
    }

    [Test]
    public void Coverage_IsFullyAccountedForAcrossTheWholeRepository()
    {
        var all = DocSnippetExtractor.MarkdownFiles()
            .Select(DocSnippetExtractor.Extract)
            .SelectMany(b => b)
            .ToArray();

        var gated = GatedBlocks();
        var ungatedCount = all.Count(b => !DocSnippetExtractor.GatedDocuments.Contains(b.DocumentPath));
        var excludedCount = gated.Count(b => b.IsExcluded);

        Assert.Multiple(() =>
        {
            Assert.That(gated.Count - excludedCount, Is.EqualTo(104),
                "expected 70 GETTING-STARTED.md + 22 EXAMPLES.md + 9 docs/STREAMING.md "
                + "+ 3 docs/AGENT-CODE-EXAMPLES.md blocks");
            Assert.That(excludedCount, Is.EqualTo(2));
            Assert.That(ungatedCount, Is.EqualTo(141));
            Assert.That(all.Length, Is.EqualTo(247),
                "the repository-wide snippet count moved; update this number deliberately. The gate "
                + "covers 106 of these blocks, so the remaining 141 are unverified and a drop here is "
                + "reduced coverage, not a neutral event");
        });
    }

    [Test]
    public void ProposalDocuments_AreExcludedByNameNotByGlob()
    {
        var gated = DocSnippetExtractor.GatedDocuments;
        var ungated = DocSnippetExtractor.UngatedDocuments;

        Assert.Multiple(() =>
        {
            Assert.That(gated, Does.Not.Contain("docs/ACCELERATION.md"));
            Assert.That(gated, Does.Not.Contain("docs/ROADMAP-SUGGESTION.md"));
            Assert.That(ungated, Does.Contain("docs/ACCELERATION.md"),
                "docs/ACCELERATION.md is a design proposal whose snippets deliberately do not compile");
        });
    }

    [Test]
    public void Coverage_EverySnippetBearingDocumentIsClassified()
    {
        var known = DocSnippetExtractor.GatedDocuments
            .Concat(DocSnippetExtractor.UngatedDocuments)
            .ToHashSet(StringComparer.Ordinal);

        var discovered = DocSnippetExtractor.MarkdownFiles()
            .Select(DocSnippetExtractor.Extract)
            .Where(b => b.Count > 0)
            .Select(b => b[0].DocumentPath)
            .Distinct(StringComparer.Ordinal)
            .OrderBy(p => p, StringComparer.Ordinal)
            .ToArray();

        var unclassified = discovered.Where(p => !known.Contains(p)).ToArray();

        Assert.That(unclassified, Is.Empty,
            "these documents contain snippets but are neither gated nor listed as ungated: "
            + string.Join(", ", unclassified));

        Assert.That(known.Except(discovered), Is.Empty,
            "these documents are listed but no longer contain snippets: "
            + string.Join(", ", known.Except(discovered).OrderBy(p => p, StringComparer.Ordinal)));
    }

    [Test]
    public void Exclusions_AreListedExplicitlyWithAStableReason()
    {
        var excluded = GatedBlocks().Where(b => b.IsExcluded).ToArray();

        Assert.Multiple(() =>
        {
            Assert.That(excluded.Select(b => b.DocumentPath).Distinct(), Is.EqualTo(new[] { "docs/STREAMING.md" }));
            Assert.That(excluded.Select(b => b.ExcludeSlug).Distinct(), Is.EqualTo(new[] { ExcludedSlug }));
            Assert.That(excluded.Select(b => b.ExcludeReason), Has.All.Contain("ASP.NET Core"));
        });
    }

    [Test]
    public void ExcludedBlocks_AreTheOnesThatGenuinelyNeedAspNetCore()
    {
        foreach (var block in GatedBlocks().Where(b => b.IsExcluded))
            Assert.That(block.Code, Does.Contain("Streamix.AspNetCore").Or.Contain("ToSseAsync"));
    }

    [Test]
    public void GatedStreamixBlocks_OnlyDependOnTheStreamixPackageTheRepoReferences()
    {
        var gated = GatedBlocks()
            .Where(b => !b.IsExcluded && b.DocumentPath == "docs/STREAMING.md")
            .ToArray();

        Assert.That(gated.Select(b => b.Code), Has.All.Not.Contain("Streamix.AspNetCore"));
        Assert.That(gated, Has.Length.EqualTo(9));
    }

    [Test]
    public void NegativeControl_TheGateStillRejectsABrokenSnippet()
    {
        var fixture = DocSnippetExtractor.Extract("tests/Nivara.Tests/Docs/Fixtures/Broken.md").Single();
        var errors = DocSnippetCompiler.Compile(fixture);

        Assert.That(errors, Is.Not.Empty,
            "the gate compiled its negative control, so a green run no longer means anything");
    }
}