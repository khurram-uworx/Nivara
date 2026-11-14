using System.Text.RegularExpressions;

namespace Nivara.Tests.Docs;

/// <summary>
/// The wider half of the reference gate. <see cref="DocCitationExtractor"/> reads the repository
/// root plus <c>docs/</c> and recognises only <c>File.cs:NN</c>; that scope let two recurrences
/// survive: #625 (a deleted <c>docs/BERT-GPU.md</c> still cited from <c>samples/</c> and
/// <c>tests/</c>) and #549 (a deleted <c>docs/TODO.md</c> cited from probes). This extractor reads
/// the product trees (<c>src/</c>, <c>samples/</c>, <c>tests/</c>, <c>docs/</c>, and the root) and
/// recognises both forms: a <c>File.cs:NN</c> citation and a reference to a markdown document
/// (<c>docs/…md</c>, optionally <c>../</c>-prefixed, backticked or inline). A reference that resolves
/// to no file is a <see cref="DocCitationFaultKind.MissingDocument"/> fault.
/// </summary>
static partial class DocReferenceExtractor
{
    /// <summary>Directory names never descended into, so build output and vendored trees stay out.</summary>
    static readonly string[] ExcludedSegments =
        ["bin", "obj", "TestResults", ".git", ".kilo", ".opencode", ".agents", "node_modules"];

    /// <summary>
    /// Paths excluded from the scan. <c>tests/Nivara.Tests/Docs/</c> is this gate's own fixtures and
    /// negative controls, which contain deliberately broken references; <c>docs/TODO.md</c> is the
    /// transient iterative-work plan document, kept out for the same reason the snippet and citation
    /// gates keep it out; <c>samples/data/</c> is the fixtures and downloaded-model cache - the
    /// repository tracks data there (no <c>.md</c> or <c>.cs</c>) and <c>.gitignore</c> treats the
    /// model directories as absent, so a machine that has run a sample carries an imported model's own
    /// <c>README.md</c> inside one. The scan walks the filesystem, so reading that tree would make the
    /// pinned file count depend on whether the machine happens to have downloaded a model - exactly how
    /// this gate first failed in CI (686 there, 687 on a machine that had run Laya).
    /// <c>tests/Nivara.Tests/Samples/DocCorpusTests.cs</c> is excluded because it contains fixture
    /// paths (<c>docs/GUIDE.md</c>, <c>docs/blog/post.md</c>, <c>docs/plan/ROADMAP.md</c>,
    /// <c>docs/plans/EVAL.md</c>) that are not real document references; counting them would inflate
    /// the pin with strings that never resolve.
    /// </summary>
    static readonly string[] ExcludedPathPrefixes = ["tests/Nivara.Tests/Docs/", "tests/Nivara.Tests/Samples/DocCorpusTests.cs", "samples/data/"];

    /// <summary>Documents excluded by exact path rather than by directory prefix.</summary>
    static readonly string[] ExcludedDocuments = ["docs/TODO.md"];

    /// <summary>
    /// References to documents that no longer exist but are named on purpose, because the sentence
    /// records the removal. #549's tolerance note: a check that does not tolerate these fires on
    /// exactly the prose doing its job. Every entry is a real historical mention - a decision log
    /// (the SHAKESPEARE plans), a note that a plan file was deleted (TODO.md), a changelog entry
    /// naming the review it fixes, or a roadmap line recording that a doc was retired. The pin only
    /// moves when a new one is added deliberately; a test asserts every entry is still present in its
    /// file, so an entry whose reference is fixed or removed fails rather than rotting in place.
    /// Entries whose target is the scan-excluded plan document cannot be observed while that document
    /// exists — the reference then resolves, so the counterfactual has nothing to show — and the
    /// presence test skips them mid-plan; CI, which runs without a plan, checks every entry.
    /// </summary>
    static readonly HashSet<(string File, string Target)> Tolerated =
    [

        ("docs/ROADMAP-SUGGESTION.md", "docs/TODO.md"),
        ("docs/adr/005-snippet-gate-authoring-contract.md", "docs/TODO.md"),
        ("docs/plan/ARROW-REVIEW.md", "docs/IDEA.md"),
        ("docs/plan/ARROW-ROADMAP.md", "docs/IDEA.md"),
        ("docs/plan/POLARS-REVIEW.md", "docs/IDEA.md"),
        ("docs/plan/POLARS-ROADMAP.md", "docs/IDEA.md"),
        ("samples/SHAKESPEARE.md", "docs/SHAKESPEARE.md"),
        ("samples/SHAKESPEARE.md", "docs/SHAKESPEARE2.md"),
    ];

    /// <summary>Every unresolvable reference in one document, both reference forms.</summary>
    internal static IReadOnlyList<DocCitationFault> Faults(string documentPath) =>
        [.. DocCitationExtractor.Faults(documentPath), .. DocumentFaults(documentPath)];

    /// <summary>Every unresolvable reference across the widened scan scope.</summary>
    internal static IReadOnlyList<DocCitationFault> AllFaults()
    {
        var faults = new List<DocCitationFault>();

        foreach (var document in ScanFiles())
            faults.AddRange(Faults(document));

        return faults;
    }

    /// <summary>Markdown documents and C# files the widened scan reads, for the coverage assertion.</summary>
    internal static IReadOnlyList<string> ScanFiles()
    {
        var root = DocSnippetExtractor.RepoRoot;
        var files = new List<string>();

        // C# files: read everywhere so File.cs:NN citations resolve (unchanged scope).
        Collect(root, recursive: false, files, static relative => Path.GetExtension(relative) == ".cs");
        foreach (var subtree in new[] { "docs", "src", "samples", "tests" })
        {
            var directory = Path.Combine(root, subtree);
            if (Directory.Exists(directory))
                Collect(directory, recursive: true, files, static relative => Path.GetExtension(relative) == ".cs");
        }

        // Markdown reference discovery (issue #704/#1178): top-level docs/*.md plus docs/adr/,
        // repo-root *.md (including root docs whose root-relative links are now checked), and
        // README.md/AGENTS.md anywhere. Subfolders of docs/ other than adr/ (blog/, research/,
        // plan/, ...) remain out of scope for the pin.
        Collect(root, recursive: false, files, static relative => Path.GetExtension(relative) == ".md");
        var docsDir = Path.Combine(root, "docs");
        if (Directory.Exists(docsDir))
        {
            Collect(docsDir, recursive: false, files, static relative => Path.GetExtension(relative) == ".md");
            var adrDir = Path.Combine(docsDir, "adr");
            if (Directory.Exists(adrDir))
                Collect(adrDir, recursive: true, files, static relative => Path.GetExtension(relative) == ".md");
        }
        Collect(root, recursive: true, files, static relative =>
            Path.GetFileName(relative) is "README.md" or "AGENTS.md");

        return files
            .Distinct(StringComparer.Ordinal)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();
    }

    static readonly Lazy<Dictionary<string, string[]>> MarkdownDocsByBaseName = new(IndexMarkdownDocsByBaseName);

    static Dictionary<string, string[]> IndexMarkdownDocsByBaseName()
    {
        var root = DocSnippetExtractor.RepoRoot;
        var files = new List<string>();

        foreach (var file in Directory.EnumerateFiles(root, "*.md", SearchOption.TopDirectoryOnly))
            files.Add(NormalizeRelative(root, file));

        foreach (var subtree in new[] { "docs", "samples", "src", "tests" })
        {
            var directory = Path.Combine(root, subtree);
            if (Directory.Exists(directory))
                CollectAllMarkdown(directory, files);
        }

        return files
            .Distinct(StringComparer.Ordinal)
            .GroupBy(Path.GetFileName, StringComparer.Ordinal)
            .ToDictionary(g => g.Key!, g => g.OrderBy(x => x, StringComparer.Ordinal).ToArray(), StringComparer.Ordinal);
    }

    static void CollectAllMarkdown(string directory, List<string> files)
    {
        foreach (var file in Directory.EnumerateFiles(directory, "*.md", SearchOption.TopDirectoryOnly))
        {
            var relative = NormalizeRelative(DocSnippetExtractor.RepoRoot, file);
            if (!ExcludedSegments.Contains(Path.GetFileName(Path.GetDirectoryName(file) ?? string.Empty), StringComparer.Ordinal))
                files.Add(relative);
        }

        foreach (var child in Directory.EnumerateDirectories(directory))
        {
            if (!ExcludedSegments.Contains(Path.GetFileName(child), StringComparer.Ordinal))
                CollectAllMarkdown(child, files);
        }
    }

    static string NormalizeRelative(string root, string path) =>
        Path.GetRelativePath(root, path).Replace('\\', '/');

    internal static int FileCount() => ScanFiles().Count;

    /// <summary>Markdown-document references discovered across the widened scope, for the coverage assertion.</summary>
    internal static int DocumentReferenceCount() =>
        ScanFiles().Sum(path => ScanDocumentRefs(path).Count);

    /// <summary><c>File.cs:NN</c> citations discovered across the widened scope, for the coverage assertion.</summary>
    internal static int CodeCitationCount() =>
        ScanFiles().Sum(DocCitationExtractor.CitationCountIn);

    /// <summary>Document references that resolve to no file, minus the tolerated historical mentions.</summary>
    internal static IReadOnlyList<DocCitationFault> DocumentFaults(string documentPath) =>
        DocumentFaults(documentPath, tolerate: true);

    /// <summary>The allowlist itself, so a test can prove every entry is both present and suppressed.</summary>
    internal static IReadOnlyCollection<(string File, string Target)> ToleratedReferences => Tolerated;

    /// <summary>
    /// The same faults with the allowlist switchable, so a test can observe the counterfactual — that
    /// a tolerated reference is genuinely present and only the allowlist keeps it green. That is the
    /// only way to tell a working allowlist from one that is merely declared.
    /// </summary>
    internal static IReadOnlyList<DocCitationFault> DocumentFaults(string documentPath, bool tolerate)
    {
        var faults = new List<DocCitationFault>();

        foreach (var (line, target) in ScanDocumentRefs(documentPath))
        {
            if (Resolves(documentPath, target))
                continue;

            if (tolerate && Tolerated.Contains((documentPath, target)))
                continue;

            faults.Add(new DocCitationFault(documentPath, line, target,
                DocCitationFaultKind.MissingDocument, $"no markdown document at {target}"));
        }

        return faults;
    }

    static bool Resolves(string documentPath, string target)
    {
        if (target.Contains('/'))
            return File.Exists(Absolute(target));

        var dir = Path.GetDirectoryName(documentPath)?.Replace('\\', '/');
        if (!string.IsNullOrEmpty(dir))
        {
            var sibling = $"{dir}/{target}";
            if (File.Exists(Absolute(sibling)))
                return true;
        }

        if (File.Exists(Absolute(target)))
            return true;

        if (File.Exists(Absolute($"docs/{target}")))
            return true;

        if (MarkdownDocsByBaseName.Value.TryGetValue(Path.GetFileName(target), out var matches) && matches.Length == 1)
            return File.Exists(Absolute(matches[0]));

        return false;
    }

    static void Collect(string directory, bool recursive, List<string> files, Func<string, bool> includeFile)
    {
        foreach (var file in Directory.EnumerateFiles(directory, "*.*", SearchOption.TopDirectoryOnly))
        {
            var relative = Path.GetRelativePath(DocSnippetExtractor.RepoRoot, file).Replace('\\', '/');

            if (ExcludedDocuments.Contains(relative, StringComparer.Ordinal))
                continue;

            if (ExcludedPathPrefixes.Any(prefix => relative.StartsWith(prefix, StringComparison.Ordinal)))
                continue;

            if (!includeFile(relative))
                continue;

            files.Add(relative);
        }

        if (!recursive)
            return;

        foreach (var child in Directory.EnumerateDirectories(directory))
        {
            if (!ExcludedSegments.Contains(Path.GetFileName(child), StringComparer.Ordinal))
                Collect(child, recursive: true, files, includeFile);
        }
    }

    static IReadOnlyList<(int Line, string Target)> ScanDocumentRefs(string documentPath)
    {
        var lines = File.ReadAllLines(Absolute(documentPath));
        var found = new List<(int, string)>();

        for (var i = 0; i < lines.Length; i++)
        {
            foreach (Match match in DocumentPattern().Matches(lines[i]))
                found.Add((i + 1, match.Groups["doc"].Value));
        }

        return found;
    }

    static string Absolute(string repoRelativePath) =>
        Path.Combine(DocSnippetExtractor.RepoRoot, repoRelativePath.Replace('/', Path.DirectorySeparatorChar));

    /// <summary>
    /// A markdown-document reference. Matches either a <c>docs/...</c> path (one or more segments
    /// ending in <c>.md</c>) or a markdown document name used as a link target/backticked path.
    /// Bare names are resolved relative to the containing document, then repo root, then <c>docs/</c>,
    /// then by unique basename. An optional <c>../</c> prefix is consumed by the lookbehind boundary,
    /// so a relative <c>../../docs/x.md</c> resolves against the repository root like a bare one.
    /// </summary>
    [GeneratedRegex(@"(?<![A-Za-z0-9_])(?<doc>(?:docs/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.md|(?:(?<=\]\()|(?<=`))[A-Za-z0-9_.-]+\.md))")]
    private static partial Regex DocumentPattern();
}
