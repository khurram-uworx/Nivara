using System.Globalization;
using System.Text.RegularExpressions;

namespace Nivara.Tests.Docs;

/// <summary>Why a <c>File.cs:NN</c> citation could not be resolved.</summary>
enum DocCitationFaultKind
{
    /// <summary>The citation named a source file that does not exist.</summary>
    MissingFile,

    /// <summary>A bare citation named a basename that exists in more than one file.</summary>
    AmbiguousFile,

    /// <summary>The cited line, or the last line of the cited range, is past end-of-file.</summary>
    PastEndOfFile,
}

/// <param name="DocumentPath">Repo-relative path of the markdown file holding the citation.</param>
/// <param name="DocumentLine">One-based line within that file.</param>
/// <param name="Reference">The citation exactly as written, e.g. <c>src/Nivara/Foo.cs:12-18</c>.</param>
/// <param name="Kind">What went wrong.</param>
/// <param name="Detail">Candidates and line counts, so the message is actionable.</param>
sealed record DocCitationFault(
    string DocumentPath,
    int DocumentLine,
    string Reference,
    DocCitationFaultKind Kind,
    string Detail)
{
    /// <summary>Location used in assertion messages, e.g. <c>docs/STREAMING.md:121</c>.</summary>
    public string Location => $"{DocumentPath}:{DocumentLine}";

    public override string ToString() => $"{Location} cites {Reference}: {Detail} [{Kind}]";
}

/// <summary>
/// Extracts <c>File.cs:NN</c> citations from markdown and checks that each resolves to exactly one
/// source file and a line inside it. Complements <see cref="DocSnippetExtractor"/>, which checks
/// that fenced snippets compile: that one catches drift in the code shown inline, this one catches
/// drift in the file references pointing at it. Only path-and-line facts are checked — whether the
/// cited lines are the *right* lines for the surrounding claim is a reviewer's judgement, not a
/// machine-checkable one.
/// </summary>
static partial class DocCitationExtractor
{
    /// <summary>Directory names never descended into, keeping build output and vendored trees out.</summary>
    static readonly string[] ExcludedSegments =
        ["bin", "obj", ".git", ".kilo", ".opencode", "node_modules", "TestResults"];

    /// <summary>Repo-relative path to absolute path, for every C# file a citation may name.</summary>
    static readonly Dictionary<string, string> Sources = IndexSources();

    /// <summary>Basename to the repo-relative paths sharing it, so bare citations can be resolved.</summary>
    static readonly Dictionary<string, string[]> SourcesByBaseName = IndexByBaseName();

    /// <summary>Line counts, read at most once per file.</summary>
    static readonly Dictionary<string, int> LineCounts = new(StringComparer.Ordinal);

    /// <summary>Every unresolvable citation in one document.</summary>
    internal static IReadOnlyList<DocCitationFault> Faults(string documentPath)
    {
        var faults = new List<DocCitationFault>();

        foreach (var (line, reference, qualifier, fileName, lastLine) in Scan(documentPath))
        {
            string resolved;

            if (qualifier.Length > 0)
            {
                resolved = qualifier + fileName;

                if (!Sources.ContainsKey(resolved))
                {
                    faults.Add(new DocCitationFault(documentPath, line, reference,
                        DocCitationFaultKind.MissingFile, $"no source file at {resolved}"));
                    continue;
                }
            }
            else if (!SourcesByBaseName.TryGetValue(fileName, out var candidates))
            {
                faults.Add(new DocCitationFault(documentPath, line, reference,
                    DocCitationFaultKind.MissingFile, $"no source file named {fileName}"));
                continue;
            }
            else if (candidates.Length > 1)
            {
                var list = string.Join(", ", candidates.Order(StringComparer.Ordinal));
                faults.Add(new DocCitationFault(documentPath, line, reference,
                    DocCitationFaultKind.AmbiguousFile,
                    $"{fileName} exists in {candidates.Length} files ({list}); qualify the path"));
                continue;
            }
            else
            {
                resolved = candidates[0];
            }

            var lineCount = LineCount(resolved);

            if (lastLine > lineCount)
                faults.Add(new DocCitationFault(documentPath, line, reference,
                    DocCitationFaultKind.PastEndOfFile, $"{resolved} has {lineCount} lines"));
        }

        return faults;
    }

    /// <summary>Every unresolvable citation across the gate's document scope.</summary>
    internal static IReadOnlyList<DocCitationFault> AllFaults()
    {
        var faults = new List<DocCitationFault>();

        foreach (var document in DocSnippetExtractor.MarkdownFiles())
            faults.AddRange(Faults(document));

        return faults;
    }

    /// <summary>Total citations discovered across the gate's scope, for the coverage assertion.</summary>
    internal static int CitationCount() =>
        DocSnippetExtractor.MarkdownFiles().Sum(document => Scan(document).Count);

    /// <summary>Documents carrying at least one citation, for the coverage assertion.</summary>
    internal static int CitedDocumentCount() =>
        DocSnippetExtractor.MarkdownFiles().Count(document => Scan(document).Count > 0);

    /// <summary>Scans one document for citations, in document order.</summary>
    static IReadOnlyList<(int Line, string Reference, string Qualifier, string FileName, int LastLine)> Scan(
        string documentPath)
    {
        var absolute = Path.Combine(DocSnippetExtractor.RepoRoot, documentPath.Replace('/', Path.DirectorySeparatorChar));

        if (!File.Exists(absolute))
            throw new FileNotFoundException($"Document not found: {documentPath}", absolute);

        var lines = File.ReadAllLines(absolute);
        var found = new List<(int, string, string, string, int)>();

        for (var i = 0; i < lines.Length; i++)
        {
            foreach (Match match in CitationPattern().Matches(lines[i]))
            {
                var start = int.Parse(match.Groups["start"].Value, CultureInfo.InvariantCulture);
                var last = match.Groups["end"] is { Success: true } end
                    ? int.Parse(end.Value, CultureInfo.InvariantCulture)
                    : start;

                found.Add((i + 1, match.Value, match.Groups["qualifier"].Value, match.Groups["file"].Value, last));
            }
        }

        return found;
    }

    static int LineCount(string repoRelativePath)
    {
        if (LineCounts.TryGetValue(repoRelativePath, out var cached))
            return cached;

        var count = File.ReadAllLines(Sources[repoRelativePath]).Length;
        LineCounts[repoRelativePath] = count;
        return count;
    }

    static Dictionary<string, string> IndexSources()
    {
        var root = DocSnippetExtractor.RepoRoot;
        var index = new Dictionary<string, string>(StringComparer.Ordinal);
        var pending = new Stack<string>();

        pending.Push(root);

        while (pending.Count > 0)
        {
            var directory = pending.Pop();

            foreach (var file in Directory.EnumerateFiles(directory, "*.cs", SearchOption.TopDirectoryOnly))
                index[Path.GetRelativePath(root, file).Replace('\\', '/')] = file;

            // Pruned rather than filtered after the fact, so vendored trees are never walked at all.
            foreach (var child in Directory.EnumerateDirectories(directory))
            {
                var name = Path.GetFileName(child);

                if (!ExcludedSegments.Contains(name, StringComparer.Ordinal))
                    pending.Push(child);
            }
        }

        return index;
    }

    static Dictionary<string, string[]> IndexByBaseName()
    {
        var buckets = new Dictionary<string, List<string>>(StringComparer.Ordinal);

        foreach (var relative in Sources.Keys)
        {
            var name = Path.GetFileName(relative);

            if (!buckets.TryGetValue(name, out var bucket))
                buckets[name] = bucket = [];

            bucket.Add(relative);
        }

        return buckets.ToDictionary(pair => pair.Key, pair => pair.Value.ToArray(), StringComparer.Ordinal);
    }

    /// <summary>
    /// A <c>File.cs:NN</c> citation with an optional directory qualifier and optional line range.
    /// The qualifier is whole directory segments, so <c>BertModel.cs:449</c> cannot mis-split into
    /// qualifier <c>BertMode</c> and file <c>l.cs</c> — the earlier version of this pattern without
    /// the trailing slash reported every bare citation as path-qualified.
    /// </summary>
    [GeneratedRegex(@"(?<qualifier>(?:[A-Za-z0-9_.-]+/)+)?(?<file>[A-Za-z0-9_]+\.cs):(?<start>\d+)(?:-(?<end>\d+))?")]
    private static partial Regex CitationPattern();
}
