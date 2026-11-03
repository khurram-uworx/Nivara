using System.Globalization;
using System.Text.RegularExpressions;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Nivara.Tests.Docs;

/// <summary>Why a citation could not be resolved.</summary>
enum DocCitationFaultKind
{
    /// <summary>The citation named a source file that does not exist.</summary>
    MissingFile,
    /// <summary>A bare citation named a basename that exists in more than one file.</summary>
    AmbiguousFile,
    /// <summary>The cited line, or the last line of the cited range, is past end-of-file.</summary>
    PastEndOfFile,
    /// <summary>The reference named a markdown document that does not exist.</summary>
    MissingDocument,
    /// <summary>The cited symbol does not exist in the named source file.</summary>
    MissingSymbol,
    /// <summary>A symbol citation named a member that exists in more than one overload and gave no signature.</summary>
    AmbiguousSymbol,
}

/// <param name="DocumentPath">Repo-relative path of the markdown file holding the citation.</param>
/// <param name="DocumentLine">One-based line within that file.</param>
/// <param name="Reference">The citation exactly as written, e.g. <c>src/Nivara/Foo.cs::Bar()</c>.</param>
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

/// <summary>A citation discovered in a markdown file.</summary>
sealed record ScannedCitation(
    int Line,
    string Reference,
    string Qualifier,
    string FileName,
    bool IsSymbol,
    string Symbol,
    string Signature,
    int StartLine,
    int EndLine);

/// <summary>
/// Extracts citations from markdown and checks that each resolves. Two forms are
/// accepted during migration:
/// <list type="bullet">
/// <item><c>File.cs:NN</c> (legacy line-number form) — resolves file + line range.</item>
/// <item><c>File.cs::MemberName</c>, optionally <c>File.cs::MemberName(sig)</c> — resolves
/// a source symbol via Roslyn.</item>
/// </list>
/// Source resolution uses Roslyn syntax trees (cached per file), never regex.
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

        foreach (var citation in Scan(documentPath))
        {
            string resolved;
            string qualifier = citation.Qualifier;

            // Resolve the source file (common to both forms).
            if (qualifier.Length > 0)
            {
                resolved = qualifier + citation.FileName;
                if (!Sources.ContainsKey(resolved))
                {
                    faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                        DocCitationFaultKind.MissingFile, $"no source file at {resolved}"));
                    continue;
                }
            }
            else if (!SourcesByBaseName.TryGetValue(citation.FileName, out var candidates))
            {
                faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                    DocCitationFaultKind.MissingFile, $"no source file named {citation.FileName}"));
                continue;
            }
            else if (candidates.Length > 1)
            {
                var list = string.Join(", ", candidates.Order(StringComparer.Ordinal));
                faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                    DocCitationFaultKind.AmbiguousFile,
                    $"{citation.FileName} exists in {candidates.Length} files ({list}); qualify the path"));
                continue;
            }
            else
            {
                resolved = candidates[0];
            }

            if (citation.IsSymbol)
            {
                var entries = DocCitationSymbolIndex.Get(resolved);
                var matches = entries.Where(e => e.Name == citation.Symbol).ToArray();

                if (matches.Length == 0)
                {
                    faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                        DocCitationFaultKind.MissingSymbol,
                        $"{resolved} has no symbol named '{citation.Symbol}'"));
                    continue;
                }

                if (citation.Signature.Length > 0)
                {
                    var wanted = NormalizeSignature(citation.Signature);
                    var hit = matches.FirstOrDefault(m => NormalizeSignature(m.Signature) == wanted);
                    if (hit is null)
                    {
                        faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                            DocCitationFaultKind.MissingSymbol,
                            $"{resolved} has no overload of '{citation.Symbol}' with signature ({citation.Signature})"));
                        continue;
                    }
                }
                else
                {
                    var unsigned = matches.Where(m => m.Signature.Length == 0).ToArray();
                    if (unsigned.Length == 1)
                    {
                        // Unique non-overloaded member (type/field/property) — resolves.
                    }
                    else if (matches.Length == 1)
                    {
                        // Single declaration — resolves.
                    }
                    else if (unsigned.Length > 1)
                    {
                        faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                            DocCitationFaultKind.AmbiguousSymbol,
                            $"'{citation.Symbol}' in {resolved} has {unsigned.Length} declarations; add a signature"));
                        continue;
                    }
                    else
                    {
                        faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                            DocCitationFaultKind.AmbiguousSymbol,
                            $"'{citation.Symbol}' in {resolved} has {matches.Length} overloads; add a signature"));
                        continue;
                    }
                }
            }
            else
            {
                var lineCount = LineCount(resolved);
                if (citation.EndLine > lineCount)
                    faults.Add(new DocCitationFault(documentPath, citation.Line, citation.Reference,
                        DocCitationFaultKind.PastEndOfFile, $"{resolved} has {lineCount} lines"));
            }
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

    /// <summary>
    /// Citations in one document, for the widened scope's coverage assertion.
    /// </summary>
    internal static int CitationCountIn(string documentPath) => Scan(documentPath).Count;

    /// <summary>Documents carrying at least one citation, for the coverage assertion.</summary>
    internal static int CitedDocumentCount() =>
        DocSnippetExtractor.MarkdownFiles().Count(document => Scan(document).Count > 0);

    /// <summary>Scans one document for citations, in document order.</summary>
    static IReadOnlyList<ScannedCitation> Scan(string documentPath)
    {
        var absolute = Path.Combine(DocSnippetExtractor.RepoRoot, documentPath.Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(absolute))
            throw new FileNotFoundException($"Document not found: {documentPath}", absolute);

        var lines = File.ReadAllLines(absolute);
        var found = new List<ScannedCitation>();

        for (var i = 0; i < lines.Length; i++)
        {
            foreach (Match match in CitationPattern().Matches(lines[i]))
            {
                var qualifier = match.Groups["qualifier"].Value;

                if (match.Groups["symbol"].Success)
                {
                    var file = match.Groups["nsFile"].Value;
                    var symbol = match.Groups["symbol"].Value;
                    var sig = match.Groups["sig"].Success ? match.Groups["sig"].Value : string.Empty;
                    found.Add(new ScannedCitation(i + 1, match.Value, qualifier, file, true, symbol, sig, 0, 0));
                }
                else
                {
                    var file = match.Groups["oldFile"].Value;
                    var start = int.Parse(match.Groups["start"].Value, CultureInfo.InvariantCulture);
                    var last = match.Groups["end"] is { Success: true } end
                        ? int.Parse(end.Value, CultureInfo.InvariantCulture)
                        : start;
                    found.Add(new ScannedCitation(i + 1, match.Value, qualifier, file, false, string.Empty, string.Empty, start, last));
                }
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

    static string NormalizeSignature(string signature) =>
        string.Join(",", signature.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(s => string.Join(" ", s.Split(' ', StringSplitOptions.RemoveEmptyEntries))));

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
    /// A citation with an optional directory qualifier and either a symbol form
    /// (<c>File.cs::Member</c>) or a legacy line form (<c>File.cs:NN</c>).
    /// </summary>
    [GeneratedRegex(@"(?<qualifier>(?:[A-Za-z0-9_.-]+/)+)?(?:(?<nsFile>[A-Za-z0-9_]+\.cs)::(?<symbol>[A-Za-z_][A-Za-z0-9_]*(?:<[^>]*>)?)(?:\((?<sig>[^)]*)\))?|(?<oldFile>[A-Za-z0-9_]+\.cs):(?<start>\d+)(?:-(?<end>\d+))?)")]
    private static partial Regex CitationPattern();
}

/// <summary>
/// Roslyn-backed index of member declarations per source file, cached for the
/// lifetime of the process. Used to resolve <c>File.cs::Member</c> citations.
/// </summary>
static class DocCitationSymbolIndex
{
    internal sealed record SymbolEntry(string Name, string Signature, int StartLine, int EndLine);

    static readonly Dictionary<string, SymbolEntry[]> Cache = new(StringComparer.Ordinal);
    static readonly object Gate = new();

    internal static SymbolEntry[] Get(string repoRelativePath)
    {
        lock (Gate)
        {
            if (Cache.TryGetValue(repoRelativePath, out var cached))
                return cached;
        }

        var absolute = Path.Combine(DocSnippetExtractor.RepoRoot, repoRelativePath.Replace('/', Path.DirectorySeparatorChar));
        var entries = File.Exists(absolute) ? Parse(absolute) : [];
        lock (Gate)
        {
            Cache[repoRelativePath] = entries;
            return entries;
        }
    }

    static SymbolEntry[] Parse(string absolutePath)
    {
        var text = File.ReadAllText(absolutePath);
        var tree = CSharpSyntaxTree.ParseText(text, new CSharpParseOptions(LanguageVersion.Latest));
        var root = tree.GetRoot();
        var list = new List<SymbolEntry>();

        foreach (var member in root.ChildNodes().OfType<MemberDeclarationSyntax>())
            Collect(member, parentFullName: null, list);

        return list.ToArray();
    }

    static void Collect(MemberDeclarationSyntax member, string? parentFullName, List<SymbolEntry> list)
    {
        switch (member)
        {
            case BaseNamespaceDeclarationSyntax nsDecl:
                var nsName = nsDecl.Name.ToString();
                var nsFull = parentFullName is null ? nsName : parentFullName + "." + nsName;
                foreach (var child in nsDecl.Members)
                    Collect(child, nsFull, list);
                break;

            case EnumDeclarationSyntax enumDecl:
                Add(enumDecl.Identifier.Text, string.Empty, enumDecl, list);
                break;

            case TypeDeclarationSyntax typeDecl:
                var typeName = typeDecl.Identifier.Text;
                if (typeDecl.TypeParameterList is not null)
                {
                    var tps = string.Join(",", typeDecl.TypeParameterList.Parameters.Select(p => p.Identifier.Text));
                    typeName = $"{typeName}<{tps}>";
                }
                var typeFull = parentFullName is null ? typeName : parentFullName + "." + typeName;
                Add(typeDecl.Identifier.Text, string.Empty, typeDecl, list);
                foreach (var child in typeDecl.Members)
                    Collect(child, typeFull, list);
                break;

            case MethodDeclarationSyntax methodDecl:
                Add(methodDecl.Identifier.Text, BuildMethodSignature(methodDecl), methodDecl, list);
                break;

            case ConstructorDeclarationSyntax ctorDecl:
                Add(ctorDecl.Identifier.Text, BuildConstructorSignature(ctorDecl), ctorDecl, list);
                break;

            case PropertyDeclarationSyntax propDecl:
                Add(propDecl.Identifier.Text, string.Empty, propDecl, list);
                break;

            case FieldDeclarationSyntax fieldDecl:
                foreach (var variable in fieldDecl.Declaration.Variables)
                    Add(variable.Identifier.Text, string.Empty, variable, list);
                break;

            case EventDeclarationSyntax eventDecl:
                Add(eventDecl.Identifier.Text, string.Empty, eventDecl, list);
                break;

            case EventFieldDeclarationSyntax eventFieldDecl:
                foreach (var variable in eventFieldDecl.Declaration.Variables)
                    Add(variable.Identifier.Text, string.Empty, variable, list);
                break;
        }
    }

    static void Add(string name, string signature, SyntaxNode node, List<SymbolEntry> list)
    {
        var span = node.GetLocation().GetLineSpan();
        list.Add(new SymbolEntry(name, signature, span.StartLinePosition.Line + 1, span.EndLinePosition.Line + 1));
    }

    static string BuildMethodSignature(MethodDeclarationSyntax method) =>
        string.Join(",", method.ParameterList.Parameters.Select(p =>
        {
            var type = p.Type?.ToString() ?? "var";
            return string.Join(" ", type.Split(' ', StringSplitOptions.RemoveEmptyEntries));
        }));

    static string BuildConstructorSignature(ConstructorDeclarationSyntax ctor) =>
        string.Join(",", ctor.ParameterList.Parameters.Select(p =>
        {
            var type = p.Type?.ToString() ?? "var";
            return string.Join(" ", type.Split(' ', StringSplitOptions.RemoveEmptyEntries));
        }));
}
