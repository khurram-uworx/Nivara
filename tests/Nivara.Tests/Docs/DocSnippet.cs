using System.Text.RegularExpressions;

namespace Nivara.Tests.Docs;

/// <summary>How a fenced block is spliced into its generated compilation unit.</summary>
enum DocWrapMode
{
    /// <summary>Statements at the top level of an executable. Supports <c>await</c>.</summary>
    TopLevel,

    /// <summary>A complete compilation unit; the preamble contributes using directives only.</summary>
    File,

    /// <summary>Statements wrapped in an unconstrained-by-usage generic local function.</summary>
    GenericLocal,
}

/// <param name="DocumentPath">Repo-relative path of the markdown file.</param>
/// <param name="StartLine">Zero-based line of the opening fence in that file.</param>
/// <param name="Code">The block body, verbatim.</param>
/// <param name="PreambleName">Name of the compiled preamble supplying referenced types, or null.</param>
/// <param name="Locals">Verbatim local declarations spliced in before the block body.</param>
/// <param name="Mode">How the block is wrapped.</param>
/// <param name="ExcludeSlug">Stable exclusion identifier, or null when the block is not excluded.</param>
/// <param name="ExcludeReason">Human-readable exclusion reason, or null.</param>
sealed record DocBlock(
    string DocumentPath,
    int StartLine,
    string Code,
    string? PreambleName,
    string Locals,
    DocWrapMode Mode,
    string? ExcludeSlug,
    string? ExcludeReason)
{
    public bool IsExcluded => ExcludeSlug is not null;

    /// <summary>Human-readable location used in failure messages, e.g. <c>docs/STREAMING.md:134</c>.</summary>
    public string Location => $"{DocumentPath}:{StartLine + 1}";
}

static partial class DocSnippetExtractor
{
    /// <summary>The explicit set of documents whose snippets this gate compiles.</summary>
    internal static readonly string[] GatedDocuments =
    [
        "docs/AGENT-CODE-EXAMPLES.md",
        "docs/STREAMING.md",
    ];

    /// <summary>
    /// Every other snippet-bearing document, listed so the gate can assert that no document
    /// drifts into the skip bucket unnoticed. A new document containing ```csharp blocks must be
    /// added here or to <see cref="GatedDocuments"/> or the coverage assertion fails.
    /// </summary>
    internal static readonly string[] UngatedDocuments =
    [
        "ARCHITECTURE.md",
        "EXAMPLES.md",
        "GETTING-STARTED.md",
        "README.md",
        "docs/ACCELERATION.md",
        "docs/AUTODIFF.md",
        "docs/BFLOAT16.md",
        "docs/ILGPU.md",
        "docs/INTEGERS.md",
        "docs/LINQ.md",
        "docs/QWEN.md",
        "docs/RETRAINING.md",
        "docs/SAFETENSORS.md",
        "docs/TENSORS.md",
        "docs/blog/1-the-workhorses-from-words-to-numbers.md",
        "docs/blog/2-self-attention-the-soft-crossbar-switch.md",
        "docs/blog/3-self-attention-implemented-the-dirty-details.md",
        "docs/blog/4-the-bypass-wire-the-rectifier-and-what-i-learned.md",
        "docs/research/AGENT-FRAMEWORK.md",
    ];

    /// <summary>Markdown files scanned for snippets: repository root plus everything under docs/.</summary>
    internal static IReadOnlyList<string> MarkdownFiles()
    {
        var root = RepoRoot;
        var files = Directory.GetFiles(root, "*.md", SearchOption.TopDirectoryOnly);
        var docsDir = Path.Combine(root, "docs");
        if (Directory.Exists(docsDir))
            files = files.Concat(Directory.GetFiles(docsDir, "*.md", SearchOption.AllDirectories)).ToArray();

        return files
            .Select(path => Path.GetRelativePath(root, path).Replace('\\', '/'))
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();
    }

    /// <summary>Repo root located by walking up to the solution file, so nesting changes cannot break it.</summary>
    internal static string RepoRoot { get; } = FindRepoRoot();

    /// <summary>Extracts every fenced csharp block from one markdown file.</summary>
    internal static IReadOnlyList<DocBlock> Extract(string documentPath)
    {
        var absolute = Path.Combine(RepoRoot, documentPath.Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(absolute))
            throw new FileNotFoundException($"Document not found: {documentPath}", absolute);

        var lines = File.ReadAllLines(absolute);
        var blocks = new List<DocBlock>();
        var openFence = -1;
        var body = new List<string>();

        for (var i = 0; i < lines.Length; i++)
        {
            if (openFence < 0)
            {
                if (CsharpFence().IsMatch(lines[i]))
                {
                    openFence = i;
                    body = [];
                }
                continue;
            }

            if (ClosingFence().IsMatch(lines[i]))
            {
                blocks.Add(Build(documentPath, openFence, string.Join(Environment.NewLine, body), lines, openFence));
                openFence = -1;
                continue;
            }

            body.Add(lines[i]);
        }

        if (openFence >= 0)
            throw new InvalidDataException($"{documentPath}:{openFence + 1} has an unterminated ```csharp fence");

        return blocks;
    }

    /// <summary>Builds a block from its body plus the gate comment immediately above the opening fence.</summary>
    static DocBlock Build(string documentPath, int fenceLine, string code, string[] lines, int fenceLineIndex)
    {
        var directives = ReadDirectives(lines, fenceLineIndex);
        var mode = DocWrapMode.TopLevel;
        if (directives.TryGetValue("mode", out var rawMode))
        {
            if (!Enum.TryParse<DocWrapMode>(rawMode, ignoreCase: true, out mode))
                throw new InvalidDataException($"{documentPath}:{fenceLine + 1} unknown mode '{rawMode}'");
        }

        return new DocBlock(
            documentPath,
            fenceLine + 1,
            code,
            directives.GetValueOrDefault("preamble"),
            directives.GetValueOrDefault("locals") ?? string.Empty,
            mode,
            directives.GetValueOrDefault("exclude"),
            directives.GetValueOrDefault("reason"));
    }

    /// <summary>
    /// Reads the multi-line gate comment that ends on the line above the opening fence.
    /// Blank lines between the comment and the fence are tolerated so the comment stays readable.
    /// </summary>
    static Dictionary<string, string> ReadDirectives(string[] lines, int fenceLineIndex)
    {
        var cursor = fenceLineIndex - 1;
        while (cursor >= 0 && string.IsNullOrWhiteSpace(lines[cursor]))
            cursor--;

        if (cursor < 0 || lines[cursor].Trim() != "-->")
            return [];

        var close = cursor;
        while (cursor >= 0 && !lines[cursor].Trim().StartsWith("<!--", StringComparison.Ordinal))
            cursor--;

        if (cursor < 0)
            throw new InvalidDataException($"Unterminated gate comment ending at line {close + 1}");

        if (!lines[cursor].TrimStart().StartsWith("<!--", StringComparison.Ordinal))
            return [];

        var directives = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = cursor + 1; i < close; i++)
        {
            var line = lines[i].Trim();
            if (line.Length == 0)
                continue;

            var separator = line.IndexOf(':', StringComparison.Ordinal);
            if (separator <= 0)
                throw new InvalidDataException($"Gate directive '{line}' must be 'key: value'");

            directives[line[..separator].Trim()] = line[(separator + 1)..].Trim();
        }

        return directives;
    }

    static string FindRepoRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Nivara.slnx")))
                return dir.FullName;
        }

        throw new InvalidOperationException(
            $"Nivara.slnx not found above {AppContext.BaseDirectory}; the doc gate cannot locate the repository root");
    }

    [GeneratedRegex("^```csharp\\s*$")]
    private static partial Regex CsharpFence();

    [GeneratedRegex("^```\\s*$")]
    private static partial Regex ClosingFence();
}