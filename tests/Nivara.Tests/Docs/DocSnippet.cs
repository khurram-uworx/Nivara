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

    /// <summary>
    /// Member declarations (a method, a property) wrapped in a class, modelling a snippet meant to
    /// be pasted inside a type such as a test fixture.
    /// </summary>
    MemberDecl,
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
        "EXAMPLES.md",
        "GETTING-STARTED.md",
    ];

    /// <summary>
    /// Every other snippet-bearing document, listed so the gate can assert that no document
    /// drifts into the skip bucket unnoticed. A new document containing ```csharp blocks must be
    /// added here or to <see cref="GatedDocuments"/> or the coverage assertion fails.
    /// </summary>
    internal static readonly string[] UngatedDocuments =
    [
        "ARCHITECTURE.md",
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

    /// <summary>
    /// Every context name a gate comment may use. The types each name supplies are compiled in
    /// <c>Preambles/SnippetStubs.cs</c>, so they cannot rot; this list is what makes the name itself
    /// a checked contract rather than a comment, and it is asserted against actual usage in both
    /// directions so a typo or a stale entry fails the gate.
    /// </summary>
    internal static readonly string[] KnownPreambles =
    [
        "agent-null-tensor",
        "agent-tensor-kernel",
        "nn-models",
        "row-types",
        "streaming-as-stream",
        "streaming-flux-basic",
        "streaming-flux-frame",
        "streaming-flux-publish",
        "streaming-flux-reverse",
        "streaming-flux-training",
        "streaming-flux-window",
    ];

    /// <summary>
    /// Transient working documents left out of the scan. They are not reader-facing, and the plan file
    /// is <c>docs/TODO.md</c> because <c>iterative-work</c> <em>commits</em> it for the duration of a
    /// multi-step piece of work and removes it at G2 — so it sits in the tree, and would otherwise be
    /// scanned, for most of a workflow's life. Excluding it by name is what keeps the snippet and
    /// citation pins from moving while it exists. A named list, never a glob: a broad rule would hide
    /// real documents. Asserted by both gate fixtures while the file exists, not merely named here.
    /// </summary>
    internal static readonly string[] ScanExcludedDocuments = ["docs/TODO.md"];

    /// <summary>
    /// Whether a repo-relative markdown path is inside the gate's scan scope. One decision site, so
    /// the filter a test exercises is the filter <see cref="MarkdownFiles"/> applies.
    /// </summary>
    internal static bool IsScanned(string repoRelativePath) => IsScanned(repoRelativePath, ScanExcludedDocuments);

    /// <summary>
    /// The scope predicate over an explicit exclusion list. The second parameter exists so a test can
    /// ask the counterfactual — would this path be scanned without that entry — which is the only way
    /// to tell a working exclusion from one that is merely declared.
    /// </summary>
    internal static bool IsScanned(string repoRelativePath, IReadOnlyList<string> exclusions) =>
        !exclusions.Contains(repoRelativePath, StringComparer.Ordinal);

    /// <summary>Absolute path of the transient plan document.</summary>
    internal static string PlanDocumentPath => Path.Combine(RepoRoot, "docs", "TODO.md");

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
            .Where(IsScanned)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();
    }

    /// <summary>
    /// Materialises the transient plan document so a test can observe the scan scope with it present.
    /// Asserting that the scan skips a file that is not there proves nothing — the file's absence and a
    /// working exclusion produce identical results — so the exclusion has to be observed while the file
    /// exists.
    /// <para>
    /// When the file already exists, the normal case while a plan is in flight, it is used as-is and
    /// never deleted. Otherwise a body engineered to move every pin it could is written and removed on
    /// dispose. NUnit runs fixtures sequentially unless marked <c>[Parallelizable]</c>, and none here
    /// are, so this cannot interleave with the coverage assertions that count documents.
    /// </para>
    /// </summary>
    internal static IDisposable PlanDocumentPresent()
    {
        if (File.Exists(PlanDocumentPath))
            return new PlanDocumentScope(null);

        File.WriteAllText(PlanDocumentPath, PlanProbeBody);
        return new PlanDocumentScope(PlanDocumentPath);
    }

    /// <summary>
    /// A body chosen so a leak is unmissable. The fenced block moves the repository-wide snippet total
    /// and the document-classification assertion; the citation moves the citation totals. The citation
    /// resolves on purpose — an unresolvable one would raise a fault of its own, and the totals would
    /// stop being the thing that catches the leak.
    /// </summary>
    const string PlanProbeBody = """
        # Plan (probe)

        ```csharp
        var sum = 1 + 1;
        ```

        Grounding: `src/Nivara/Execution/NivaraExecutionContext.cs:17`.
        """;

    /// <summary>Removes the probe document, and only if this scope is what created it.</summary>
    sealed class PlanDocumentScope(string? createdPath) : IDisposable
    {
        public void Dispose()
        {
            if (createdPath is not null)
                File.Delete(createdPath);
        }
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