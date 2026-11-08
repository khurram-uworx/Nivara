namespace NivaraChat.Helpers;

/// <summary>
/// The markdown corpus every RAG mode shares. Plan documents under <c>docs/plan/</c> and
/// <c>docs/plans/</c> are working notes and roadmaps, not reader-facing documentation, so they are
/// excluded from chunking, from the auto-derived heading/title queries and from the eval ground
/// truth. Keeping one enumeration site means <c>--rag</c>, <c>--rag-multiview</c>, <c>--rag-eval</c>
/// and <c>--rag-learn</c> cannot disagree about what the corpus is.
/// </summary>
public static class DocCorpus
{
    /// <summary>Directory names under the docs root that hold plan documents.</summary>
    static readonly string[] PlanDirectories = ["plan"];

    /// <summary>
    /// Markdown files under <paramref name="docsDir"/>, plus the sample README, minus plan documents.
    /// </summary>
    public static IReadOnlyList<string> EnumerateMarkdownFiles(string docsDir, string repoRoot)
    {
        var files = new List<string>();
        if (Directory.Exists(docsDir))
        {
            files.AddRange(Directory
                .GetFiles(docsDir, "*.md", SearchOption.AllDirectories)
                .Where(path => !IsPlanDocument(docsDir, path))
                .OrderBy(path => path, StringComparer.Ordinal));
        }

        var readmePath = Path.Combine(repoRoot, "samples", "NivaraChat", "README.md");
        if (File.Exists(readmePath))
            files.Add(readmePath);

        return files;
    }

    /// <summary>Whether <paramref name="path"/> sits directly under a plan directory of the docs root.</summary>
    public static bool IsPlanDocument(string docsDir, string path)
    {
        var relative = Path.GetRelativePath(docsDir, path);
        var firstSegment = relative.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)[0];
        return PlanDirectories.Contains(firstSegment, StringComparer.OrdinalIgnoreCase);
    }
}
