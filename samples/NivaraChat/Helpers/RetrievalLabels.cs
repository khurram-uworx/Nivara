using System.Text;

namespace NivaraChat.Helpers;

/// <summary>Which labeling process produced the grades the headline metric is computed on.</summary>
public enum LabelKind
{
    /// <summary>LLM judge cache written by <see cref="RagEvalMode"/>; graded 0/1/2 per chunk.</summary>
    Judge,

    /// <summary>Deprecated alias for Judge (kept for CLI compatibility).</summary>
    Fixture,

    /// <summary>Heading/title-derived pseudo-queries. Convenient and self-fulfilling; never a headline.</summary>
    Auto
}

/// <summary>
/// Graded labels resolved to index rows, with the coverage numbers needed to say whether they can be
/// trusted. A label file written against an older corpus resolves only partially, and a shrunken
/// relevant set lowers every ranker's recall without any of them being worse.
/// </summary>
public sealed record ResolvedLabels(
    LabelKind Kind,
    string Description,
    IReadOnlyDictionary<string, IReadOnlyDictionary<int, int>> Grades,
    int MatchedLabelRows,
    int TotalLabelRows)
{
    /// <summary>Fraction of label rows that resolved to at least one index row.</summary>
    public double Coverage => TotalLabelRows == 0 ? 0d : (double)MatchedLabelRows / TotalLabelRows;

    /// <summary>Queries that resolved to at least one graded row.</summary>
    public int QueryCount => Grades.Count;

    /// <summary>Whether every label row resolved; the mode refuses to train below its own threshold.</summary>
    public bool IsComplete => TotalLabelRows > 0 && MatchedLabelRows == TotalLabelRows;
}

/// <summary>
/// Loads and resolves the three label sources, keeping the leakage caveat visible in the result.
/// </summary>
/// <remarks>
/// Label sources:
/// <list type="table">
/// <item><term>judge</term><description><c>samples/data/nivarachat/eval/relevance.tsv</c> — keyed by chunk id (<c>path#ordinal</c>), graded 0/1/2 per chunk. Stable only while chunk ordinals stay stable; any docs edit that renumbers chunks invalidates it and the coverage gate catches it.</description></item>
/// <item><term>auto</term><description><c>samples/data/nivarachat/queries.tsv</c> — heading/title-derived pseudo-queries, keyed by chunk id. Leakage caveat: Title/Section score ~1 for the positive. Refresh via <c>--regenerate queries</c>.</description></item>
/// </list>
/// Precedence: an explicit <c>--labels</c> value wins; otherwise judge is used when <c>relevance.tsv</c> exists.
/// </remarks>
public static class RetrievalLabels
{
    /// <summary>Coverage below this share of resolved label rows means the grade file no longer describes this corpus.</summary>
    public const double MinimumCoverage = 0.5;

    /// <summary>Whether <paramref name="labels"/> clears the coverage gate.</summary>
    public static bool Accepts(ResolvedLabels labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        return labels.TotalLabelRows > 0 && labels.Coverage >= MinimumCoverage;
    }

    /// <summary>
    /// Pure precedence resolution for the label source.
    /// </summary>
    public static LabelKind SelectSource(string? requested, bool judgeCacheExists)
    {
        if (string.IsNullOrWhiteSpace(requested))
        {
            if (judgeCacheExists)
                return LabelKind.Judge;
            throw new ArgumentException(
                "No label source available: relevance.tsv not found. " +
                "Run --rag-eval --ollama --model <model>, or pass --labels auto.");
        }

        return requested.Trim().ToLowerInvariant() switch
        {
            "judge" => LabelKind.Judge,
            "auto" => LabelKind.Auto,
            "fixture" => LabelKind.Judge,
            _ => throw new ArgumentException(
                $"Unknown --labels value '{requested}'. Expected judge, auto or fixture.")
        };
    }

    /// <summary>Parses the committed fixture: <c>query&lt;TAB&gt;document-path&lt;TAB&gt;grade</c> per line, with <c>#</c> comments.</summary>
    public static Dictionary<string, Dictionary<string, int>> LoadFixture(string path)
    {
        var result = new Dictionary<string, Dictionary<string, int>>(StringComparer.Ordinal);
        if (!File.Exists(path))
            return result;

        foreach (var raw in File.ReadAllLines(path))
        {
            var line = raw.TrimEnd('\r');
            if (string.IsNullOrWhiteSpace(line) || line.TrimStart().StartsWith('#'))
                continue;

            var parts = line.Split('\t');
            if (parts.Length < 3)
                continue;

            var query = parts[0].Trim();
            var target = parts[1].Trim();
            if (string.IsNullOrWhiteSpace(query) || string.IsNullOrWhiteSpace(target))
                continue;
            if (!int.TryParse(parts[2], out var grade))
                continue;

            if (!result.TryGetValue(query, out var targets))
            {
                targets = new Dictionary<string, int>(StringComparer.Ordinal);
                result[query] = targets;
            }

            targets[target] = Math.Max(targets.TryGetValue(target, out var existing) ? existing : 0, grade);
        }

        return result;
    }

    /// <summary>
    /// Resolves chunk-id labels (judge, auto) or document-path labels (fixture) against the built
    /// chunks.
    /// </summary>
    public static ResolvedLabels Resolve(
        LabelKind kind,
        IReadOnlyList<MultiViewChunk> chunks,
        IEnumerable<KeyValuePair<string, IReadOnlyDictionary<string, int>>> labels)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(labels);

        var byChunkId = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var byDocument = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        for (var row = 0; row < chunks.Count; row++)
        {
            var chunk = chunks[row];
            if (!byChunkId.TryGetValue(chunk.ChunkId, out var idRows))
                byChunkId[chunk.ChunkId] = idRows = [];
            idRows.Add(row);

            if (!byDocument.TryGetValue(chunk.SourcePath, out var documentRows))
                byDocument[chunk.SourcePath] = documentRows = [];
            documentRows.Add(row);
        }

        var targetIndex = kind == LabelKind.Fixture ? byDocument : byChunkId;
        var resolved = new Dictionary<string, IReadOnlyDictionary<int, int>>(StringComparer.Ordinal);
        var matched = 0;
        var total = 0;

        foreach (var (query, targets) in labels)
        {
            var rows = new Dictionary<int, int>();
            foreach (var (target, grade) in targets)
            {
                total++;
                if (!targetIndex.TryGetValue(target, out var matchedRows))
                    continue;

                matched++;
                foreach (var row in matchedRows)
                {
                    if (!rows.TryGetValue(row, out var existing) || grade > existing)
                        rows[row] = grade;
                }
            }

            if (rows.Count > 0)
                resolved[query] = rows;
        }

        return new ResolvedLabels(kind, Describe(kind), resolved, matched, total);
    }

    /// <summary>Human-readable source description, including the leakage caveat for auto labels.</summary>
    public static string Describe(LabelKind kind) => kind switch
    {
        LabelKind.Judge => "LLM judge (graded 0/1/2 per chunk)",
        LabelKind.Fixture => "committed fixture (graded 0/1/2 per document, expanded to its chunks)",
        _ => "auto-derived from headings/titles (leakage: Title/Section score ~1 for the positive)"
    };
}
