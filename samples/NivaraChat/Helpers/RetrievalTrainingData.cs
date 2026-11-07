using System.Text;

namespace NivaraChat.Helpers;

/// <summary>One pooled candidate: the row it came from, its stable id, and its graded relevance.</summary>
public sealed record RankedCandidate(int Row, string ChunkId, int Grade);

/// <summary>
/// One query's training batch. <see cref="Scores"/> and <see cref="Present"/> are the candidate rows
/// flattened row-major into a <c>[rows, 4]</c> block, which is the exact layout the AutoDiff fusion
/// multiplies against. <see cref="LossTarget"/> is the per-row listwise target, normalized once so
/// the trainer never rebuilds it per epoch.
/// </summary>
public sealed record TrainingQuery(
    string Query,
    float[] QueryVector,
    IReadOnlyList<RankedCandidate> Candidates,
    float[] Scores,
    bool[] Present,
    float[] LossTarget,
    int DroppedCandidates,
    string LabelSource)
{
    /// <summary>Number of pooled candidate rows.</summary>
    public int RowCount => Candidates.Count;

    /// <summary>Whether this query can contribute a ranking loss at all.</summary>
    public bool IsUsable => RowCount >= 2 && Candidates.Any(candidate => candidate.Grade > 0);

    /// <summary>Number of the four views present at <paramref name="row"/>.</summary>
    public int PresentCount(int row)
    {
        var count = 0;
        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
        {
            if (Present[row * TrainingSetBuilder.ViewCount + view])
                count++;
        }

        return count;
    }

    /// <summary>One row's scores in <see cref="ViewSpace"/> order, with 0 where the view is absent.</summary>
    public float[] ScoresFor(int row)
    {
        var values = new float[TrainingSetBuilder.ViewCount];
        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
            values[view] = Scores[row * TrainingSetBuilder.ViewCount + view];

        return values;
    }

    /// <summary>One row's presence mask in <see cref="ViewSpace"/> order.</summary>
    public bool[] PresentFor(int row)
    {
        var values = new bool[TrainingSetBuilder.ViewCount];
        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
            values[view] = Present[row * TrainingSetBuilder.ViewCount + view];

        return values;
    }
}

/// <summary>
/// Chooses which rows the reranker is allowed to reorder. Selection is deliberately weight-agnostic:
/// the union of each view's own top-K (plus every labeled positive) means a chunk that only the Code
/// or Title view ranks highly can still be promoted by a learned fusion. Generating the pool from the
/// fixed baseline's top-K would let the baseline choose its own competitors.
/// </summary>
public static class CandidatePool
{
    /// <summary>Rows each view contributes before the union is taken.</summary>
    public const int DefaultPerView = 10;

    /// <summary>
    /// Returns the union of each view's top-<paramref name="perView"/> rows and every row graded
    /// above zero, ordered by ascending row index so the batch layout is deterministic.
    /// </summary>
    /// <param name="scores">One entry per index row</param>
    /// <param name="grades">Graded relevance by row; rows absent from the map are unlabeled</param>
    /// <param name="perView">Rows to take from each view's ranking</param>
    public static IReadOnlyList<int> Select(
        IReadOnlyList<ViewScores> scores,
        IReadOnlyDictionary<int, int> grades,
        int perView = DefaultPerView)
    {
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(grades);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(perView);

        var selected = new HashSet<int>();

        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
        {
            var space = (ViewSpace)view;
            var ranked = new List<(int Row, float Score)>();
            for (var row = 0; row < scores.Count; row++)
            {
                if (scores[row][space] is { } score)
                    ranked.Add((row, score));
            }

            ranked.Sort(static (left, right) =>
            {
                var byScore = right.Score.CompareTo(left.Score);
                return byScore != 0 ? byScore : left.Row.CompareTo(right.Row);
            });

            for (var i = 0; i < ranked.Count && i < perView; i++)
                selected.Add(ranked[i].Row);
        }

        foreach (var (row, grade) in grades)
        {
            if (grade > 0)
                selected.Add(row);
        }

        var rows = new int[selected.Count];
        var index = 0;
        foreach (var row in selected)
            rows[index++] = row;

        Array.Sort(rows);
        return rows;
    }
}

/// <summary>
/// Builds a <see cref="TrainingQuery"/> from an index's scores and a query's graded labels. Kept free
/// of MiniLM so the fusion properties can be tested with synthetic vectors.
/// </summary>
public static class TrainingSetBuilder
{
    /// <summary>Number of views the #914 index scores. Also the width of every flat score block.</summary>
    public const int ViewCount = 4;

    /// <summary>Graded relevance feeds the listwise target as <c>2^grade</c>, so grade 2 outranks 1 and 0.</summary>
    public const float GradeBase = 2f;

    /// <summary>
    /// Selects the candidate pool, flattens it, and computes the listwise target. Rows with no scored
    /// view are dropped and counted: they cannot contribute a gradient, and leaving them in would put a
    /// zero denominator into the fusion divide.
    /// </summary>
    public static TrainingQuery Create(
        string query,
        float[] queryVector,
        IReadOnlyList<MultiViewChunk> chunks,
        IReadOnlyList<ViewScores> scores,
        IReadOnlyDictionary<int, int> grades,
        string labelSource,
        int perView = CandidatePool.DefaultPerView)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(queryVector);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(scores);
        ArgumentNullException.ThrowIfNull(grades);

        if (scores.Count != chunks.Count)
            throw new ArgumentException(
                $"Score count ({scores.Count}) must match the chunk count ({chunks.Count}).", nameof(scores));

        var pooled = CandidatePool.Select(scores, grades, perView);

        var candidates = new List<RankedCandidate>(pooled.Count);
        var scoresBlock = new List<float>(pooled.Count * ViewCount);
        var presentBlock = new List<bool>(pooled.Count * ViewCount);
        var dropped = 0;

        foreach (var row in pooled)
        {
            var viewScores = scores[row];
            if (viewScores.PresentCount == 0)
            {
                dropped++;
                continue;
            }

            var grade = grades.TryGetValue(row, out var value) ? value : 0;
            candidates.Add(new RankedCandidate(row, chunks[row].ChunkId, grade));

            for (var view = 0; view < ViewCount; view++)
            {
                var score = viewScores[(ViewSpace)view];
                presentBlock.Add(score is not null);
                scoresBlock.Add(score ?? 0f);
            }
        }

        return new TrainingQuery(
            query,
            queryVector,
            candidates,
            [.. scoresBlock],
            [.. presentBlock],
            BuildTarget(candidates),
            dropped,
            labelSource);
    }

    /// <summary>
    /// Normalizes <c>2^grade</c> across the pool. Falls back to a uniform target when every row is
    /// graded zero, so a query with no positive cannot produce a NaN loss or a zero-length target.
    /// </summary>
    static float[] BuildTarget(IReadOnlyList<RankedCandidate> candidates)
    {
        var targets = new float[candidates.Count];
        var total = 0f;
        for (var i = 0; i < candidates.Count; i++)
        {
            targets[i] = MathF.Pow(GradeBase, candidates[i].Grade);
            total += targets[i];
        }

        if (total <= 0f)
        {
            Array.Fill(targets, 1f / Math.Max(1, candidates.Count));
            return targets;
        }

        for (var i = 0; i < targets.Length; i++)
            targets[i] /= total;

        return targets;
    }
}

/// <summary>Which labeling process produced the grades the headline metric is computed on.</summary>
public enum LabelKind
{
    /// <summary>LLM judge cache written by <see cref="RagEvalMode"/>; graded 0/1/2 per chunk.</summary>
    Judge,

    /// <summary>Committed hand-graded fixture keyed by document path; stable across docs edits.</summary>
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

/// <summary>Loads and resolves the three label sources, keeping the leakage caveat visible in the result.</summary>
public static class RetrievalLabels
{
    /// <summary>
    /// Parses the committed fixture: <c>query&lt;TAB&gt;document-path&lt;TAB&gt;grade</c> per line,
    /// with <c>#</c> comments. The target is a <em>document</em> path rather than a
    /// <c>path#chunk</c> id on purpose — chunk ordinals move whenever a document is edited, which is
    /// how the judge cache went stale, while document paths survive.
    /// </summary>
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
    /// chunks. A duplicated chunk id grades every row that carries it; a document path grades every
    /// chunk of that document.
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

/// <summary>
/// Deterministic 80/10/10 split by stable hash. <see cref="string.GetHashCode()"/> is randomized per
/// process, so a split built on it would silently differ between runs and make two reports
/// incomparable; FNV-1a over UTF-8 bytes with the seed mixed in is reproducible forever.
/// </summary>
public sealed record QuerySplit(
    IReadOnlyList<string> Train,
    IReadOnlyList<string> Validation,
    IReadOnlyList<string> Test)
{
    /// <summary>Default seed so the split is reproducible without a flag.</summary>
    public const int DefaultSeed = 42;

    /// <summary>Splits <paramref name="queries"/> by hash bucket, ordering each part ordinal for stable output.</summary>
    public static QuerySplit Create(
        IEnumerable<string> queries,
        int seed = DefaultSeed,
        double validationFraction = 0.1,
        double testFraction = 0.1)
    {
        ArgumentNullException.ThrowIfNull(queries);
        if (validationFraction < 0 || testFraction < 0 || validationFraction + testFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(testFraction), "Validation and test fractions must be non-negative and sum to less than 1.");

        var train = new List<string>();
        var validation = new List<string>();
        var test = new List<string>();

        foreach (var query in queries.Distinct(StringComparer.Ordinal))
        {
            var bucket = Bucket(query, seed);
            if (bucket < testFraction)
                test.Add(query);
            else if (bucket < testFraction + validationFraction)
                validation.Add(query);
            else
                train.Add(query);
        }

        return new QuerySplit(
            [.. train.OrderBy(query => query, StringComparer.Ordinal)],
            [.. validation.OrderBy(query => query, StringComparer.Ordinal)],
            [.. test.OrderBy(query => query, StringComparer.Ordinal)]);
    }

    /// <summary>Maps a query into <c>[0, 1)</c> with a process-independent hash.</summary>
    static double Bucket(string query, int seed)
    {
        var hash = 2166136261u ^ (uint)seed;
        foreach (var value in Encoding.UTF8.GetBytes(query))
        {
            hash ^= value;
            hash *= 16777619u;
        }

        return hash / (double)uint.MaxValue;
    }
}
