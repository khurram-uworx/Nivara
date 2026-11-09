namespace NivaraChat.Helpers;

/// <summary>The four retrieval metrics for one ranker over one query set.</summary>
public sealed record RankerMetrics(double Precision, double Recall, double Mrr, double Ndcg);

/// <summary>
/// One ranker's result on a query set: the mean, the per-query metrics (for a paired test), and the
/// per-query NDCG on its own so a win/loss comparison does not have to re-derive it.
/// </summary>
public sealed record RankerResult(
    string Name,
    RankerMetrics Mean,
    IReadOnlyList<RankerMetrics> PerQuery,
    IReadOnlyList<double> Ndcg)
{
    /// <summary>Number of queries that contributed a metric.</summary>
    public int QueryCount => PerQuery.Count;
}

/// <summary>
/// Ranks pooled candidates and measures them with the #914 metric suite. Every ranker is measured
/// over the same candidate pool, so a comparison is a reranking comparison and not a retrieval one.
/// </summary>
public static class RetrievalEvaluation
{
    /// <summary>Cut-off used by the headline tables.</summary>
    public const int DefaultK = 10;

    /// <summary>Reconstructs the per-row <see cref="ViewScores"/> a <see cref="TrainingQuery"/> was flattened from.</summary>
    public static ViewScores[] Reconstruct(TrainingQuery query)
    {
        ArgumentNullException.ThrowIfNull(query);

        var rows = new ViewScores[query.RowCount];
        for (var row = 0; row < query.RowCount; row++)
        {
            rows[row] = new ViewScores(
                Value(query, row, ViewSpace.Content),
                Value(query, row, ViewSpace.Title),
                Value(query, row, ViewSpace.Section),
                Value(query, row, ViewSpace.Code));
        }

        return rows;
    }

    /// <summary>
    /// Fused scores for the #914 fixed weights, computed by <see cref="MultiViewScoring.Fuse"/> rather
    /// than by the tensor path. The comparison baseline is the sample's own arithmetic, not a second
    /// implementation that only claims to reproduce it.
    /// </summary>
    public static float[] FixedScores(TrainingQuery query, FusionWeights weights)
    {
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(weights);

        var rows = Reconstruct(query);
        var scores = new float[rows.Length];
        for (var row = 0; row < rows.Length; row++)
            scores[row] = MultiViewScoring.Fuse(rows[row], weights) ?? float.NegativeInfinity;

        return scores;
    }

    /// <summary>Fused scores from a trainable ranker, outside any gradient scope.</summary>
    public static float[] Scores(IFusionRanker ranker, TrainingQuery query)
    {
        ArgumentNullException.ThrowIfNull(ranker);
        ArgumentNullException.ThrowIfNull(query);

        return MaskedFusion.ToArray(ranker.ForwardMasked(query));
    }

    /// <summary>
    /// Indices of the candidates in descending fused order, cut at <paramref name="topK"/>. Ties break
    /// by candidate index (which is ascending row order, because the pool is built sorted), so two runs
    /// over the same data cannot disagree about which chunk was first.
    /// </summary>
    public static IReadOnlyList<int> Rank(float[] fused, int topK = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(fused);
        if (topK <= 0)
            return [];

        return
        [
            .. Enumerable.Range(0, fused.Length)
                .OrderByDescending(index => fused[index])
                .ThenBy(index => index)
                .Take(Math.Min(topK, fused.Length))
        ];
    }

    /// <summary>Measures one ranked list against the query's graded candidates.</summary>
    public static RankerMetrics Measure(IReadOnlyList<int> ranked, TrainingQuery query, int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(ranked);
        ArgumentNullException.ThrowIfNull(query);

        var retrieved = new List<string>(ranked.Count);
        foreach (var index in ranked)
        {
            if (index >= 0 && index < query.RowCount)
                retrieved.Add(query.Candidates[index].ChunkId);
        }

        var relevant = new HashSet<string>(StringComparer.Ordinal);
        var grades = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (var candidate in query.Candidates)
        {
            if (candidate.Grade > 0)
                relevant.Add(candidate.ChunkId);

            if (!grades.TryGetValue(candidate.ChunkId, out var existing) || candidate.Grade > existing)
                grades[candidate.ChunkId] = candidate.Grade;
        }

        return new RankerMetrics(
            RagEvalMetrics.PrecisionAtK(retrieved, relevant, k),
            RagEvalMetrics.RecallAtK(retrieved, relevant, k),
            RagEvalMetrics.Mrr(retrieved, relevant),
            RagEvalMetrics.NdcgAtK(retrieved, grades, k));
    }

    /// <summary>Mean NDCG@K across the usable queries; NaN when there are none.</summary>
    public static double MeanNdcg(IFusionRanker ranker, IReadOnlyList<TrainingQuery> queries, int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(ranker);
        ArgumentNullException.ThrowIfNull(queries);

        var total = 0d;
        var count = 0;
        foreach (var query in queries)
        {
            if (!query.IsUsable)
                continue;

            total += Measure(Rank(Scores(ranker, query), k), query, k).Ndcg;
            count++;
        }

        return count == 0 ? double.NaN : total / count;
    }

    /// <summary>Full evaluation of a trainable ranker.</summary>
    public static RankerResult Evaluate(
        string name,
        IFusionRanker ranker,
        IReadOnlyList<TrainingQuery> queries,
        int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(ranker);
        return Evaluate(name, queries, query => Scores(ranker, query), k);
    }

    /// <summary>Full evaluation of the fixed #914 fusion.</summary>
    public static RankerResult EvaluateFixed(
        string name,
        FusionWeights weights,
        IReadOnlyList<TrainingQuery> queries,
        int k = DefaultK)
    {
        ArgumentNullException.ThrowIfNull(weights);
        return Evaluate(name, queries, query => FixedScores(query, weights), k);
    }

    /// <summary>Reports the mean and per-query metrics for a fixed score vector per query.</summary>
    static RankerResult Evaluate(
        string name,
        IReadOnlyList<TrainingQuery> queries,
        Func<TrainingQuery, float[]> score,
        int k)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(queries);
        ArgumentNullException.ThrowIfNull(score);

        var perQuery = new List<RankerMetrics>();
        foreach (var query in queries)
        {
            if (!query.IsUsable)
                continue;

            perQuery.Add(Measure(Rank(score(query), k), query, k));
        }

        var mean = perQuery.Count == 0
            ? new RankerMetrics(0d, 0d, 0d, 0d)
            : new RankerMetrics(
                RagEvalMetrics.Mean([.. perQuery.Select(metrics => metrics.Precision)]),
                RagEvalMetrics.Mean([.. perQuery.Select(metrics => metrics.Recall)]),
                RagEvalMetrics.Mean([.. perQuery.Select(metrics => metrics.Mrr)]),
                RagEvalMetrics.Mean([.. perQuery.Select(metrics => metrics.Ndcg)]));

        return new RankerResult(name, mean, perQuery, [.. perQuery.Select(metrics => metrics.Ndcg)]);
    }

    /// <summary>Paired comparison of two rankers over the same queries.</summary>
    public static PairedComparison Compare(RankerResult challenger, RankerResult baseline)
    {
        ArgumentNullException.ThrowIfNull(challenger);
        ArgumentNullException.ThrowIfNull(baseline);

        if (challenger.Ndcg.Count != baseline.Ndcg.Count)
            throw new ArgumentException(
                $"Paired comparison needs the same query count: {challenger.Ndcg.Count} vs {baseline.Ndcg.Count}.");

        var (wins, losses, ties) = RagEvalMetrics.WinLossTie(challenger.Ndcg, baseline.Ndcg);
        return new PairedComparison(
            challenger.Name,
            baseline.Name,
            wins,
            losses,
            ties,
            RagEvalMetrics.SignTestPValue(wins, losses),
            RagEvalMetrics.CohensD(challenger.Ndcg, baseline.Ndcg),
            challenger.Mean.Ndcg - baseline.Mean.Ndcg);
    }

    static float? Value(TrainingQuery query, int row, ViewSpace space)
    {
        var index = row * TrainingSetBuilder.ViewCount + (int)space;
        return query.Present[index] ? query.Scores[index] : null;
    }
}

/// <summary>Structured, byte-stable record of a fixed-vs-baseline evaluation run.</summary>
public sealed record RetrievalBaseline(
    int SchemaVersion,
    CorpusIdentity Corpus,
    string LabelKind,
    string LabelDescription,
    string SourcePath,
    int MatchedLabelRows,
    int TotalLabelRows,
    double Coverage,
    bool IsComplete,
    int Seed,
    int K,
    int Candidates,
    int TrainCount,
    int ValidationCount,
    int TestCount,
    IReadOnlyDictionary<string, RankerMetrics> FixedFullSet,
    IReadOnlyDictionary<string, RankerMetrics> FixedTestSplit)
{
    public const int CurrentSchemaVersion = 1;

    public static RetrievalBaseline Create(
        CorpusIdentity corpus,
        ResolvedLabels labels,
        string sourcePath,
        int seed,
        int k,
        int candidates,
        QuerySplit split,
        int trainCount,
        IReadOnlyDictionary<string, RankerMetrics> fixedFullSet,
        IReadOnlyDictionary<string, RankerMetrics> fixedTestSplit)
        => new(
            CurrentSchemaVersion,
            corpus,
            labels.Kind.ToString().ToLowerInvariant(),
            labels.Description,
            sourcePath,
            labels.MatchedLabelRows,
            labels.TotalLabelRows,
            labels.Coverage,
            labels.IsComplete,
            seed,
            k,
            candidates,
            trainCount,
            split.Validation.Count,
            split.Test.Count,
            fixedFullSet,
            fixedTestSplit);

    public static void Write(string path, RetrievalBaseline baseline)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(baseline);
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
        File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(baseline, options));
    }

    public static RetrievalBaseline Read(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        var json = File.ReadAllText(path);
        return System.Text.Json.JsonSerializer.Deserialize<RetrievalBaseline>(json)!;
    }
}

/// <summary>
/// Challenger vs baseline on paired NDCG. The sign test answers "is the direction consistent" and
/// Cohen's d answers "how large", so neither a tiny consistent win nor a large noisy one is mistaken
/// for the other.
/// </summary>
public sealed record PairedComparison(
    string Challenger,
    string Baseline,
    int Wins,
    int Losses,
    int Ties,
    double SignTestPValue,
    double CohensD,
    double MeanNdcgDelta)
{
    /// <summary>Whether the challenger is ahead on mean NDCG with a sign test under 0.05.</summary>
    public bool IsSignificant => MeanNdcgDelta > 0 && SignTestPValue < 0.05;

    /// <summary>One-line summary used by both the console table and the markdown report.</summary>
    public string Summary =>
        $"{Challenger} vs {Baseline}: wins={Wins}, losses={Losses}, ties={Ties}, " +
        $"dNDCG={MeanNdcgDelta:+0.0000;-0.0000;0.0000}, p={SignTestPValue:0.0000}, d={CohensD:0.000}";
}
