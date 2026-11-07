using NivaraChat.Helpers;
using System.Globalization;
using System.Text;

namespace NivaraChat.Modes;

/// <summary>
/// --rag-learn: #915. Trains the #914 late-fusion weights and a query-gated mixture of the four
/// cosine experts, then compares both against the fixed baseline on queries neither saw.
/// Candidate generation stays outside the gradient path: every ranker reranks the same weight-agnostic
/// pool, so the table is a reranking comparison and not three different retrieval systems.
/// </summary>
public static class RagLearnMode
{
    /// <summary>Below this share of resolved label rows the grade file no longer describes this corpus.</summary>
    public const double MinimumLabelCoverage = 0.5;

    /// <summary>Two fusion implementations that disagree by more than this are not the same formula.</summary>
    public const double ParityTolerance = 1e-5;

    /// <summary>Decisive queries a two-sided sign test needs before p can fall below 0.05.</summary>
    public const int MinimumSignificantQueries = 6;

    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Differentiable Retrieval (#915) ===\n");

        var repoRoot = ModeHelpers.GetRepoRoot();
        var topK = ctx.EvalK ?? RetrievalEvaluation.DefaultK;
        var candidates = ctx.Candidates > 0 ? ctx.Candidates : CandidatePool.DefaultPerView;

        var options = new TrainingOptions(
            Epochs: ctx.Epochs,
            LearningRate: ctx.LearningRate,
            WeightDecay: ctx.WeightDecay,
            Patience: ctx.Patience,
            Seed: ctx.Seed,
            Loss: ParseLoss(ctx.Loss));

        using var index = await MultiViewIndex.BuildAsync(ctx);
        if (index is null)
            return;

        var labels = LoadLabels(ctx, repoRoot, index);
        if (labels is null)
            return;

        PrintLabels(labels, index.Chunks.Count);
        if (labels.Coverage < MinimumLabelCoverage)
        {
            Console.WriteLine();
            Console.WriteLine($"Refusing to train: only {labels.Coverage:P1} of label rows resolve against this corpus");
            Console.WriteLine($"(minimum {MinimumLabelCoverage:P0}). Re-run the judge, or use --labels fixture.");
            return;
        }

        var queries = BuildTrainingQueries(ctx, index, labels, candidates);
        if (queries.Count == 0)
        {
            Console.WriteLine("No query produced a usable candidate set; nothing to train on.");
            return;
        }

        var split = QuerySplit.Create(queries.Select(query => query.Query), ctx.Seed);
        var byQuery = queries.ToDictionary(query => query.Query, StringComparer.Ordinal);
        var train = Select(split.Train, byQuery, ctx.TrainQueries);
        var validation = Select(split.Validation, byQuery, 0);
        var test = Select(split.Test, byQuery, 0);

        Console.WriteLine();
        Console.WriteLine($"  Queries:      {queries.Count} usable ({split.Train.Count} train / {split.Validation.Count} validation / {split.Test.Count} test)");
        if (ctx.TrainQueries > 0 && train.Count < split.Train.Count)
            Console.WriteLine($"  Train capped: {train.Count} of {split.Train.Count} (--train-queries)");
        Console.WriteLine($"  Candidates:   union of each view's top-{candidates} + every labeled positive");
        Console.WriteLine($"  Loss:         {options.Loss}, lr {options.LearningRate}, epochs {options.Epochs}, weight decay {options.WeightDecay}");

        if (train.Count == 0 || test.Count == 0)
        {
            Console.WriteLine();
            Console.WriteLine("The split left no training or no test queries. Lower --epochs? No — pass more queries,");
            Console.WriteLine("or a smaller validation/test fraction is not configurable; use a larger label file.");
            return;
        }

        var results = new List<RankerResult>
        {
            RetrievalEvaluation.EvaluateFixed("fixed(default)", FusionWeights.Default, test, topK),
            RetrievalEvaluation.EvaluateFixed("fixed(content-only)", new FusionWeights(1f, 0f, 0f, 0f), test, topK)
        };

        var maxRows = queries.Count == 0 ? 1 : queries.Max(q => q.RowCount);
        IFusionRanker fusion;
        double parity;
        TrainingReport fusionReport;
        if (ctx.UseSilk)
        {
            try
            {
                var silkGlobal = SilkRetrievalTrainer.CreateGlobal(options.LearningRate, options.WeightDecay, maxRows);
                fusion = new SilkFusionRanker(silkGlobal);
                parity = InitialParity(fusion, test, topK);
                fusionReport = silkGlobal.Train(train, validation, options, topK);
                Console.WriteLine($"  Fusion training: Silk/OpenCL ({silkGlobal.DeviceName}), listwise per-query");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\n  Silk fusion unavailable ({ex.Message}); falling back to CPU training for fusion.");
                var cpuFusion = new FusionModel();
                fusion = cpuFusion;
                parity = InitialParity(fusion, test, topK);
                fusionReport = RetrievalTrainer.Train(cpuFusion, train, validation, options, topK);
            }
        }
        else
        {
            var cpuFusion = new FusionModel();
            fusion = cpuFusion;
            parity = InitialParity(fusion, test, topK);
            fusionReport = RetrievalTrainer.Train(cpuFusion, train, validation, options, topK);
        }


        results.Add(RetrievalEvaluation.Evaluate("learned-global", fusion, test, topK));

        IFusionRanker? gated = null;
        TrainingReport? gatedReport = null;
        RankerResult? gatedResult = null;
        var gateKind = ParseGate(ctx.Gate);
        if (gateKind is null)
        {
            Console.WriteLine("\n  Gate: skipped (--gate none)");
        }
        else
        {
            if (ctx.UseSilk)
            {
                try
                {
                    var silkGated = gateKind.Value == GateFeatureKind.ScoreStats
                        ? SilkRetrievalTrainer.CreateForStats(options.LearningRate, options.WeightDecay, maxRows)
                        : SilkRetrievalTrainer.CreateForEmbed(options.LearningRate, options.WeightDecay, maxRows, index.Dimension);
                    gatedReport = silkGated.Train(train, validation, options, topK);
                    gated = new SilkFusionRanker(silkGated);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\n  Silk gate unavailable ({ex.Message}); falling back to CPU training for the gate.");
                    var cpuGated = new GatedFusionModel(gateKind.Value, index.Dimension);
                    gatedReport = RetrievalTrainer.Train(cpuGated, train, validation, options, topK);
                    gated = cpuGated;
                }
            }
            else
            {
                var cpuGated = new GatedFusionModel(gateKind.Value, index.Dimension);
                gatedReport = RetrievalTrainer.Train(cpuGated, train, validation, options, topK);
                gated = cpuGated;
            }
            gatedResult = RetrievalEvaluation.Evaluate($"learned-gated({KindName(gateKind.Value)})", gated, test, topK);
            results.Add(gatedResult);
            PrintTraining($"learned-gated({KindName(gateKind.Value)})", gatedReport);
        }

        PrintTable(results);
        Console.WriteLine();
        Console.WriteLine($"  Init parity: max |learned - fixed| = {parity:0.0000000} " +
            (parity <= ParityTolerance ? "(ok)" : "(MISMATCH — the two fusion paths have drifted)"));

        PrintTraining("learned-global", fusionReport);
        if (gatedReport is not null)
            PrintTraining($"learned-gated({KindName(gateKind!.Value)})", gatedReport);

        Console.WriteLine();
        Console.WriteLine("  Learned weights:");
        var fixedWeights = new[] { FusionWeights.Default.Content, FusionWeights.Default.Title, FusionWeights.Default.Section, FusionWeights.Default.Code };
        var learnedWeights = fusion.Weights;
        Console.WriteLine($"    fixed:   {FormatWeights(fixedWeights)}");
        Console.WriteLine($"    learned: {FormatWeights(learnedWeights)}");
        if (gated is not null)
        {
            var mean = gated is SilkFusionRanker silkRanker
                ? MeanGateWeightsSilk(silkRanker.Trainer, test)
                : MeanGateWeights((GatedFusionModel)gated, test);
            Console.WriteLine($"    gated:   {FormatWeights(mean)} (mean over test queries)");
        }

        Console.WriteLine();
        var comparisons = new List<PairedComparison>();
        foreach (var result in results.Skip(2))
        {
            var comparison = RetrievalEvaluation.Compare(result, results[0]);
            comparisons.Add(comparison);
            Console.WriteLine($"  {comparison.Summary}");
        }

        // The two-sided sign test needs six decisive queries before p can fall below 0.05, so a tiny
        // held-out split is said out loud rather than left for the reader to notice.
        var decisive = comparisons.Count == 0 ? 0 : comparisons.Max(comparison => comparison.Wins + comparison.Losses);
        if (decisive < MinimumSignificantQueries)
        {
            Console.WriteLine(
                $"  Note: at most {decisive} decisive held-out queries; the sign test cannot reach p < 0.05 " +
                $"below {MinimumSignificantQueries}. Widen the label set for a headline comparison.");
        }

        var reportPath = ResolveReportPath(ctx.ReportPath, repoRoot);
        WriteReport(reportPath, labels, options, topK, candidates, split, train.Count, parity, results, comparisons, learnedWeights, gated, test, gateKind);
        Console.WriteLine();
        Console.WriteLine($"  Report written to {reportPath}");

        Console.WriteLine();
        Console.WriteLine("  Acceptance is a fair fixed-vs-learned-vs-gated comparison, not a guaranteed win.");
        if (comparisons.Count > 0 && !comparisons.Any(comparison => comparison.IsSignificant))
            Console.WriteLine("  No challenger beat the baseline with a significant sign test. That is a valid outcome.");

        Console.WriteLine();
        Console.WriteLine("Differentiable retrieval complete.");
    }

    static List<TrainingQuery> Select(
        IReadOnlyList<string> queries,
        IReadOnlyDictionary<string, TrainingQuery> byQuery,
        int cap)
    {
        var selected = new List<TrainingQuery>(queries.Count);
        foreach (var query in queries)
        {
            if (byQuery.TryGetValue(query, out var training))
                selected.Add(training);
        }

        if (cap > 0 && selected.Count > cap)
            selected.RemoveRange(cap, selected.Count - cap);

        return selected;
    }

    static List<TrainingQuery> BuildTrainingQueries(
        ModeContext ctx,
        MultiViewIndex index,
        ResolvedLabels labels,
        int candidates)
    {
        var queries = new List<TrainingQuery>(labels.Grades.Count);
        var failed = 0;

        foreach (var (query, grades) in labels.Grades.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            float[] vector;
            try
            {
                vector = index.EmbedQuery(query);
            }
            catch (InvalidOperationException ex)
            {
                if (failed++ == 0)
                    Console.WriteLine($"\n  Skipping queries that cannot be embedded: {ex.Message}");
                continue;
            }

            var scores = MultiViewScoring.ScoreAll(index.Matrices, index.Chunks.Count, vector);
            var training = TrainingSetBuilder.Create(
                query, vector, index.Chunks, scores, grades, labels.Description, candidates);

            if (training.IsUsable)
                queries.Add(training);
        }

        if (failed > 0)
            Console.WriteLine($"  Skipped {failed} queries with a failed embedding.");

        return queries;
    }

    static ResolvedLabels? LoadLabels(ModeContext ctx, string repoRoot, MultiViewIndex index)
    {
        var kind = ParseLabelKind(ctx.Labels, repoRoot);

        switch (kind)
        {
            case LabelKind.Judge:
                {
                    var path = JudgePath(repoRoot);
                    var cache = RagEvalLabels.Load(path);
                    if (cache.Count == 0)
                    {
                        Console.WriteLine($"No judge labels at {path}. Re-run --rag-eval --ollama, or use --labels fixture.");
                        return null;
                    }

                    var grades = cache.Select(entry => new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                        entry.Key,
                        entry.Value.ToDictionary(item => item.Key, item => item.Value.Grade, StringComparer.Ordinal)));
                    return RetrievalLabels.Resolve(LabelKind.Judge, index.Chunks, grades);
                }

            case LabelKind.Fixture:
                {
                    var path = FixturePath(repoRoot);
                    var fixture = RetrievalLabels.LoadFixture(path);
                    if (fixture.Count == 0)
                    {
                        Console.WriteLine($"No fixture labels at {path}.");
                        return null;
                    }

                    var grades = fixture.Select(entry => new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                        entry.Key, entry.Value));
                    return RetrievalLabels.Resolve(LabelKind.Fixture, index.Chunks, grades);
                }

            default:
                {
                    var path = QueriesPath(ctx, repoRoot);
                    var derived = EvalGroundTruth.Load(path);
                    if (derived.Count == 0)
                    {
                        Console.WriteLine($"No auto queries at {path}.");
                        return null;
                    }

                    Console.WriteLine();
                    Console.WriteLine("  WARNING: auto labels derive from headings and titles, so the Title and Section");
                    Console.WriteLine("  views score their own positive near 1.0. Any gain measured here rewards relearning");
                    Console.WriteLine("  the label generator; treat it as a leakage diagnostic, not retrieval quality.");

                    var grades = derived.Select(entry => new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                        entry.Query, entry.Relevant));
                    return RetrievalLabels.Resolve(LabelKind.Auto, index.Chunks, grades);
                }
        }
    }

    static void PrintLabels(ResolvedLabels labels, int chunks)
    {
        Console.WriteLine();
        Console.WriteLine($"  Labels:       {labels.Description}");
        Console.WriteLine($"  Resolved:     {labels.MatchedLabelRows}/{labels.TotalLabelRows} rows ({labels.Coverage:P1}), " +
            $"{labels.QueryCount} queries against {chunks} chunks");
        if (!labels.IsComplete)
            Console.WriteLine("  Note:         unresolved rows belong to chunks this corpus no longer produces.");
    }

    static void PrintTable(IReadOnlyList<RankerResult> results)
    {
        Console.WriteLine();
        Console.WriteLine("  Ranker                   P@k    R@k    MRR    nDCG@k  Queries");
        Console.WriteLine("  ----------------------   -----  -----  -----  -------  -------");
        foreach (var result in results)
        {
            Console.WriteLine(
                $"  {result.Name,-22}   {result.Mean.Precision:0.000}  {result.Mean.Recall:0.000}  " +
                $"{result.Mean.Mrr:0.000}  {result.Mean.Ndcg:0.0000}   {result.QueryCount}");
        }
    }

    static void PrintTraining(string name, TrainingReport report)
    {
        if (report.History.Count == 0)
        {
            Console.WriteLine($"  {name}: no usable training query");
            return;
        }

        var early = report.StoppedEarly ? ", early stop" : "";
        Console.WriteLine(
            $"  {name}: best epoch {report.BestEpoch}/{report.History.Count} " +
            $"val nDCG {report.BestValidationNdcg:0.0000}, train loss {report.BestTrainLoss:0.0000} " +
            $"({report.OptimizerSteps} steps, {report.QueriesUsed} queries{early})");
    }

    /// <summary>
    /// Maximum absolute difference between an untrained model's fused scores and the #914 fusion.
    /// This is the parity property stated as a number in the product, not only in a test.
    /// </summary>
    static double InitialParity(IFusionRanker model, IReadOnlyList<TrainingQuery> queries, int topK)
    {
        var worst = 0d; TrainingQuery? worstQuery = null; int worstR = 0;
        foreach (var query in queries)
        {
            if (!query.IsUsable)
                continue;

            var learned = RetrievalEvaluation.Scores(model, query);
            var baseline = RetrievalEvaluation.FixedScores(query, FusionWeights.Default);
            for (var row = 0; row < learned.Length && row < baseline.Length; row++)
            {
                var diff = Math.Abs(learned[row] - baseline[row]);
                if (diff > worst) { worst = diff; worstQuery = query; worstR = row; }
            }
        }
        return worst;
    }

    static float[] MeanGateWeights(GatedFusionModel model, IReadOnlyList<TrainingQuery> queries)
    {
        var mean = new float[TrainingSetBuilder.ViewCount];
        var count = 0;
        foreach (var query in queries)
        {
            if (!query.IsUsable)
                continue;

            var weights = model.WeightsFor(query);
            for (var view = 0; view < mean.Length; view++)
                mean[view] += weights[view];
            count++;
        }

        if (count == 0)
            return mean;

        for (var view = 0; view < mean.Length; view++)
            mean[view] /= count;

        return mean;
    }
    static float[] MeanGateWeightsSilk(SilkRetrievalTrainer model, IReadOnlyList<TrainingQuery> queries)
    {
        var mean = new float[TrainingSetBuilder.ViewCount];
        var count = 0;
        foreach (var query in queries)
        {
            if (!query.IsUsable) continue;
            var weights = model.WeightsFor(query);
            for (var view = 0; view < mean.Length; view++) mean[view] += weights[view];
            count++;
        }
        if (count == 0) return mean;
        for (var view = 0; view < mean.Length; view++) mean[view] /= count;
        return mean;
    }


    static string FormatWeights(IReadOnlyList<float> weights) =>
        string.Join(", ", Enumerable.Range(0, weights.Count)
            .Select(view => $"{(ViewSpace)view}={weights[view]:0.000}"));

    static LabelKind ParseLabelKind(string? value, string repoRoot)
    {
        if (string.IsNullOrWhiteSpace(value))
            return File.Exists(JudgePath(repoRoot)) ? LabelKind.Judge : LabelKind.Fixture;

        return value.Trim().ToLowerInvariant() switch
        {
            "judge" => LabelKind.Judge,
            "fixture" => LabelKind.Fixture,
            "auto" => LabelKind.Auto,
            _ => throw new ArgumentException($"Unknown --labels value '{value}'. Expected judge, fixture or auto.")
        };
    }

    static GateFeatureKind? ParseGate(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return GateFeatureKind.ScoreStats;

        return value.Trim().ToLowerInvariant() switch
        {
            "stats" => GateFeatureKind.ScoreStats,
            "embed" => GateFeatureKind.QueryEmbedding,
            "none" => null,
            _ => throw new ArgumentException($"Unknown --gate value '{value}'. Expected stats, embed or none.")
        };
    }

    static LossKind ParseLoss(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return LossKind.Listwise;

        return value.Trim().ToLowerInvariant() switch
        {
            "listwise" => LossKind.Listwise,
            "pairwise" => LossKind.Pairwise,
            _ => throw new ArgumentException($"Unknown --loss value '{value}'. Expected listwise or pairwise.")
        };
    }

    static string KindName(GateFeatureKind kind) =>
        kind == GateFeatureKind.ScoreStats ? "stats" : "embed";

    static string JudgePath(string repoRoot) =>
        Path.Combine(repoRoot, "samples", "data", "nivarachat", "eval", "relevance.tsv");

    static string FixturePath(string repoRoot) =>
        Path.Combine(repoRoot, "samples", "NivaraChat", "relevance-fixture.tsv");

    static string QueriesPath(ModeContext ctx, string repoRoot)
    {
        if (!string.IsNullOrWhiteSpace(ctx.Queries))
            return ctx.Queries;

        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "queries.tsv"),
            Path.Combine(repoRoot, "samples", "NivaraChat", "queries.tsv")
        };

        return candidates.FirstOrDefault(File.Exists) ?? candidates[0];
    }

    static string ResolveReportPath(string? path, string repoRoot)
    {
        var directory = Path.Combine(repoRoot, "samples", "data", "nivarachat", "learn");
        if (string.IsNullOrWhiteSpace(path))
            return Path.Combine(directory, "rag-learn-report.md");
        if (Path.IsPathRooted(path))
            return path;

        return Path.Combine(directory, path);
    }

    static void WriteReport(
        string path,
        ResolvedLabels labels,
        TrainingOptions options,
        int topK,
        int candidates,
        QuerySplit split,
        int trainCount,
        double parity,
        IReadOnlyList<RankerResult> results,
        IReadOnlyList<PairedComparison> comparisons,
        IReadOnlyList<float> learnedWeights,
        IFusionRanker? gated,
        IReadOnlyList<TrainingQuery> test,
        GateFeatureKind? gateKind)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var content = new StringBuilder();
        content.AppendLine("# Differentiable retrieval (#915)");
        content.AppendLine();
        content.AppendLine($"- Label source: {labels.Description}");
        content.AppendLine($"- Label coverage: {labels.MatchedLabelRows}/{labels.TotalLabelRows} rows ({labels.Coverage:P1}), {labels.QueryCount} queries");
        content.AppendLine($"- Split: {trainCount} train / {split.Validation.Count} validation / {split.Test.Count} test (seed {options.Seed})");
        content.AppendLine($"- Candidate pool: union of each view's top-{candidates} plus every labeled positive");
        content.AppendLine($"- Loss: {options.Loss}; lr {options.LearningRate}; epochs {options.Epochs}; weight decay {options.WeightDecay}");
        content.AppendLine($"- Init parity: max |learned - fixed| = {parity.ToString("0.0000000", CultureInfo.InvariantCulture)} (tolerance {ParityTolerance.ToString("0.0E+0", CultureInfo.InvariantCulture)})");
        content.AppendLine();
        content.AppendLine($"## Held-out test metrics (K = {topK})");
        content.AppendLine();
        content.AppendLine("| Ranker | P@K | R@K | MRR | nDCG@K | Queries |");
        content.AppendLine("| --- | --- | --- | --- | --- | --- |");
        foreach (var result in results)
        {
            content.AppendLine(
                $"| {result.Name} | {result.Mean.Precision:0.000} | {result.Mean.Recall:0.000} | " +
                $"{result.Mean.Mrr:0.000} | {result.Mean.Ndcg:0.0000} | {result.QueryCount} |");
        }

        content.AppendLine();
        content.AppendLine("## Paired comparisons (challenger vs fixed(default), per-query nDCG)");
        content.AppendLine();
        foreach (var comparison in comparisons)
            content.AppendLine($"- {comparison.Summary}");

        content.AppendLine();
        content.AppendLine("## Learned weights");
        content.AppendLine();
        var fixedWeights = new[] { FusionWeights.Default.Content, FusionWeights.Default.Title, FusionWeights.Default.Section, FusionWeights.Default.Code };
        content.AppendLine($"- fixed: {FormatWeights(fixedWeights)}");
        content.AppendLine($"- learned-global: {FormatWeights(learnedWeights)}");
        if (gated is not null && gateKind is not null)
        {
            float[] gatedMean;
            if (gated is GatedFusionModel gm)
                gatedMean = MeanGateWeights(gm, test);
            else
                gatedMean = MeanGateWeightsSilk(((SilkFusionRanker)gated).Trainer, test);
            content.AppendLine($"- learned-gated({KindName(gateKind.Value)}): {FormatWeights(gatedMean)} (mean over test queries)");
        }

        content.AppendLine();
        content.AppendLine("## Reading this report");
        content.AppendLine();
        content.AppendLine("- Every ranker reranks the same candidate pool, so this is a reranking comparison, not a retrieval one.");
        content.AppendLine("- Both trainable models start numerically equal to the fixed baseline (the parity line above), so a delta is attributable to learning and not to a different fusion formula.");
        content.AppendLine("- The test split is scored once, after training; checkpoint selection uses the validation split only.");
        content.AppendLine("- Acceptance is a fair comparison, not a guaranteed win. A learned model that loses on held-out queries is a valid outcome.");

        File.WriteAllText(path, content.ToString());
    }
}
