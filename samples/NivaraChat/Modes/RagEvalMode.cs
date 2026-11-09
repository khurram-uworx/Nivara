using NivaraChat.Helpers;
using OllamaSharp;
using System.Diagnostics;
using System.Text.Json;

namespace NivaraChat.Modes;

public static class RagEvalMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat - RAG Evaluation (#943) ===\n");

        var repoRoot = ModeHelpers.GetRepoRoot();
        var queriesCandidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "queries.tsv"),
            Path.Combine(repoRoot, "samples", "data", "nivarachat", "queries.tsv"),
            Path.Combine(repoRoot, "samples", "NivaraChat", "queries.tsv")
        };
        var queriesPath = ctx.Queries ?? queriesCandidates.FirstOrDefault(File.Exists) ?? queriesCandidates[0];
        var labelsPath = Path.Combine(repoRoot, "samples", "data", "nivarachat", "eval", "relevance.tsv");
        var reportPath = ctx.ReportPath is null ? null : ResolveReportPath(ctx.ReportPath, repoRoot);

        var buildSw = Stopwatch.StartNew();
        using var index = await MultiViewIndex.BuildAsync(ctx);
        buildSw.Stop();
        if (index is null)
            return;

        var queriesFileExisted = File.Exists(queriesPath);
        if (!queriesFileExisted)
            Console.WriteLine($"queries.tsv not found at {queriesPath}; deriving from corpus chunks (use --regenerate queries to write it explicitly).");

        List<EvalQuery> queries;
        try
        {
            queries = EvalGroundTruth.Load(queriesPath).ToList();
        }
        catch
        {
            queries = new List<EvalQuery>();
        }

        if (queries.Count == 0)
        {
            queries = EvalGroundTruth.DeriveFromChunks(index.Chunks).ToList();
            try { EvalGroundTruth.Save(queriesPath, queries); } catch { }
        }

        Console.WriteLine($"Loaded {queries.Count} queries from {queriesPath}");

        OllamaApiClient? client = null;
        if (ctx.UseOllama)
        {
            Console.WriteLine($"Connecting to Ollama at {ctx.OllamaUrl} (model: {ctx.ModelName})...");
            client = new OllamaApiClient(new Uri(ctx.OllamaUrl), ctx.ModelName);
            Console.WriteLine("Ollama connected.\n");
        }
        else if (!ctx.Offline)
        {
            Console.WriteLine("This mode needs --ollama (or --offline with cached labels).");
            return;
        }

        var labels = RagEvalLabels.Load(labelsPath);

        var runs = Math.Max(1, ctx.Runs);
        var topK = ctx.TopK > 0 ? ctx.TopK : 10;

        var configs = new (string Name, FusionWeights Weights)[]
        {
            ("fused", FusionWeights.Default),
            ("content-only", new FusionWeights(1f, 0f, 0f, 0f)),
            ("no-content", new FusionWeights(0f, 0.2f, 0.2f, 0.1f)),
            ("no-title", new FusionWeights(0.5f, 0f, 0.2f, 0.1f)),
            ("no-section", new FusionWeights(0.5f, 0.2f, 0f, 0.1f)),
            ("no-code", new FusionWeights(0.5f, 0.2f, 0.2f, 0f)),
        };

        var results = new List<QueryResult>();

        foreach (var q in queries)
        {
            // Retrieve per config.
            var retrievedByConfig = new Dictionary<string, IReadOnlyList<MultiViewHit>>(StringComparer.Ordinal);
            long latencySum = 0;
            var latencyCount = 0;

            foreach (var (name, weights) in configs)
            {
                var (hits, ms) = index.Retrieve(q.Query, weights, topK);
                retrievedByConfig[name] = hits;
                latencySum += ms;
                latencyCount++;
            }

            // Candidate chunks to grade.
            var candidateIds = new HashSet<string>(StringComparer.Ordinal);
            foreach (var hits in retrievedByConfig.Values)
                foreach (var h in hits)
                    candidateIds.Add(h.Chunk.ChunkId);

            if (ctx.GradeAll)
            {
                foreach (var c in index.Chunks)
                    candidateIds.Add(c.ChunkId);
            }
            else
            {
                var nonRetrieved = index.Chunks.Select(c => c.ChunkId).Where(id => !candidateIds.Contains(id)).ToList();
                var rng = new Random(42);
                foreach (var id in nonRetrieved.OrderBy(_ => rng.Next()).Take(20))
                    candidateIds.Add(id);
            }

            // Grade missing candidates.
            if (!labels.TryGetValue(q.Query, out var labelMap))
            {
                labelMap = new Dictionary<string, LabelResult>(StringComparer.Ordinal);
                labels[q.Query] = labelMap;
            }

            var toGrade = candidateIds.Where(id => !labelMap.ContainsKey(id)).ToList();
            if (toGrade.Count > 0)
            {
                if (ctx.Offline)
                {
                    Console.WriteLine($"  Offline: missing labels for '{q.Query}' ({toGrade.Count}); skipping.");
                }
                else
                {
                    var passages = toGrade
                        .Select(id => (id, index.Chunks.FirstOrDefault(c => c.ChunkId == id)?.Content ?? string.Empty))
                        .Where(p => !string.IsNullOrWhiteSpace(p.Item2))
                        .ToList();

                    if (passages.Count > 0 && client is not null)
                    {
                        var graded = await RagEvalJudge.GradeChunksAsync(client, q.Query, passages, runs, ctx.Temperature);
                        foreach (var (id, res) in graded)
                            labelMap[id] = res;
                        RagEvalLabels.Save(labelsPath, labels);
                    }
                }
            }

            // Build grades dictionary for metrics (grade>0 relevant).
            var grades = candidateIds.ToDictionary(
                id => id,
                id => labelMap.TryGetValue(id, out var r) ? r.Grade : 0,
                StringComparer.Ordinal);

            var relevantSet = new HashSet<string>(grades.Where(kv => kv.Value > 0).Select(kv => kv.Key), StringComparer.Ordinal);

            var perConfig = new Dictionary<string, ConfigMetrics>(StringComparer.Ordinal);
            foreach (var (name, weights) in configs)
            {
                var retrievedIds = retrievedByConfig[name].Select(h => h.Chunk.ChunkId).ToList();
                perConfig[name] = new ConfigMetrics(
                    RagEvalMetrics.PrecisionAtK(retrievedIds, relevantSet, topK),
                    RagEvalMetrics.RecallAtK(retrievedIds, relevantSet, topK),
                    RagEvalMetrics.Mrr(retrievedIds, relevantSet),
                    RagEvalMetrics.NdcgAtK(retrievedIds, grades, topK));
            }

            results.Add(new QueryResult(q.Query, retrievedByConfig, perConfig, grades, (double)latencySum / Math.Max(1, latencyCount)));

            if (results.Count % 50 == 0 || results.Count == queries.Count)
                Console.WriteLine($"  Evaluated {results.Count}/{queries.Count} queries...");
        }

        // Aggregate.
        Console.WriteLine();
        Console.WriteLine("  Config          P@k    R@k    MRR    nDCG@k");
        Console.WriteLine("  ------------    -----  -----  -----  -------");
        foreach (var (name, _) in configs)
        {
            var p = RagEvalMetrics.Mean(results.Select(r => r.Metrics[name].Precision).ToList());
            var r = RagEvalMetrics.Mean(results.Select(r => r.Metrics[name].Recall).ToList());
            var m = RagEvalMetrics.Mean(results.Select(r => r.Metrics[name].Mrr).ToList());
            var n = RagEvalMetrics.Mean(results.Select(r => r.Metrics[name].Ndcg).ToList());
            Console.WriteLine($"  {name,-14}  {p:0.000}  {r:0.000}  {m:0.000}  {n:0.000}");
        }

        // Win/loss/tie fused vs content-only.
        var fusedScores = results.Select(r => r.Metrics["fused"].Ndcg).ToList();
        var contentScores = results.Select(r => r.Metrics["content-only"].Ndcg).ToList();
        var (wins, losses, ties) = RagEvalMetrics.WinLossTie(fusedScores, contentScores);
        Console.WriteLine();
        Console.WriteLine($"  fused vs content-only: wins={wins}, losses={losses}, ties={ties}");
        Console.WriteLine($"  sign-test p={RagEvalMetrics.SignTestPValue(wins, losses):0.0000}, Cohen's d={RagEvalMetrics.CohensD(fusedScores, contentScores):0.000}");

        // Diagnostics (fused top-k across queries).
        var allHits = results.SelectMany(r => r.Retrieved["fused"]).ToList();
        var bestCounts = allHits.GroupBy(h => h.BestSpace?.ToString() ?? "none").ToDictionary(g => g.Key, g => g.Count());
        Console.WriteLine();
        Console.WriteLine("  BestSpace distribution:");
        foreach (var (space, count) in bestCounts)
            Console.WriteLine($"    {space}: {count}");

        var agree = allHits.Count(h => h.MaxDivergence is { } d && d <= MultiViewExplorer.AgreeBound);
        var mild = allHits.Count(h => h.MaxDivergence is { } d && d > MultiViewExplorer.AgreeBound && d <= MultiViewExplorer.MildBound);
        var strong = allHits.Count(h => h.MaxDivergence is { } d && d > MultiViewExplorer.MildBound);
        Console.WriteLine($"  Agreement: agree={agree}, mild={mild}, strong={strong}");

        // Cost reporting.
        var cacheDir = Path.Combine(repoRoot, "samples", "data", "nivarachat", "rag_multiview_cache");
        var indexSize = Directory.Exists(cacheDir)
            ? Directory.GetFiles(cacheDir, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length)
            : 0L;
        var uniqueTexts = new HashSet<string>(StringComparer.Ordinal);
        foreach (var chunk in index.Chunks)
        {
            uniqueTexts.Add(chunk.Content);
            uniqueTexts.Add(chunk.Title);
            uniqueTexts.Add(chunk.Section);
            if (chunk.Code is not null)
                uniqueTexts.Add(chunk.Code);
        }
        var avgLatency = results.Count == 0 ? 0d : results.Average(r => r.LatencyMs);
        Console.WriteLine();
        Console.WriteLine("  Cost:");
        Console.WriteLine($"    build time:    {buildSw.ElapsedMilliseconds} ms");
        Console.WriteLine($"    unique texts:  {uniqueTexts.Count}");
        Console.WriteLine($"    chunks:        {index.Chunks.Count}");
        Console.WriteLine($"    index size:    {indexSize} bytes");
        Console.WriteLine($"    avg query ms:  {avgLatency:0.0}");

        // Verbose per-query movement.
        if (ctx.Verbose)
        {
            Console.WriteLine();
            Console.WriteLine("  Per-query movement (fused vs content-only top-1):");
            foreach (var r in results)
            {
                var fusedTop = r.Retrieved["fused"].FirstOrDefault()?.Chunk.ChunkId ?? "-";
                var contentTop = r.Retrieved["content-only"].FirstOrDefault()?.Chunk.ChunkId ?? "-";
                Console.WriteLine($"    {Truncate(r.Query, 60)}: fused={fusedTop} content={contentTop}");
            }
        }

        if (reportPath is not null)
        {
            WriteReport(reportPath, results, configs);
            Console.WriteLine($"  Report written to {reportPath}");
        }

        Console.WriteLine();
        Console.WriteLine("Evaluation complete.");
    }

    static string Truncate(string text, int max) =>
        text.Length <= max ? text : text[..(max - 3)] + "...";

    static void WriteReport(string path, IReadOnlyList<QueryResult> results, (string Name, FusionWeights Weights)[] configs)
    {
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        var payload = new
        {
            Results = results.Select(r => new
            {
                r.Query,
                Metrics = r.Metrics.ToDictionary(k => k.Key, v => new
                {
                    v.Value.Precision,
                    v.Value.Recall,
                    v.Value.Mrr,
                    v.Value.Ndcg,
                }),
                FusedTop = r.Retrieved["fused"].FirstOrDefault()?.Chunk.ChunkId,
                ContentTop = r.Retrieved["content-only"].FirstOrDefault()?.Chunk.ChunkId,
            }),
        };

        var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }

    static string ResolveReportPath(string path, string repoRoot)
    {
        if (Path.IsPathRooted(path))
            return path;

        var directory = Path.Combine(repoRoot, "samples", "data", "nivarachat");
        return Path.Combine(directory, path);
    }
}

sealed record ConfigMetrics(double Precision, double Recall, double Mrr, double Ndcg);

sealed record QueryResult(
    string Query,
    IReadOnlyDictionary<string, IReadOnlyList<MultiViewHit>> Retrieved,
    IReadOnlyDictionary<string, ConfigMetrics> Metrics,
    IReadOnlyDictionary<string, int> Grades,
    double LatencyMs);
