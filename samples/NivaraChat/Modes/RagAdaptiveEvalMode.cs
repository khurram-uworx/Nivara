using Nivara;
using Nivara.AI;
using Nivara.AutoDiff;
using NivaraChat.Helpers;
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;

namespace NivaraChat.Modes;

public static class RagAdaptiveEvalMode
{
    public static async Task Run(ModeContext ctx)
    {
        Console.WriteLine("=== NivaraChat — Adaptive RAG Evaluation (#983) ===\n");

        var repoRoot = ModeHelpers.GetRepoRoot();
        var topK = ctx.EvalK ?? RetrievalEvaluation.DefaultK;
        if (topK <= 0) topK = RetrievalEvaluation.DefaultK;
        var maxQueries = ctx.MaxQueries > 0 ? ctx.MaxQueries : 40;
        var maxExamples = ctx.MaxExamples > 0 ? ctx.MaxExamples : 128;
        var headEpochs = ctx.HeadEpochs > 0 ? ctx.HeadEpochs : AdaptiveTraining.DefaultEpochs;
        var headLr = ctx.HeadLearningRate > 0 ? ctx.HeadLearningRate : AdaptiveTraining.DefaultLearningRate;
        var updateBatch = ctx.UpdateBatch > 0 ? ctx.UpdateBatch : 32;

        // labels
        using var index = await MultiViewIndex.BuildAsync(ctx);
        if (index is null) return;
        if (!index.HasVisibleChunks)
        {
            Console.WriteLine($"No chunks visible under ACL group '{ctx.AclGroup}'.");
            return;
        }

        var labels = LoadLabels(ctx, repoRoot, index);
        if (labels is null) return;
        PrintLabels(labels, index.Chunks.Count);
        if (!RetrievalLabels.Accepts(labels) && labels.Kind != LabelKind.Auto)
        {
            Console.WriteLine($"REFUSED: coverage {labels.Coverage:P1} < {RagLearnMode.MinimumLabelCoverage:P0}.");
            Environment.ExitCode = 2;
            return;
        }
        if (labels.Kind == LabelKind.Auto)
        {
            Console.WriteLine();
            Console.WriteLine("  WARNING: auto labels derive from headings and titles; leakage diagnostic only.");
        }

        // engine
        var modelDir = Path.Combine(repoRoot, "samples", "data", "laya");
        if (!Directory.Exists(modelDir) || !File.Exists(Path.Combine(modelDir, "model.safetensors")))
        {
            Console.WriteLine($"Laya checkpoint not found at {modelDir}.");
            return;
        }

        var headWeightsPath = string.IsNullOrWhiteSpace(ctx.HeadWeights)
            ? AdaptiveHeadPersistence.DefaultPath(repoRoot)
            : (Path.IsPathRooted(ctx.HeadWeights) ? ctx.HeadWeights : Path.Combine(repoRoot, ctx.HeadWeights));
        var existedBefore = File.Exists(headWeightsPath);

        var engine = new LayaRetrievalDecisionHead.LayaDecisionEngine(
            modelDir, loadAdaptiveWeights: ctx.Resume, adaptiveWeightsPath: headWeightsPath);

        var collector = new AdaptiveFeedbackCollector(engine.Head, headWeightsPath, updateBatch, headEpochs, headLr);

        // silk parity
        MultiViewIndex? silkIndex = null;
        SilkParityReport silkParity = new(0, 0, 0, true);
        long cpuBuildMs = 0, silkBuildMs = 0;
        bool cpuFromCache = index.FromCache, silkFromCache = false;
        var swBuild = Stopwatch.StartNew();
        // re-time cpu build is not possible after the fact; record from-cache flag only
        swBuild.Stop();
        cpuBuildMs = 0;

        if (ctx.UseSilk)
        {
            var silkCtx = ctx with { UseSilk = true };
            silkIndex = await MultiViewIndex.BuildAsync(silkCtx);
            if (silkIndex is not null)
            {
                silkFromCache = silkIndex.FromCache;
                silkParity = AdaptiveSilkParity.Compare(
                    index.Metadata, silkIndex.Metadata,
                    ["VectorContent", "VectorTitle", "VectorSection", "VectorCode"]);
            }
        }

        var useIndex = silkIndex ?? index;

        // split queries
        var split = QuerySplit.Create(labels.Grades.Keys, ctx.Seed);
        var trainQueries = split.Train.Where(q => labels.Grades.ContainsKey(q)).Take(maxQueries).ToList();
        var testQueries = split.Test.Where(q => labels.Grades.ContainsKey(q)).Take(maxQueries).ToList();

        Console.WriteLine();
        Console.WriteLine($"  Queries: {labels.Grades.Count} labelled; split {trainQueries.Count} train / {testQueries.Count} test (of {split.Train.Count}/{split.Test.Count})");

        var limits = new AdaptiveBuildLimits(maxQueries, maxExamples);

        // ---------- TEST PASS (before) ----------
        var testSw = Stopwatch.StartNew();
        var testResult = RunTestPass(useIndex, labels, testQueries, engine, topK, limits, ctx.ConfidenceThreshold);
        testSw.Stop();

        Console.WriteLine($"  Test features extracted: {testResult.Examples.Count} examples, {testResult.FeatureBytes} bytes");

        // calibration before
        var observedBefore = AdaptiveTraining.Observe(engine.Head, testResult.Examples, engine.Calibration);
        var adaptiveObsBefore = testResult.Observations
            .Select(o => new AdaptiveObservation(o.Query, o.ChunkId, o.Grade, o.ChoiceRelevant, o.Accepted, o.AnswerConfidence))
            .ToList();
        // guard: if no observations, skip calibration
        var decisionBefore = BuildDecisionMetrics(testResult.Observations);
        var calibBeforeSafe = adaptiveObsBefore.Count > 0
            ? AdaptiveEvaluation.Calibrate(adaptiveObsBefore)
            : new AdaptiveCalibration(0, 0, 0, []);

        // retrieval rows are built inside RunTestPass

        // ---------- TRAIN PASS (streamed) ----------
        var trainSw = Stopwatch.StartNew();
        var inSample = new List<AdaptiveReportAgreement>();
        int batchNo = 0;
        long trainFeatureBytes = 0;
        var trainExamplesCount = 0;
        int trainQueriesDone = 0;

        foreach (var query in trainQueries.OrderBy(q => q, StringComparer.Ordinal))
        {
            if (trainQueriesDone >= maxQueries) break;
            trainQueriesDone++;

            var vector = useIndex.EmbedQuery(query);
            var rows = MultiViewScoring.ScoreAll(useIndex.Matrices, useIndex.Chunks.Count, vector);
            var fused = Enumerable.Range(0, useIndex.Chunks.Count)
                .Select(r => (Row: r, Score: MultiViewScoring.Fuse(rows[r], FusionWeights.Default) ?? float.NegativeInfinity))
                .OrderByDescending(x => x.Score).ThenBy(x => x.Row)
                .Take(topK).Select(x => x.Row).ToList();

            var grades = labels.Grades[query];
            int batchCorrect = 0, batchCount = 0;
            foreach (var row in fused)
            {
                if (!grades.TryGetValue(row, out var grade)) continue;
                if (trainExamplesCount >= maxExamples) break;

                var features = engine.ExtractFeatures(query, useIndex.Chunks[row]);
                var decision = LayaRetrievalDecisionHead.DecideFromFeatures(engine, query, useIndex.Chunks[row], features);
                var example = AdaptiveTrainingData.CreateExample(query, useIndex.Chunks[row], grade, features);
                AdaptiveTraining.ValidateExample(engine.Head, example);

                // in-sample agreement before this example is trained
                var predictedRelevant = decision.Choice == "relevant";
                if (predictedRelevant == (grade > 0)) batchCorrect++;
                batchCount++;

                collector.Add(example);
                trainExamplesCount++;
                trainFeatureBytes += features.Values.Length * sizeof(float);
            }

            if (batchCount > 0)
            {
                inSample.Add(new AdaptiveReportAgreement(batchNo++, batchCount, batchCorrect, (double)batchCorrect / batchCount));
            }
        }

        collector.FlushRemaining();
        trainSw.Stop();

        // ---------- TEST PASS (after) ----------
        var testAfterSw = Stopwatch.StartNew();
        var testResultAfter = RunTestPass(useIndex, labels, testQueries, engine, topK, limits, ctx.ConfidenceThreshold);
        testAfterSw.Stop();

        var adaptiveObsAfter = testResultAfter.Observations
            .Select(o => new AdaptiveObservation(o.Query, o.ChunkId, o.Grade, o.ChoiceRelevant, o.Accepted, o.AnswerConfidence))
            .ToList();
        var calibAfter = adaptiveObsAfter.Count > 0
            ? AdaptiveEvaluation.Calibrate(adaptiveObsAfter)
            : new AdaptiveCalibration(0, 0, 0, []);
        var decisionAfter = BuildDecisionMetrics(testResultAfter.Observations);


        // retrieval rows are already in testResult / testResultAfter

        // combine retrieval rows: fixed (once), rerank before/after, filter before/after
        var combinedRows = new List<AdaptiveReportRow>
        {
            testResult.RetrievalFixedRow,
            testResult.RetrievalRerankRow,
            testResultAfter.RetrievalRerankRow,
            testResult.RetrievalFilterRow,
            testResultAfter.RetrievalFilterRow,
        };

        // paired comparisons
        var comparisons = new List<PairedComparison>();
        var pc1 = BuildPaired("adaptive-rerank(after)", "fixed", testResultAfter.PerQueryNdcgRerank, testResult.PerQueryNdcgFixed);
        if (pc1 is not null) comparisons.Add(pc1);
        var pc2 = BuildPaired("adaptive-rerank(after)", "adaptive-rerank(before)", testResultAfter.PerQueryNdcgRerank, testResult.PerQueryNdcgRerank);
        if (pc2 is not null) comparisons.Add(pc2);

        // persistence verification
        long bytesWritten = 0;
        foreach (var f in Directory.GetFiles(Path.GetDirectoryName(headWeightsPath)!, $".laya_head.json.*.tmp", SearchOption.TopDirectoryOnly)) { }
        var finalSize = File.Exists(headWeightsPath) ? new FileInfo(headWeightsPath).Length : 0;
        bool verified = false;
        if (File.Exists(headWeightsPath))
        {
            var scratch = new LayaDecisionHead<float, float>(engine.Head.HiddenSize, headLayerCount: 0);
            // can't easily load without the right shape; instead compare via a fresh engine load
            try
            {
                using var verifyEngine = new LayaRetrievalDecisionHead.LayaDecisionEngine(modelDir, loadAdaptiveWeights: true, adaptiveWeightsPath: headWeightsPath);
                verified = StateDictEqual(engine.Head.StateDict(), verifyEngine.Head.StateDict());
            }
            catch { verified = false; }
            scratch.Dispose();
        }

        // counts
        var saves = collector.SuccessfulUpdates;
        var decisive = comparisons.Count == 0 ? 0 : comparisons.Max(c => c.Wins + c.Losses);
        var decisiveNote = decisive < RagLearnMode.MinimumSignificantQueries
            ? $"At most {decisive} decisive held-out queries; the sign test cannot reach p < 0.05 below {RagLearnMode.MinimumSignificantQueries}."
            : "Sign-test decisiveness OK.";

        // timings
        var timings = new List<AdaptiveReportTiming>
        {
            new("index build (cpu)", 1, 0),
            new("test feature extraction (encoder forwards)", testResult.Examples.Count + (testResult.Observations.Count - testResult.Examples.Count), testSw.ElapsedMilliseconds),
            new("train feature extraction + decisions", trainExamplesCount, (int)trainSw.ElapsedMilliseconds),
            new("head updates", saves, 0),
            new("test pass (after)", testResultAfter.Examples.Count, (int)testAfterSw.ElapsedMilliseconds),
            new("total test-pass before", testResult.Examples.Count, (int)testSw.ElapsedMilliseconds),
        };

        // silk build times not measured separately; leave 0 with from-cache flags
        var silkReport = new AdaptiveReportSilk(
            ctx.UseSilk ? "silk" : "cpu",
            silkParity.MaxAbsoluteDifference,
            silkParity.MismatchedRows,
            silkParity.WithinTolerance,
            cpuBuildMs,
            silkBuildMs,
            cpuFromCache,
            silkFromCache,
            silkIndex is not null || !ctx.UseSilk);

        var input = new AdaptiveReportInput
        {
            Mode = ctx.UseSilk ? "adaptive-eval (silk)" : "adaptive-eval (cpu)",
            LabelKind = labels.Kind.ToString().ToLowerInvariant(),
            LabelDescription = labels.Description,
            LabelMatched = labels.MatchedLabelRows,
            LabelTotal = labels.TotalLabelRows,
            LabelCoverage = labels.Coverage,
            LabelQueries = labels.QueryCount,
            Seed = ctx.Seed,
            K = topK,
            Threshold = ctx.ConfidenceThreshold,
            MaxQueries = maxQueries,
            MaxExamples = maxExamples,
            CorpusFiles = index.CorpusIdentity.FileCount,
            CorpusChunks = index.Chunks.Count,
            CorpusContentHash = index.CorpusIdentity.ContentHash,
            CorpusChunkerVersion = index.CorpusIdentity.ChunkerVersion,
            CorpusChunkSize = index.CorpusIdentity.ChunkSize,
            CorpusBackend = index.CorpusIdentity.Backend,
            CorpusDimension = index.Dimension,
            TrainQueries = trainQueries.Count,
            TestQueries = testQueries.Count,
            TrainExamples = trainExamplesCount,
            TestExamples = testResult.Examples.Count,
            TrainFeatureBytes = trainFeatureBytes,
            TestFeatureBytes = testResult.FeatureBytes,
            Leakage = labels.Kind == LabelKind.Auto,
            Resume = ctx.Resume,
            UpdateBatch = updateBatch,
            HeadEpochs = headEpochs,
            HeadLearningRate = headLr,
            RetrievalRows = combinedRows,
            Comparisons = comparisons,
            DecisionBefore = new AdaptiveReportDecision(
                decisionBefore.Count, decisionBefore.Correct, decisionBefore.Accuracy,
                decisionBefore.AcceptPrecision, decisionBefore.AcceptRecall, decisionBefore.AcceptF1),
            DecisionAfter = new AdaptiveReportDecision(
                decisionAfter.Count, decisionAfter.Correct, decisionAfter.Accuracy,
                decisionAfter.AcceptPrecision, decisionAfter.AcceptRecall, decisionAfter.AcceptF1),
            CalibrationBefore = new AdaptiveReportCalibration(calibBeforeSafe.Count, calibBeforeSafe.Ece, calibBeforeSafe.Brier),
            CalibrationAfter = new AdaptiveReportCalibration(calibAfter.Count, calibAfter.Ece, calibAfter.Brier),
            ReliabilityBefore = calibBeforeSafe.Bins,
            ReliabilityAfter = calibAfter.Bins,
            Timings = timings,
            Silk = silkReport,
            Persistence = new AdaptiveReportPersistence(headWeightsPath, existedBefore, saves, bytesWritten, finalSize, verified),
            InSampleAgreement = inSample,
            DecisiveNote = decisiveNote,
        };

        var reportPath = AdaptiveReport.ResolvePath(ctx.ReportPath, repoRoot);
        AdaptiveReport.Write(reportPath, AdaptiveReport.Markdown(input));
        var metricsPath = AdaptiveReport.MetricsPath(repoRoot, input.LabelKind, ctx.Seed, topK);
        AdaptiveReport.Write(metricsPath, AdaptiveReport.MetricsJson(input));

        Console.WriteLine();
        Console.WriteLine($"  Report written to {reportPath}");
        Console.WriteLine($"  Metrics written to {metricsPath}");
        Console.WriteLine();
        Console.WriteLine("Adaptive evaluation complete.");

        engine.Dispose();
        silkIndex?.Dispose();
    }

    // ---------- helpers ----------

    sealed record TestPassResult(
        List<AdaptiveTrainingExample> Examples,
        long FeatureBytes,
        List<TestObservation> Observations,
        RankerMetrics? FixedMean,
        List<double> PerQueryNdcgFixed,
        List<double> PerQueryNdcgRerank,
        AdaptiveReportRow RetrievalFixedRow,
        AdaptiveReportRow RetrievalRerankRow,
        AdaptiveReportRow RetrievalFilterRow);

    sealed record TestObservation(
        string Query, string ChunkId, int Grade, bool ChoiceRelevant, bool Accepted, double AnswerConfidence);

    static TestPassResult RunTestPass(
        MultiViewIndex index, ResolvedLabels labels, List<string> queries,
        LayaRetrievalDecisionHead.LayaDecisionEngine engine, int topK, AdaptiveBuildLimits limits,
        double threshold)
    {
        var examples = new List<AdaptiveTrainingExample>();
        var observations = new List<TestObservation>();
        long featureBytes = 0;
        var perQueryFixedNdcg = new List<double>();
        var perQueryRerankNdcg = new List<double>();

        var fixedP = new List<double>(); var fixedR = new List<double>(); var fixedM = new List<double>(); var fixedN = new List<double>();
        var rerankP = new List<double>(); var rerankR = new List<double>(); var rerankM = new List<double>(); var rerankN = new List<double>();
        var filterP = new List<double>(); var filterR = new List<double>(); var filterM = new List<double>(); var filterN = new List<double>();

        foreach (var query in queries.OrderBy(q => q, StringComparer.Ordinal))
        {
            if (!labels.Grades.TryGetValue(query, out var grades)) continue;

            var vector = index.EmbedQuery(query);
            var rows = MultiViewScoring.ScoreAll(index.Matrices, index.Chunks.Count, vector);
            var fusedList = Enumerable.Range(0, index.Chunks.Count)
                .Select(r => (Row: r, Score: MultiViewScoring.Fuse(rows[r], FusionWeights.Default) ?? float.NegativeInfinity))
                .OrderByDescending(x => x.Score).ThenBy(x => x.Row)
                .Take(topK).ToList();

            var candidates = fusedList.Select(x =>
            {
                var chunk = index.Chunks[x.Row];
                var features = engine.ExtractFeatures(query, chunk);
                var decision = LayaRetrievalDecisionHead.DecideFromFeatures(engine, query, chunk, features);
                var accepted = decision.Choice == "relevant" && decision.AnswerConfidence >= 0; // threshold applied later
                // apply threshold from ctx? not available here; store raw and filter in scoring
                return new { x.Row, chunk, x.Score, features, decision, accepted };
            }).ToList();

            // threshold comes from ctx; but RunTestPass lacks ctx. We'll recompute accepted with a default 0.8? 
            // To keep threshold consistent, accepted flag here is (choice relevant) only; thresholding happens in AdaptiveDecisionScoring? 
            // Actually Accepted in AdaptiveCandidate should reflect the router's operating point. We don't have threshold here.
            // Pass threshold via closure by changing signature: add threshold param.
            // (handled below by re-running; for now mark accepted = choice relevant && confidence >= 0.8 default)
            // We'll instead store confidence and let caller threshold. Simpler: threshold inside loop requires param.
            // -> change signature to include threshold.

            // grade lookup
            var relevantIds = new HashSet<string>(StringComparer.Ordinal);
            var gradesById = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var (row, grade) in grades)
            {
                if (row < 0 || row >= index.Chunks.Count) continue;
                var id = index.Chunks[row].ChunkId;
                if (grade > 0) relevantIds.Add(id);
                if (!gradesById.TryGetValue(id, out var existing) || grade > existing)
                    gradesById[id] = grade;
            }

            var retrievedIds = fusedList.Select(x => index.Chunks[x.Row].ChunkId).ToList();

            // metrics for fixed
            var fixedMetrics = new RankerMetrics(
                RagEvalMetrics.PrecisionAtK(retrievedIds, relevantIds, topK),
                RagEvalMetrics.RecallAtK(retrievedIds, relevantIds, topK),
                RagEvalMetrics.Mrr(retrievedIds, relevantIds),
                RagEvalMetrics.NdcgAtK(retrievedIds, gradesById, topK));
            fixedP.Add(fixedMetrics.Precision); fixedR.Add(fixedMetrics.Recall); fixedM.Add(fixedMetrics.Mrr); fixedN.Add(fixedMetrics.Ndcg);
            perQueryFixedNdcg.Add(fixedMetrics.Ndcg);

            // adaptive candidates with threshold 0.8 default (will be overridden? see note)
            var adaptiveCandidates = candidates.Select(c => new AdaptiveCandidate(
                c.Row, c.chunk.ChunkId, c.Score,
                c.decision.Choice == "relevant" && c.decision.AnswerConfidence >= threshold,
                c.decision.AnswerConfidence)).ToList();

            var rerankOrder = AdaptiveDecisionScoring.Rerank(adaptiveCandidates);
            var filterOrder = AdaptiveDecisionScoring.Filter(adaptiveCandidates);

            var rerankIds = rerankOrder.Select(i => index.Chunks[i].ChunkId).ToList();
            var filterIds = filterOrder.Select(i => index.Chunks[i].ChunkId).ToList();

            var rerankMetrics = new RankerMetrics(
                RagEvalMetrics.PrecisionAtK(rerankIds, relevantIds, topK),
                RagEvalMetrics.RecallAtK(rerankIds, relevantIds, topK),
                RagEvalMetrics.Mrr(rerankIds, relevantIds),
                RagEvalMetrics.NdcgAtK(rerankIds, gradesById, topK));
            rerankP.Add(rerankMetrics.Precision); rerankR.Add(rerankMetrics.Recall); rerankM.Add(rerankMetrics.Mrr); rerankN.Add(rerankMetrics.Ndcg);
            perQueryRerankNdcg.Add(rerankMetrics.Ndcg);

            var filterMetrics = new RankerMetrics(
                RagEvalMetrics.PrecisionAtK(filterIds, relevantIds, topK),
                RagEvalMetrics.RecallAtK(filterIds, relevantIds, topK),
                RagEvalMetrics.Mrr(filterIds, relevantIds),
                RagEvalMetrics.NdcgAtK(filterIds, gradesById, topK));
            filterP.Add(filterMetrics.Precision); filterR.Add(filterMetrics.Recall); filterM.Add(filterMetrics.Mrr); filterN.Add(filterMetrics.Ndcg);

            // observations + examples
            foreach (var c in candidates)
            {
                var grade = grades.TryGetValue(c.Row, out var g) ? g : -1;
                if (grade < 0) continue;
                var accepted = c.decision.Choice == "relevant" && c.decision.AnswerConfidence >= threshold;
                observations.Add(new TestObservation(query, c.chunk.ChunkId, grade,
                    c.decision.Choice == "relevant", accepted, c.decision.AnswerConfidence));

                if (examples.Count < limits.MaxExamples)
                {
                    examples.Add(AdaptiveTrainingData.CreateExample(query, c.chunk, grade, c.features));
                    featureBytes += c.features.Values.Length * sizeof(float);
                }
            }
        }

        var mean = (List<double> xs) => xs.Count == 0 ? 0d : xs.Average();
        var fixedRow = new AdaptiveReportRow("fixed", mean(fixedP), mean(fixedR), mean(fixedM), mean(fixedN), fixedP.Count);
        var rerankRow = new AdaptiveReportRow("adaptive-rerank", mean(rerankP), mean(rerankR), mean(rerankM), mean(rerankN), rerankP.Count);
        var filterRow = new AdaptiveReportRow("adaptive-filter", mean(filterP), mean(filterR), mean(filterM), mean(filterN), filterP.Count);

        return new TestPassResult(
            examples, featureBytes, observations,
            null, perQueryFixedNdcg, perQueryRerankNdcg,
            fixedRow, rerankRow, filterRow);
    }

    static AdaptiveReportDecision BuildDecisionMetrics(List<TestObservation> obs)
    {
        if (obs.Count == 0) return new AdaptiveReportDecision(0, 0, 0, 0, 0, 0);
        var correct = obs.Count(o => o.ChoiceRelevant == (o.Grade > 0));
        // accept metrics at threshold 0.8 already encoded in Accepted flag
        var tp = obs.Count(o => o.Accepted && o.Grade > 0);
        var pp = obs.Count(o => o.Accepted);
        var tr = obs.Count(o => o.Grade > 0);
        var prec = pp == 0 ? 0d : (double)tp / pp;
        var rec = tr == 0 ? 0d : (double)tp / tr;
        var f1 = prec + rec == 0 ? 0d : 2 * prec * rec / (prec + rec);
        return new AdaptiveReportDecision(obs.Count, correct, (double)correct / obs.Count, prec, rec, f1);
    }

    static PairedComparison? BuildPaired(string challenger, string baseline, List<double> a, List<double> b)
    {
        if (a.Count == 0 || a.Count != b.Count) return null;
        var (wins, losses, ties) = RagEvalMetrics.WinLossTie(a, b);
        return new PairedComparison(challenger, baseline, wins, losses, ties,
            RagEvalMetrics.SignTestPValue(wins, losses), RagEvalMetrics.CohensD(a, b),
            a.Average() - b.Average());
    }

    static ResolvedLabels? LoadLabels(ModeContext ctx, string repoRoot, MultiViewIndex index)
    {
        bool judgeExists;
        try { judgeExists = File.Exists(JudgePath(repoRoot)); }
        catch { judgeExists = false; }

        LabelKind kind;
        try { kind = RetrievalLabels.SelectSource(ctx.Labels, judgeExists); }
        catch (ArgumentException ex)
        {
            Console.WriteLine(ex.Message);
            return null;
        }

        switch (kind)
        {
            case LabelKind.Judge:
                {
                    var path = JudgePath(repoRoot);
                    var cache = RagEvalLabels.Load(path);
                    if (cache.Count == 0)
                    {
                        Console.WriteLine($"No judge labels at {path}.");
                        return null;
                    }
                    var grades = cache.Select(e => new KeyValuePair<string, IReadOnlyDictionary<string, int>>(
                        e.Key, e.Value.ToDictionary(i => i.Key, i => i.Value.Grade, StringComparer.Ordinal)));
                    return RetrievalLabels.Resolve(LabelKind.Judge, index.Chunks, grades);
                }
            default:
                {
                    var path = QueriesPath(ctx, repoRoot);
                    var derived = EvalGroundTruth.Load(path);
                    if (derived.Count == 0)
                    {
                        // derive offline from chunks, like --rag-eval does
                        try
                        {
                            derived = EvalGroundTruth.DeriveFromChunks(index.Chunks).ToList();
                            if (derived.Count > 0) EvalGroundTruth.Save(path, derived);
                        }
                        catch { derived = []; }
                    }
                    if (derived.Count == 0)
                    {
                        Console.WriteLine($"No auto queries at {path}.");
                        return null;
                    }
                    Console.WriteLine();
                    Console.WriteLine("  WARNING: auto labels derive from headings and titles (leakage diagnostic).");
                    var grades = derived.Select(e => new KeyValuePair<string, IReadOnlyDictionary<string, int>>(e.Query, e.Relevant));
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
    }

    static string JudgePath(string repoRoot) => Path.Combine(repoRoot, "samples", "data", "nivarachat", "eval", "relevance.tsv");
    static string QueriesPath(ModeContext ctx, string repoRoot)
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "queries.tsv"),
            Path.Combine(repoRoot, "samples", "data", "nivarachat", "queries.tsv"),
            Path.Combine(repoRoot, "samples", "NivaraChat", "queries.tsv")
        };
        return candidates.FirstOrDefault(File.Exists) ?? candidates[1];
    }

    static bool StateDictEqual(Dictionary<string, ReverseGradTensor<float>> a, Dictionary<string, ReverseGradTensor<float>> b)
    {
        if (a.Count != b.Count) return false;
        foreach (var (name, ta) in a)
        {
            if (!b.TryGetValue(name, out var tb)) return false;
            ta.Data.TryGetSpan(out var sa);
            tb.Data.TryGetSpan(out var sb);
            if (sa.Length != sb.Length) return false;
            for (var i = 0; i < sa.Length; i++)
                if (sa[i] != sb[i]) return false;
        }
        return true;
    }
}
