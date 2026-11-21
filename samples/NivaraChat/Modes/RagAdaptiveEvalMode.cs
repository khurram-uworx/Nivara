using Nivara;
using Nivara.AI;
using Nivara.AutoDiff;
using Nivara.Silk;
using NivaraChat.Helpers;
using System.Diagnostics;

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
        var cpuBuildSw = Stopwatch.StartNew();
        using var index = await MultiViewIndex.BuildAsync(ctx);
        cpuBuildSw.Stop();
        var cpuBuildMs = cpuBuildSw.ElapsedMilliseconds;
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
            modelDir, loadAdaptiveWeights: ctx.Resume, adaptiveWeightsPath: headWeightsPath, useSilk: ctx.UseSilk);
        LayaRetrievalDecisionHead.LayaDecisionEngine? cpuLayaEngine = null;
        if (ctx.UseSilk)
            cpuLayaEngine = new LayaRetrievalDecisionHead.LayaDecisionEngine(
                modelDir, loadAdaptiveWeights: ctx.Resume, adaptiveWeightsPath: headWeightsPath, useSilk: false);

        bool silkHeadAvailable = false;
        string headTrainingBackend = ctx.UseSilk ? "cpu (silk unavailable)" : "cpu";
        Action<LayaDecisionHead<float, float>, IReadOnlyList<AdaptiveTrainingExample>, int, float>? silkUpdate = null;

        // silk parity
        MultiViewIndex? silkIndex = null;
        SilkParityReport silkParity = new(0, 0, 0, true);
        long silkBuildMs = 0;
        bool cpuFromCache = index.FromCache, silkFromCache = false;

        if (ctx.UseSilk)
        {
            var silkCtx = ctx with { UseSilk = true };
            var silkBuildSw = Stopwatch.StartNew();
            silkIndex = await MultiViewIndex.BuildAsync(silkCtx);
            silkBuildSw.Stop();
            silkBuildMs = silkBuildSw.ElapsedMilliseconds;
            if (silkIndex is not null)
            {
                silkFromCache = silkIndex.FromCache;
                silkParity = AdaptiveSilkParity.Compare(
                    index.Metadata, silkIndex.Metadata,
                    ["VectorContent", "VectorTitle", "VectorSection", "VectorCode"]);
            }
        }

        // head-training backend selection (after index builds, to avoid device contention)
        SilkRuntime? silkHeadRuntime = null;
        if (ctx.UseSilk)
        {
            try
            {
                var probeTensors = HeadToCheckpointTensors(engine.Head);
                silkHeadRuntime = new SilkRuntime();
                using var probeTrainer = new SilkLayaHeadTrainer(silkHeadRuntime, probeTensors, engine.Head.HeadLayerCount, 2, 8);
                silkHeadAvailable = true;
                silkUpdate = (h, ex, ep, lr) => SilkUpdate(silkHeadRuntime, h, ex, ep, lr);
                headTrainingBackend = "silk";
            }
            catch (Exception probeError)
            {
                silkHeadAvailable = false;
                headTrainingBackend = "cpu (silk unavailable)";
                silkHeadRuntime?.Dispose();
                silkHeadRuntime = null;
                Console.WriteLine($"Silk head training unavailable ({probeError.GetType().Name}: {probeError.Message}); falling back to CPU AutoDiff.");
            }
        }

        var collector = new AdaptiveFeedbackCollector(
            engine.Head, headWeightsPath, updateBatch, headEpochs, headLr, persist: !ctx.NoPersist,
            updateAction: silkHeadAvailable ? silkUpdate : null);

        double layaMaxAbs = 0d;
        bool layaWithin = true;
        bool layaAvailable = false;
        if (ctx.UseSilk && cpuLayaEngine is not null && index.Chunks.Count > 0)
        {
            try
            {
                var sampleChunk = index.Chunks[0];
                var cpuFeatures = cpuLayaEngine.ExtractFeatures("laya-parity", sampleChunk);
                var silkFeatures = engine.ExtractFeatures("laya-parity", sampleChunk);
                if (cpuFeatures.Values.Length == silkFeatures.Values.Length && cpuFeatures.Values.Length > 0)
                {
                    var report = AdaptiveSilkParity.CompareLaya(
                        cpuFeatures.Values, silkFeatures.Values);
                    layaMaxAbs = report.MaxAbsoluteDifference;
                    layaWithin = report.WithinTolerance;
                    layaAvailable = true;
                }
            }
            catch
            {
                layaAvailable = false;
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
            testResult.RetrievalRerankRow with { Name = "adaptive-rerank (before)" },
            testResultAfter.RetrievalRerankRow with { Name = "adaptive-rerank (after)" },
            testResult.RetrievalFilterRow with { Name = "adaptive-filter (before)" },
            testResultAfter.RetrievalFilterRow with { Name = "adaptive-filter (after)" },
        };

        // paired comparisons
        var comparisons = new List<PairedComparison>();
        var pc1 = BuildPaired("adaptive-rerank(after)", "fixed", testResultAfter.PerQueryNdcgRerank, testResult.PerQueryNdcgFixed);
        if (pc1 is not null) comparisons.Add(pc1);
        var pc2 = BuildPaired("adaptive-rerank(after)", "adaptive-rerank(before)", testResultAfter.PerQueryNdcgRerank, testResult.PerQueryNdcgRerank);
        if (pc2 is not null) comparisons.Add(pc2);

        // persistence verification
        var finalSize = File.Exists(headWeightsPath) ? new FileInfo(headWeightsPath).Length : 0;
        var bytesWritten = finalSize;
        bool verified = false;
        if (File.Exists(headWeightsPath) && !ctx.NoPersist)
        {
            try
            {
                using var verifyEngine = new LayaRetrievalDecisionHead.LayaDecisionEngine(modelDir, loadAdaptiveWeights: true, adaptiveWeightsPath: headWeightsPath);
                verified = StateDictEqual(engine.Head.StateDict(), verifyEngine.Head.StateDict());
            }
            catch { verified = false; }
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
            new("index build (cpu)", 1, (int)cpuBuildMs),
        };
        if (ctx.UseSilk)
            timings.Add(new AdaptiveReportTiming("index build (silk)", 1, (int)silkBuildMs));
        timings.Add(new AdaptiveReportTiming("test feature extraction (before)", testResult.Examples.Count + (testResult.Observations.Count - testResult.Examples.Count), (int)testSw.ElapsedMilliseconds));
        timings.Add(new AdaptiveReportTiming("train feature extraction + decisions", trainExamplesCount, (int)trainSw.ElapsedMilliseconds));
        timings.Add(new AdaptiveReportTiming(silkHeadAvailable ? "head updates (Silk)" : "head updates (AutoDiff)", saves, (int)collector.UpdateMilliseconds));
        timings.Add(new AdaptiveReportTiming("head persistence", saves, (int)collector.PersistMilliseconds));
        timings.Add(new AdaptiveReportTiming("test feature extraction (after)", testResultAfter.Examples.Count, (int)testAfterSw.ElapsedMilliseconds));

        var silkReport = new AdaptiveReportSilk(
            ctx.UseSilk ? "silk" : "cpu (silk not requested)",
            silkParity.MaxAbsoluteDifference,
            silkParity.MismatchedRows,
            silkParity.WithinTolerance,
            cpuBuildMs,
            silkBuildMs,
            cpuFromCache,
            silkFromCache,
            ctx.UseSilk && silkIndex is not null,
            layaMaxAbs,
            layaWithin,
            layaAvailable)
        {
            HeadTrainingBackend = headTrainingBackend
        };

        var backendMatrix = new BackendMatrix(
            new ComponentBackend("embeddings", ctx.UseSilk ? "Silk (OpenCL)" : "CPU", "F32"),
            new ComponentBackend("llm", "not used", "n/a"),
            [],
            new ComponentBackend("laya-encoder", ctx.UseSilk ? "Silk (OpenCL)" : "CPU", "F32"),
            new ComponentBackend("laya-head-training", headTrainingBackend, "F32"));
        BackendMatrixPrinter.Print(backendMatrix);

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
            HeadTrainingBackend = headTrainingBackend,
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
            Backends = backendMatrix,
            Persistence = new AdaptiveReportPersistence(headWeightsPath, existedBefore, saves, bytesWritten, finalSize, verified) { Disabled = ctx.NoPersist },
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
        cpuLayaEngine?.Dispose();
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
                return new { x.Row, chunk, x.Score, features, decision };
            }).ToList();

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

    static void SilkUpdate(
        SilkRuntime runtime,
        LayaDecisionHead<float, float> head,
        IReadOnlyList<AdaptiveTrainingExample> examples,
        int epochs,
        float learningRate)
    {
        if (examples.Count == 0) return;

        var tensors = HeadToCheckpointTensors(head);
        int maxSeq = 1;
        foreach (var e in examples)
            if (e.SequenceLength > maxSeq) maxSeq = e.SequenceLength;

        using var trainer = new SilkLayaHeadTrainer(runtime, tensors, head.HeadLayerCount, 2, maxSeq);

        int n = examples.Count;
        for (var epoch = 0; epoch < epochs; epoch++)
        {
            trainer.ZeroGradients();
            foreach (var example in examples)
            {
                var output = trainer.Forward(example.Features, LayaQuestionType.Choice, example.MarkerPositions, example.SequenceLength);

                var dScorer = new float[example.MarkerPositions.Length];
                var labels = new[] { example.Target };
                SilkSoftmaxInputGradient(output.Logits, labels, 1, example.MarkerPositions.Length, 1f / n, dScorer);

                var dAct = new float[2];
                trainer.Backward(dScorer, dAct);
            }
            trainer.AdamUpdateAll(learningRate, 0f);
        }

        var exported = trainer.ExportCheckpoint();
        using var reloaded = LayaDecisionHead<float, float>.LoadWeights(exported, head.HeadLayerCount, 2);
        head.LoadStateDict(reloaded.StateDict(), strict: true);
    }

    static void SilkSoftmaxInputGradient(float[] logits, int[] labels, int n, int numClasses, float scale, float[] dZ)
    {
        for (int i = 0; i < n; i++)
        {
            int rowBase = i * numClasses;
            float max = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
                max = MathF.Max(max, logits[rowBase + c]);
            float sum = 0f;
            for (int c = 0; c < numClasses; c++)
                sum += MathF.Exp(logits[rowBase + c] - max);
            int label = labels[i];
            for (int c = 0; c < numClasses; c++)
            {
                float p = MathF.Exp(logits[rowBase + c] - max) / sum;
                dZ[rowBase + c] = (p - (c == label ? 1f : 0f)) * scale;
            }
        }
    }

    static Dictionary<string, (float[] Data, int[] Shape)> HeadToCheckpointTensors(LayaDecisionHead<float, float> head)
    {
        var t = new Dictionary<string, (float[] Data, int[] Shape)>(StringComparer.Ordinal);
        int h = head.HiddenSize;

        // type_emb
        t["type_emb.weight"] = (TensorToFloat(head.typeEmb.Weight!.Tensor), new[] { 3, h });

        // head layers
        for (int i = 0; i < head.HeadLayers.Count; i++)
        {
            var layer = head.HeadLayers[i];
            string hp = $"head.layers.{i}";

            // norm1
            AddLayerNorm(t, layer.norm1, $"{hp}.norm1", h);
            // attn fused in_proj
            var attn = layer.attn;
            var qW = TensorToFloat(attn.qProj.Weight!.Tensor);
            var kW = TensorToFloat(attn.kProj.Weight!.Tensor);
            var vW = TensorToFloat(attn.vProj.Weight!.Tensor);
            var fusedW = new float[3 * h * h];
            var ws = new[] { qW, kW, vW };
            for (int b = 0; b < 3; b++)
                for (int r = 0; r < h; r++)
                    for (int c = 0; c < h; c++)
                        fusedW[(b * h + r) * h + c] = ws[b][r * h + c];
            t[$"{hp}.self_attn.in_proj_weight"] = (fusedW, new[] { 3 * h, h });

            var qB = TensorToFloat(attn.qProj.Bias!.Tensor);
            var kB = TensorToFloat(attn.kProj.Bias!.Tensor);
            var vB = TensorToFloat(attn.vProj.Bias!.Tensor);
            var fusedB = new float[3 * h];
            Array.Copy(qB, 0, fusedB, 0, h);
            Array.Copy(kB, 0, fusedB, h, h);
            Array.Copy(vB, 0, fusedB, 2 * h, h);
            t[$"{hp}.self_attn.in_proj_bias"] = (fusedB, new[] { 3 * h });

            AddLinear(t, attn.oProj, $"{hp}.self_attn.out_proj");

            // norm2
            AddLayerNorm(t, layer.norm2, $"{hp}.norm2", h);

            // linear1 / linear2
            AddLinear(t, layer.linear1, $"{hp}.linear1");
            AddLinear(t, layer.linear2, $"{hp}.linear2");
        }

        // scorer
        AddLayerNorm(t, head.scorerNorm, "scorer.0", h);
        AddLinear(t, head.scorerHidden, "scorer.1");
        AddLinear(t, head.scorerOut, "scorer.3");

        // act_head
        AddLinear(t, head.actInput, "act_head.0");
        AddLinear(t, head.actOut, "act_head.2");

        // temperature
        var tempArr = new float[head.CheckpointTemperature.Count];
        for (int i = 0; i < tempArr.Length; i++)
            tempArr[i] = float.CreateChecked(head.CheckpointTemperature[i]);
        t["temperature"] = (tempArr, new[] { 3 });

        return t;
    }

    static void AddLayerNorm(Dictionary<string, (float[] Data, int[] Shape)> t, Nivara.AutoDiff.Nn.LayerNorm<float> ln, string prefix, int n)
    {
        if (ln.Weight is not null)
            t[$"{prefix}.weight"] = (TensorToFloat(ln.Weight.Tensor), new[] { n });
        if (ln.Bias is not null)
            t[$"{prefix}.bias"] = (TensorToFloat(ln.Bias.Tensor), new[] { n });
    }

    static void AddLinear(Dictionary<string, (float[] Data, int[] Shape)> t, Nivara.AutoDiff.Nn.Linear<float, float> lin, string prefix)
    {
        t[$"{prefix}.weight"] = (TensorToFloat(lin.Weight!.Tensor), new[] { lin.OutFeatures, lin.InFeatures });
        if (lin.Bias is not null)
            t[$"{prefix}.bias"] = (TensorToFloat(lin.Bias.Tensor), new[] { lin.OutFeatures });
    }

    static float[] TensorToFloat(ReverseGradTensor<float> tensor)
        => tensor.Data.ToArray();

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
