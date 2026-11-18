using System.Globalization;
using System.Text;

namespace NivaraChat.Helpers;

/// <summary>
/// All the data the adaptive report renders. Properties are init-only so call sites stay readable.
/// </summary>
public sealed record AdaptiveReportInput
{
    public required string Mode { get; init; }
    public required string LabelKind { get; init; }
    public required string LabelDescription { get; init; }
    public required int LabelMatched { get; init; }
    public required int LabelTotal { get; init; }
    public required double LabelCoverage { get; init; }
    public required int LabelQueries { get; init; }
    public required int Seed { get; init; }
    public required int K { get; init; }
    public required double Threshold { get; init; }
    public required int MaxQueries { get; init; }
    public required int MaxExamples { get; init; }
    public required int CorpusFiles { get; init; }
    public required int CorpusChunks { get; init; }
    public required string CorpusContentHash { get; init; }
    public required string CorpusChunkerVersion { get; init; }
    public required int CorpusChunkSize { get; init; }
    public required string CorpusBackend { get; init; }
    public required int CorpusDimension { get; init; }
    public required int TrainQueries { get; init; }
    public required int TestQueries { get; init; }
    public required int TrainExamples { get; init; }
    public required int TestExamples { get; init; }
    public required long TrainFeatureBytes { get; init; }
    public required long TestFeatureBytes { get; init; }
    public required bool Leakage { get; init; }
    public required bool Resume { get; init; }
    public required int UpdateBatch { get; init; }
    public required int HeadEpochs { get; init; }
    public required double HeadLearningRate { get; init; }
    public required IReadOnlyList<AdaptiveReportRow> RetrievalRows { get; init; }
    public required IReadOnlyList<PairedComparison> Comparisons { get; init; }
    public required AdaptiveReportDecision DecisionBefore { get; init; }
    public required AdaptiveReportDecision DecisionAfter { get; init; }
    public required AdaptiveReportCalibration CalibrationBefore { get; init; }
    public required AdaptiveReportCalibration CalibrationAfter { get; init; }
    public required IReadOnlyList<AdaptiveCalibrationBin> ReliabilityBefore { get; init; }
    public required IReadOnlyList<AdaptiveCalibrationBin> ReliabilityAfter { get; init; }
    public required IReadOnlyList<AdaptiveReportTiming> Timings { get; init; }
    public required AdaptiveReportSilk Silk { get; init; }
    public required AdaptiveReportPersistence Persistence { get; init; }
    public required IReadOnlyList<AdaptiveReportAgreement> InSampleAgreement { get; init; }
    public required string DecisiveNote { get; init; }
}

public sealed record AdaptiveReportRow(
    string Name, double Precision, double Recall, double Mrr, double Ndcg, int Queries);

public sealed record AdaptiveReportDecision(
    int Count, int Correct, double Accuracy,
    double AcceptPrecision, double AcceptRecall, double AcceptF1);

public sealed record AdaptiveReportCalibration(int Count, double Ece, double Brier);

public sealed record AdaptiveReportTiming(string Stage, int Count, long Milliseconds);

public sealed record AdaptiveReportSilk(
    string Backend, double MaxAbsoluteDifference, int MismatchedRows,
    bool WithinTolerance, long CpuBuildMs, long SilkBuildMs,
    bool CpuFromCache, bool SilkFromCache, bool Available,
    double LayaMaxAbsoluteDifference = 0d,
    bool LayaWithinTolerance = true,
    bool LayaAvailable = false)
{
    /// <summary>Optional explicit statement of which components were compared.</summary>
    public string? ComponentsCompared { get; init; }

    public string ComparedComponents =>
        ComponentsCompared ?? (Available ? (LayaAvailable ? "embeddings, Laya" : "embeddings") : "none");

    public string Components => ComparedComponents;
    public string Compared => ComparedComponents;
    public bool EmbeddingsCompared => Available;
    public bool LayaCompared => LayaAvailable;
    public bool ComparedEmbeddings => Available;
    public bool ComparedLaya => LayaAvailable;
}

public sealed record AdaptiveReportPersistence(
    string Path, bool ExistedBefore, int Saves, long BytesWritten, long FinalSize, bool Verified, bool Disabled = false);

public sealed record AdaptiveReportAgreement(int Batch, int Count, int Correct, double Accuracy);

public static class AdaptiveReport
{
    public static string ResolvePath(string? reportPath, string repoRoot)
    {
        var directory = Path.Combine(repoRoot, "samples", "data", "nivarachat", "adaptive");
        if (string.IsNullOrWhiteSpace(reportPath))
            return Path.Combine(directory, "rag-adaptive-report.md");
        if (Path.IsPathRooted(reportPath))
            return reportPath;
        return Path.Combine(directory, reportPath);
    }

    public static string MetricsPath(string repoRoot, string labelKind, int seed, int k)
    {
        var directory = Path.Combine(repoRoot, "samples", "data", "nivarachat", "adaptive");
        return Path.Combine(directory, $"metrics-{labelKind}-s{seed}-k{k}.json");
    }

    public static void Write(string path, string content)
    {
        ArgumentNullException.ThrowIfNull(path);
        ArgumentNullException.ThrowIfNull(content);
        var directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        File.WriteAllText(path, content);
    }

    public static string Markdown(AdaptiveReportInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var c = new StringBuilder();
        var inv = CultureInfo.InvariantCulture;

        c.AppendLine("# Adaptive retrieval and decision (#983)");
        c.AppendLine();
        c.AppendLine($"- Mode: {input.Mode}");
        c.AppendLine($"- Label source: {input.LabelDescription}");
        c.AppendLine($"- Label coverage: {input.LabelMatched}/{input.LabelTotal} rows ({input.LabelCoverage.ToString("P1", inv)}), {input.LabelQueries} queries");
        c.AppendLine($"- Split: {input.TrainQueries} train / {input.TestQueries} test (seed {input.Seed})");
        c.AppendLine($"- K: {input.K}; decision threshold: {input.Threshold.ToString("0.00", inv)}");
        c.AppendLine($"- Caps: max-queries {input.MaxQueries}, max-examples {input.MaxExamples} per split");
        c.AppendLine($"- Head: epochs {input.HeadEpochs}, lr {input.HeadLearningRate.ToString("0.0000", inv)}, update batch {input.UpdateBatch}");
        c.AppendLine($"- Resume from persisted head: {(input.Resume ? "yes" : "no")}");
        c.AppendLine($"- Leakage diagnostic: {(input.Leakage ? "yes (auto labels)" : "no")}");
        c.AppendLine();
        c.AppendLine("## Corpus");
        c.AppendLine();
        c.AppendLine($"- Files: {input.CorpusFiles}; chunks: {input.CorpusChunks}; chunk size {input.CorpusChunkSize}; chunker {input.CorpusChunkerVersion}");
        c.AppendLine($"- Content hash: {input.CorpusContentHash}");
        c.AppendLine($"- Backend: {input.CorpusBackend}; dimension: {input.CorpusDimension}");
        c.AppendLine();
        c.AppendLine("## Examples");
        c.AppendLine();
        c.AppendLine($"- Train: {input.TrainExamples} examples, {input.TrainFeatureBytes} feature bytes");
        c.AppendLine($"- Test: {input.TestExamples} examples, {input.TestFeatureBytes} feature bytes");
        c.AppendLine();
        c.AppendLine($"## Held-out test retrieval (K = {input.K})");
        c.AppendLine();
        c.AppendLine("| Ranker | P@K | R@K | MRR | nDCG@K | Queries |");
        c.AppendLine("| --- | --- | --- | --- | --- | --- |");
        foreach (var row in input.RetrievalRows)
            c.AppendLine($"| {row.Name} | {row.Precision.ToString("0.000", inv)} | {row.Recall.ToString("0.000", inv)} | {row.Mrr.ToString("0.000", inv)} | {row.Ndcg.ToString("0.0000", inv)} | {row.Queries} |");
        c.AppendLine();
        if (input.Comparisons.Count > 0)
        {
            c.AppendLine("## Paired comparisons (per-query nDCG)");
            c.AppendLine();
            foreach (var comparison in input.Comparisons)
                c.AppendLine($"- {comparison.Summary}");
            c.AppendLine();
        }
        c.AppendLine("## Decisions (graded pairs in the top-K)");
        c.AppendLine();
        c.AppendLine("| Metric | Before | After |");
        c.AppendLine("| --- | --- | --- |");
        c.AppendLine($"| Choice accuracy | {input.DecisionBefore.Accuracy.ToString("0.000", inv)} ({input.DecisionBefore.Correct}/{input.DecisionBefore.Count}) | {input.DecisionAfter.Accuracy.ToString("0.000", inv)} ({input.DecisionAfter.Correct}/{input.DecisionAfter.Count}) |");
        c.AppendLine($"| Accept precision | {input.DecisionBefore.AcceptPrecision.ToString("0.000", inv)} | {input.DecisionAfter.AcceptPrecision.ToString("0.000", inv)} |");
        c.AppendLine($"| Accept recall | {input.DecisionBefore.AcceptRecall.ToString("0.000", inv)} | {input.DecisionAfter.AcceptRecall.ToString("0.000", inv)} |");
        c.AppendLine($"| Accept F1 | {input.DecisionBefore.AcceptF1.ToString("0.000", inv)} | {input.DecisionAfter.AcceptF1.ToString("0.000", inv)} |");
        c.AppendLine();
        c.AppendLine("## Calibration (choice correctness vs AnswerConfidence)");
        c.AppendLine();
        c.AppendLine("| Metric | Before | After |");
        c.AppendLine("| --- | --- | --- |");
        c.AppendLine($"| ECE | {input.CalibrationBefore.Ece.ToString("0.0000", inv)} | {input.CalibrationAfter.Ece.ToString("0.0000", inv)} |");
        c.AppendLine($"| Brier | {input.CalibrationBefore.Brier.ToString("0.0000", inv)} | {input.CalibrationAfter.Brier.ToString("0.0000", inv)} |");
        c.AppendLine($"| Pairs | {input.CalibrationBefore.Count} | {input.CalibrationAfter.Count} |");
        c.AppendLine();
        c.AppendLine("### Reliability (after)");
        c.AppendLine();
        c.AppendLine("| Bin | n | Mean conf | Empirical acc | Gap |");
        c.AppendLine("| --- | --- | --- | --- | --- |");
        foreach (var bin in input.ReliabilityAfter)
            c.AppendLine($"| [{bin.Lower.ToString("0.0", inv)}, {bin.Upper.ToString("0.0", inv)}) | {bin.Count} | {bin.MeanConfidence.ToString("0.000", inv)} | {bin.Accuracy.ToString("0.000", inv)} | {bin.Gap.ToString("0.000", inv)} |");
        c.AppendLine();
        if (input.InSampleAgreement.Count > 0)
        {
            c.AppendLine("## In-sample agreement per update batch (fit curve, not generalisation)");
            c.AppendLine();
            c.AppendLine("| Batch | n | Correct | Accuracy |");
            c.AppendLine("| --- | --- | --- | --- |");
            foreach (var a in input.InSampleAgreement)
                c.AppendLine($"| {a.Batch} | {a.Count} | {a.Correct} | {a.Accuracy.ToString("0.000", inv)} |");
            c.AppendLine();
        }
        c.AppendLine("## Timings (wall-clock, not reproducible across machines)");
        c.AppendLine();
        c.AppendLine("| Stage | n | ms |");
        c.AppendLine("| --- | --- | --- |");
        foreach (var t in input.Timings)
            c.AppendLine($"| {t.Stage} | {t.Count} | {t.Milliseconds} |");
        c.AppendLine();
        c.AppendLine("## Silk scope");
        c.AppendLine();
        c.AppendLine($"- Backend used for embeddings: {input.Silk.Backend}");
        c.AppendLine($"- Available: {(input.Silk.Available ? "yes" : "no")}");
        c.AppendLine($"- Components compared: {input.Silk.ComparedComponents}");
        if (!input.Silk.Available)
            c.AppendLine("- Embedding parity: not run (pass --silk to enable).");
        else
            c.AppendLine($"- Embedding parity max |cpu - silk|: {input.Silk.MaxAbsoluteDifference.ToString("0.0000000", inv)} (tolerance {AdaptiveSilkParity.Tolerance.ToString("0.0E+0", inv)})");
        if (input.Silk.Available)
        {
            c.AppendLine($"- Mismatched rows: {input.Silk.MismatchedRows}");
            c.AppendLine($"- CPU index build: {input.Silk.CpuBuildMs} ms (cache hit: {input.Silk.CpuFromCache})");
            c.AppendLine($"- Silk index build: {input.Silk.SilkBuildMs} ms (cache hit: {input.Silk.SilkFromCache})");
        }
        c.AppendLine();
        c.AppendLine("### Laya (decision head)");
        if (!input.Silk.LayaAvailable)
            c.AppendLine("- Laya parity: not run (pass --silk to enable, or OpenCL unavailable).");
        else
            c.AppendLine($"- Laya encoder parity max |cpu - silk|: {input.Silk.LayaMaxAbsoluteDifference.ToString("0.0000000", inv)} (within tolerance: {input.Silk.LayaWithinTolerance})");
        c.AppendLine("- Scope note: embedding parity uses the 1e-5 tolerance; ModernBERT feature extraction parity is reported under Laya when available. Decisions and head training remain CPU in both runs.");
        c.AppendLine();
        c.AppendLine("## Persistence");
        c.AppendLine();
        c.AppendLine($"- Path: {input.Persistence.Path}");
        if (input.Persistence.Disabled)
        {
            c.AppendLine("- Status: disabled (--no-persist); no writes performed.");
            c.AppendLine();
        }
        else
        {
            c.AppendLine($"- Existed before run: {(input.Persistence.ExistedBefore ? "yes" : "no")}");
            c.AppendLine($"- Updates persisted: {input.Persistence.Saves}");
            c.AppendLine($"- Checkpoint size (per save): {input.Persistence.BytesWritten}");
            c.AppendLine($"- Final file size: {input.Persistence.FinalSize}");
            c.AppendLine($"- Reload verified bit-identical: {(input.Persistence.Verified ? "yes" : "no")}");
            c.AppendLine();
        }
        c.AppendLine("## Reading this report");
        c.AppendLine();
        c.AppendLine("- The test split is scored twice (before and after) with no model selection in between, so the delta is attributable to learning rather than selection.");
        c.AppendLine("- `adaptive-rerank` preserves the fixed top-K membership, so its P@K and R@K are invariant by construction; only `adaptive-filter` can move them.");
        c.AppendLine("- Feedback is simulated from judge labels and is stated as such; it is not human click feedback.");
        c.AppendLine("- Auto labels are a leakage diagnostic: Title and Section views score their own positive near 1.0.");
        c.AppendLine("- Decisions and head training are CPU in both backends; only MiniLM embeddings are Silk-accelerated.");
        c.AppendLine($"- {input.DecisiveNote}");
        c.AppendLine();
        c.AppendLine("Metrics above are deterministic given the corpus hash, label file, seed and caps. Timings are wall-clock and vary by machine.");

        return c.ToString();
    }

    public static string MetricsJson(AdaptiveReportInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var obj = new
        {
            schemaVersion = 1,
            mode = input.Mode,
            labelKind = input.LabelKind,
            labelDescription = input.LabelDescription,
            labelMatched = input.LabelMatched,
            labelTotal = input.LabelTotal,
            labelCoverage = input.LabelCoverage,
            labelQueries = input.LabelQueries,
            seed = input.Seed,
            k = input.K,
            threshold = input.Threshold,
            maxQueries = input.MaxQueries,
            maxExamples = input.MaxExamples,
            corpus = new
            {
                files = input.CorpusFiles,
                chunks = input.CorpusChunks,
                contentHash = input.CorpusContentHash,
                chunkerVersion = input.CorpusChunkerVersion,
                chunkSize = input.CorpusChunkSize,
                backend = input.CorpusBackend,
                dimension = input.CorpusDimension
            },
            split = new { train = input.TrainQueries, test = input.TestQueries },
            examples = new { train = input.TrainExamples, test = input.TestExamples },
            featureBytes = new { train = input.TrainFeatureBytes, test = input.TestFeatureBytes },
            leakage = input.Leakage,
            resume = input.Resume,
            head = new { epochs = input.HeadEpochs, learningRate = input.HeadLearningRate, updateBatch = input.UpdateBatch },
            retrieval = input.RetrievalRows.Select(r => new { name = r.Name, pAtK = r.Precision, rAtK = r.Recall, mrr = r.Mrr, ndcgAtK = r.Ndcg, queries = r.Queries }).ToArray(),
            comparisons = input.Comparisons.Select(x => new { challenger = x.Challenger, baseline = x.Baseline, wins = x.Wins, losses = x.Losses, ties = x.Ties, p = x.SignTestPValue, d = x.CohensD, delta = x.MeanNdcgDelta }).ToArray(),
            decisions = new
            {
                before = new { count = input.DecisionBefore.Count, correct = input.DecisionBefore.Correct, accuracy = input.DecisionBefore.Accuracy, acceptPrecision = input.DecisionBefore.AcceptPrecision, acceptRecall = input.DecisionBefore.AcceptRecall, acceptF1 = input.DecisionBefore.AcceptF1 },
                after = new { count = input.DecisionAfter.Count, correct = input.DecisionAfter.Correct, accuracy = input.DecisionAfter.Accuracy, acceptPrecision = input.DecisionAfter.AcceptPrecision, acceptRecall = input.DecisionAfter.AcceptRecall, acceptF1 = input.DecisionAfter.AcceptF1 }
            },
            calibration = new
            {
                before = new { count = input.CalibrationBefore.Count, ece = input.CalibrationBefore.Ece, brier = input.CalibrationBefore.Brier },
                after = new { count = input.CalibrationAfter.Count, ece = input.CalibrationAfter.Ece, brier = input.CalibrationAfter.Brier }
            },
            silk = new
            {
                backend = input.Silk.Backend,
                available = input.Silk.Available,
                maxAbs = input.Silk.MaxAbsoluteDifference,
                mismatchedRows = input.Silk.MismatchedRows,
                withinTolerance = input.Silk.WithinTolerance,
                cpuBuildMs = input.Silk.CpuBuildMs,
                silkBuildMs = input.Silk.SilkBuildMs,
                cpuFromCache = input.Silk.CpuFromCache,
                silkFromCache = input.Silk.SilkFromCache,
                layaAvailable = input.Silk.LayaAvailable,
                layaMaxAbs = input.Silk.LayaMaxAbsoluteDifference,
                layaWithinTolerance = input.Silk.LayaWithinTolerance,
                componentsCompared = input.Silk.ComparedComponents
            },
            persistence = new
            {
                path = input.Persistence.Path,
                disabled = input.Persistence.Disabled,
                existedBefore = input.Persistence.ExistedBefore,
                saves = input.Persistence.Saves,
                bytesWritten = input.Persistence.BytesWritten,
                finalSize = input.Persistence.FinalSize,
                verified = input.Persistence.Verified
            }
        };

        var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
        return System.Text.Json.JsonSerializer.Serialize(obj, options);
    }
}
