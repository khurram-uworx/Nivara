namespace NivaraChat.Helpers;

/// <summary>
/// One observed (query, retrieved chunk) decision paired with its judge grade. <see cref="Grade"/>
/// is the raw judge grade (0/1/2); <see cref="ChoiceRelevant"/> is the head's choice before any
/// threshold, and <see cref="Accepted"/> is whether the router's operating point accepted it.
/// </summary>
public sealed record AdaptiveObservation(
    string Query,
    string ChunkId,
    int Grade,
    bool ChoiceRelevant,
    bool Accepted,
    double AnswerConfidence);

/// <summary>One reliability-diagram bin for an adaptive decision run.</summary>
public sealed record AdaptiveCalibrationBin(
    double Lower,
    double Upper,
    int Count,
    double MeanConfidence,
    double Accuracy,
    double Gap);

/// <summary>Expected calibration error and Brier score for a set of decisions.</summary>
public sealed record AdaptiveCalibration(
    int Count,
    double Ece,
    double Brier,
    IReadOnlyList<AdaptiveCalibrationBin> Bins);

/// <summary>
/// Pure, model-free metrics for an adaptive decision run. Separated from the mode so they can be
/// unit-tested without MiniLM/Laya checkpoints.
/// </summary>
public static class AdaptiveEvaluation
{
    public const int DefaultBins = 10;

    /// <summary>
    /// Bin index for a confidence in [0, 1]. Equal-width bins; a confidence of exactly 1.0 lands in
    /// the last bin so the right edge is inclusive.
    /// </summary>
    public static int BinIndex(double confidence, int bins)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bins);
        if (double.IsNaN(confidence) || confidence < 0d || confidence > 1d)
            throw new ArgumentOutOfRangeException(nameof(confidence), confidence, "Confidence must be in [0, 1].");

        var index = (int)(confidence * bins);
        return index >= bins ? bins - 1 : index;
    }

    /// <summary>
    /// Reliability-diagram statistics. ECE is the weighted mean absolute gap between the empirical
    /// accuracy and the mean confidence over the non-empty bins. Empty bins contribute nothing.
    /// Brier is the mean squared error of the reported confidence against the binary correctness.
    /// </summary>
    public static AdaptiveCalibration Calibrate(IEnumerable<AdaptiveObservation> observations, int bins = DefaultBins)
    {
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(bins);

        var list = observations.ToList();
        if (list.Count == 0)
            throw new ArgumentException("At least one observation is required.", nameof(observations));

        var confidences = new double[bins];
        var accuracies = new double[bins];
        var counts = new int[bins];

        foreach (var observation in list)
        {
            if (observation.AnswerConfidence < 0d || observation.AnswerConfidence > 1d)
                throw new ArgumentOutOfRangeException(nameof(observations), observation.AnswerConfidence, "Confidence must be in [0, 1].");

            var correct = observation.ChoiceRelevant == (observation.Grade > 0);
            var bin = BinIndex(observation.AnswerConfidence, bins);
            counts[bin]++;
            confidences[bin] += observation.AnswerConfidence;
            accuracies[bin] += correct ? 1d : 0d;
        }

        var ece = 0d;
        var brier = 0d;
        var binList = new List<AdaptiveCalibrationBin>(bins);
        for (var i = 0; i < bins; i++)
        {
            double meanConfidence = 0d;
            double accuracy = 0d;
            if (counts[i] > 0)
            {
                meanConfidence = confidences[i] / counts[i];
                accuracy = accuracies[i] / counts[i];
                ece += (double)counts[i] / list.Count * Math.Abs(accuracy - meanConfidence);
            }

            binList.Add(new AdaptiveCalibrationBin(
                i / (double)bins,
                (i + 1) / (double)bins,
                counts[i],
                meanConfidence,
                accuracy,
                counts[i] == 0 ? 0d : accuracy - meanConfidence));
        }

        foreach (var observation in list)
        {
            var y = observation.ChoiceRelevant == (observation.Grade > 0) ? 1d : 0d;
            var gap = observation.AnswerConfidence - y;
            brier += gap * gap;
        }

        return new AdaptiveCalibration(list.Count, ece, brier / list.Count, binList);
    }

    /// <summary>
    /// Choice and accept metrics at the router's operating point. Reuses <see cref="AdaptiveChoiceMetrics"/>
    /// so the counts record stays single-source.
    /// </summary>
    public static AdaptiveChoiceMetrics Decisions(IEnumerable<AdaptiveObservation> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        var list = observations.ToList();
        if (list.Count == 0)
            throw new ArgumentException("At least one observation is required.", nameof(observations));

        var correct = 0;
        var predictedRelevant = 0;
        var trueRelevant = 0;
        var truePositive = 0;
        foreach (var observation in list)
        {
            var truth = observation.Grade > 0;
            var predicted = observation.ChoiceRelevant;
            if (predicted == truth)
                correct++;
            if (predicted)
                predictedRelevant++;
            if (truth)
            {
                trueRelevant++;
                if (predicted)
                    truePositive++;
            }
        }

        var precision = predictedRelevant == 0 ? 0d : (double)truePositive / predictedRelevant;
        var recall = trueRelevant == 0 ? 0d : (double)truePositive / trueRelevant;
        var f1 = precision + recall == 0d ? 0d : 2d * precision * recall / (precision + recall);
        return new AdaptiveChoiceMetrics(
            list.Count, correct, predictedRelevant, trueRelevant,
            (double)correct / list.Count, precision, recall, f1);
    }
}
