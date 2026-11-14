using Nivara.AI;
using Nivara.AutoDiff.Utilities;

namespace NivaraChat.Helpers;

public sealed record AdaptiveFrozenFeatures(
    float[] Values,
    int SequenceLength,
    int HiddenSize,
    int[] MarkerPositions);

public sealed record AdaptiveTrainingExample(
    string Query,
    string ChunkId,
    float[] Features,
    int SequenceLength,
    int HiddenSize,
    int[] MarkerPositions,
    int Target);

public sealed record AdaptiveTrainingSplit(
    IReadOnlyList<AdaptiveTrainingExample> Train,
    IReadOnlyList<AdaptiveTrainingExample> Validation,
    IReadOnlyList<AdaptiveTrainingExample> Test);

public static class AdaptiveTrainingData
{
    public static IReadOnlyList<AdaptiveTrainingExample> Build(
        ResolvedLabels labels,
        IReadOnlyList<MultiViewChunk> chunks,
        ModernBertEncoder<float, float> encoder,
        Gpt2BpeTokenizer tokenizer,
        int maxLen,
        int headMaxLen)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(tokenizer);

        return Build(labels, chunks, (query, chunk) =>
        {
            var (_, sequence) = LayaRetrievalDecisionHead.BuildPrompt(
                tokenizer, query, chunk.ChunkId, chunk.Content, maxLen, headMaxLen);
            var hidden = encoder.Forward(sequence.Ids, sequence.Length);
            if (!hidden.Data.TryGetSpan(out var span))
                throw new InvalidOperationException("ModernBERT encoder output is not available as a contiguous span.");
            var values = span.ToArray();
            return new AdaptiveFrozenFeatures(
                values, sequence.Length, hidden.Shape[1], sequence.Markers);
        });
    }

    public static IReadOnlyList<AdaptiveTrainingExample> Build(
        ResolvedLabels labels,
        IReadOnlyList<MultiViewChunk> chunks,
        Func<string, MultiViewChunk, AdaptiveFrozenFeatures> extractFeatures)
    {
        ArgumentNullException.ThrowIfNull(labels);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(extractFeatures);

        if (GradientUtils.IsGradEnabled)
            throw new InvalidOperationException("ModernBERT feature extraction must run outside GradientUtils.Grad().");
        if (labels.Kind != LabelKind.Judge)
            throw new ArgumentException("Adaptive head training requires resolved judge labels.", nameof(labels));
        if (!RetrievalLabels.Accepts(labels))
            throw new ArgumentException("Judge label coverage is below the accepted retrieval-label threshold.", nameof(labels));

        var examples = new List<AdaptiveTrainingExample>();
        foreach (var (query, grades) in labels.Grades.OrderBy(entry => entry.Key, StringComparer.Ordinal))
        {
            foreach (var (row, grade) in grades.OrderBy(entry => entry.Key))
            {
                if ((uint)row >= (uint)chunks.Count)
                    throw new ArgumentOutOfRangeException(nameof(labels), row, "Resolved label row is outside the chunk list.");
                if (grade is < 0 or > 2)
                    throw new ArgumentOutOfRangeException(nameof(labels), grade, "Judge relevance grades must be 0, 1 or 2.");

                var features = extractFeatures(query, chunks[row]);
                ValidateFeatures(features);
                examples.Add(new AdaptiveTrainingExample(
                    query,
                    chunks[row].ChunkId,
                    (float[])features.Values.Clone(),
                    features.SequenceLength,
                    features.HiddenSize,
                    (int[])features.MarkerPositions.Clone(),
                    grade > 0 ? 0 : 1));
            }
        }

        if (examples.Count == 0)
            throw new InvalidOperationException("No resolved judge examples are available for adaptive head training.");

        return examples;
    }

    public static AdaptiveTrainingSplit Split(
        IReadOnlyList<AdaptiveTrainingExample> examples,
        int seed = QuerySplit.DefaultSeed)
    {
        ArgumentNullException.ThrowIfNull(examples);
        if (examples.Count == 0)
            throw new ArgumentException("At least one adaptive training example is required.", nameof(examples));

        var querySplit = QuerySplit.Create(examples.Select(example => example.Query), seed);
        var trainQueries = querySplit.Train.ToHashSet(StringComparer.Ordinal);
        var validationQueries = querySplit.Validation.ToHashSet(StringComparer.Ordinal);
        var testQueries = querySplit.Test.ToHashSet(StringComparer.Ordinal);

        return new AdaptiveTrainingSplit(
            examples.Where(example => trainQueries.Contains(example.Query)).ToArray(),
            examples.Where(example => validationQueries.Contains(example.Query)).ToArray(),
            examples.Where(example => testQueries.Contains(example.Query)).ToArray());
    }

    static void ValidateFeatures(AdaptiveFrozenFeatures features)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(features.Values);
        ArgumentNullException.ThrowIfNull(features.MarkerPositions);

        if (features.SequenceLength <= 0 || features.HiddenSize <= 0
            || features.Values.Length != (long)features.SequenceLength * features.HiddenSize)
            throw new ArgumentException("Frozen feature values must match [sequenceLength, hiddenSize].", nameof(features));
        if (features.MarkerPositions.Length != 2
            || features.MarkerPositions.Any(marker => marker < 0 || marker >= features.SequenceLength))
            throw new ArgumentException("Adaptive relevance examples require two valid option markers.", nameof(features));
    }
}
