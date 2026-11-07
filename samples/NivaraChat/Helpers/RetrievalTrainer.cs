using Nivara.AutoDiff;
using Nivara.AutoDiff.Nn;
using Nivara.AutoDiff.Nn.Functional;
using Nivara.AutoDiff.Operations;
using Nivara.AutoDiff.Optimizer;
using Nivara.AutoDiff.Utilities;

namespace NivaraChat.Helpers;

/// <summary>
/// What a trainable fusion must provide so the trainer and the evaluator share one forward path.
/// The point of the interface is that a model, its evaluation and its report all agree on how a
/// candidate row becomes a fused score; a second implementation would let the reported metric
/// describe a model that was never trained.
/// </summary>
public interface IFusionRanker
{
    /// <summary>Number of gate features consumed per query; zero for a global fusion.</summary>
    int FeatureCount { get; }

    /// <summary>The trainable leaf parameters, in optimizer group order.</summary>
    IReadOnlyList<Parameter<float>> TrainableParameters { get; }

    /// <summary>Builds the gate feature vector a query contributes.</summary>
    float[] FeatureVector(TrainingQuery query);

    /// <summary>
    /// Runs the masked fusion for one query and returns a <c>[rows, 1]</c> tensor of fused scores.
    /// Outside a <see cref="GradientUtils.Grad"/> scope this builds no graph, which is what evaluation
    /// relies on.
    /// </summary>
    ReverseGradTensor<float> ForwardMasked(TrainingQuery query);

    /// <summary>The four softmax fusion weights for a query, for reporting only.</summary>
    float[] WeightsFor(TrainingQuery query);
}

/// <summary>Shared masked-fusion arithmetic and reporting helpers.</summary>
static class MaskedFusion
{
    /// <summary>
    /// <c>fused = ((S ⊙ M) @ w) / (M @ w)</c>, the AutoDiff form of
    /// <see cref="MultiViewScoring.Fuse"/> for every row at once.
    /// </summary>
    /// <remarks>
    /// The denominator cannot be zero: <paramref name="weights"/> is a softmax output, so every entry
    /// is strictly positive, and every pooled row has at least one present view. That is why the
    /// elementwise <see cref="ReverseGradOperations.Divide"/> is safe here and a zero weight is not a
    /// representable view weight.
    /// </remarks>
    internal static ReverseGradTensor<float> Apply(
        ReverseGradTensor<float> scores,
        ReverseGradTensor<float> weights,
        bool[] present,
        int rows)
    {
        if (scores.Length != rows * TrainingSetBuilder.ViewCount)
            throw new ArgumentException(
                $"Score block length ({scores.Length}) must be rows × {TrainingSetBuilder.ViewCount} ({rows * TrainingSetBuilder.ViewCount}).",
                nameof(scores));
        if (weights.Length != TrainingSetBuilder.ViewCount)
            throw new ArgumentException(
                $"Fusion weights must have {TrainingSetBuilder.ViewCount} entries, got {weights.Length}.", nameof(weights));
        if (present.Length != scores.Length)
            throw new ArgumentException(
                $"Presence mask length ({present.Length}) must match the score block ({scores.Length}).", nameof(present));

        var maskValues = new float[present.Length];
        for (var i = 0; i < present.Length; i++)
            maskValues[i] = present[i] ? 1f : 0f;

        var mask = ReverseGradTensor<float>.FromMatrix(maskValues, rows, TrainingSetBuilder.ViewCount, requiresGrad: false);
        var masked = ReverseGradOperations.Multiply(scores, mask);
        var numerator = ReverseGradOperations.MatMul(masked, weights);
        var denominator = ReverseGradOperations.MatMul(mask, weights);
        return ReverseGradOperations.Divide(numerator, denominator);
    }

    /// <summary>
    /// Copies a tensor's storage into an array. <see cref="ReverseGradTensor{T}.AsSpan"/> is internal
    /// to Nivara, so the sample reads through the public column instead.
    /// </summary>
    internal static float[] ToArray(ReverseGradTensor<float> tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);

        var data = tensor.Data;
        var values = new float[tensor.Length];
        for (var i = 0; i < values.Length; i++)
            values[i] = data[i];

        return values;
    }

    /// <summary>
    /// Softmax over parameter values for reporting. Mirrors the op's numerics (max-shifted) so a
    /// printed weight matches the weight the model actually used.
    /// </summary>
    internal static float[] Weights(ReadOnlySpan<float> logits)
    {
        var max = float.NegativeInfinity;
        foreach (var value in logits)
            max = MathF.Max(max, value);

        var weights = new float[logits.Length];
        var total = 0f;
        for (var i = 0; i < logits.Length; i++)
        {
            weights[i] = MathF.Exp(logits[i] - max);
            total += weights[i];
        }

        for (var i = 0; i < weights.Length; i++)
            weights[i] /= total;
        return weights;
    }

    /// <summary>
    /// Initialises a parameter vector to <c>ln(weight)</c>, so <c>softmax</c> returns that weight
    /// vector exactly and a trainable model starts numerically equal to the #914 baseline.
    /// </summary>
    internal static float[] LogWeights(FusionWeights weights)
    {
        var values = new float[TrainingSetBuilder.ViewCount];
        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
        {
            var weight = weights[(ViewSpace)view];
            values[view] = weight > 0f ? MathF.Log(weight) : -20f;
        }

        return values;
    }
}

/// <summary>
/// Four learnable fusion weights. Initialised to <c>ln(FusionWeights.Default)</c>, which makes the
/// model exactly the #914 baseline at step 0 because those four weights sum to 1.0.
/// </summary>
public sealed class FusionModel : Module<float>, IFusionRanker
{
    readonly Parameter<float> logits;

    public FusionModel(FusionWeights? initial = null)
    {
        var values = MaskedFusion.LogWeights(initial ?? FusionWeights.Default);
        logits = new Parameter<float>(
            "FusionLogits",
            ReverseGradTensor<float>.FromMatrix(values, TrainingSetBuilder.ViewCount, 1, requiresGrad: true));
        RegisterParameters(logits);
    }

    /// <summary>The raw fusion logits, shape <c>[4, 1]</c>.</summary>
    public Parameter<float> Logits => logits;

    public int FeatureCount => 0;

    public IReadOnlyList<Parameter<float>> TrainableParameters => [logits];

    public float[] FeatureVector(TrainingQuery query) => [];

    /// <summary>
    /// Treats the input as a <c>[rows, 4]</c> score block with every view present and returns the
    /// fused scores. The masked path is the general one; this exists so the module contract is real.
    /// </summary>
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Length % TrainingSetBuilder.ViewCount != 0)
            throw new ArgumentException(
                $"Input length ({input.Length}) must be a multiple of {TrainingSetBuilder.ViewCount}.", nameof(input));

        var rows = input.Length / TrainingSetBuilder.ViewCount;
        var present = new bool[input.Length];
        Array.Fill(present, true);
        return Apply(input, Activation.Softmax(logits.Tensor, dim: 0), present, rows);
    }

    public ReverseGradTensor<float> ForwardMasked(TrainingQuery query)
    {
        ArgumentNullException.ThrowIfNull(query);
        var scores = ReverseGradTensor<float>.FromMatrix(
            query.Scores, query.RowCount, TrainingSetBuilder.ViewCount, requiresGrad: false);
        return Apply(scores, Activation.Softmax(logits.Tensor, dim: 0), query.Present, query.RowCount);
    }

    public float[] WeightsFor(TrainingQuery query) => MaskedFusion.Weights(MaskedFusion.ToArray(logits.Tensor));

    /// <summary>Current weights without a query, for reporting a model that ignores the query.</summary>
    public float[] Weights => MaskedFusion.Weights(MaskedFusion.ToArray(logits.Tensor));

    static ReverseGradTensor<float> Apply(
        ReverseGradTensor<float> scores,
        ReverseGradTensor<float> weights,
        bool[] present,
        int rows) => MaskedFusion.Apply(scores, weights, present, rows);
}

/// <summary>What the gate reads to produce its four fusion weights.</summary>
public enum GateFeatureKind
{
    /// <summary>
    /// Per-view mean and max cosine over the candidate pool (8 features). Routes on "which expert is
    /// responding to this query" with 36 parameters, which is the size that can survive ~100 training
    /// queries.
    /// </summary>
    ScoreStats,

    /// <summary>Raw MiniLM query embedding (384 features). More expressive, far more parameters.</summary>
    QueryEmbedding
}

/// <summary>
/// Query-gated mixture of experts: the four cosine scorers are the experts and a gate produces their
/// weights from the query. <c>V = 0</c> and <c>b = ln(defaults)</c> make the gate invariant to its
/// input at initialisation, so step 0 is exactly the #914 baseline.
/// </summary>
public sealed class GatedFusionModel : Module<float>, IFusionRanker
{
    readonly Parameter<float> weight;
    readonly Parameter<float> bias;
    readonly GateFeatureKind kind;
    readonly int features;

    public GatedFusionModel(GateFeatureKind kind, int queryDimension, FusionWeights? initial = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queryDimension);
        this.kind = kind;
        features = kind == GateFeatureKind.ScoreStats ? TrainingSetBuilder.ViewCount * 2 : queryDimension;

        weight = new Parameter<float>(
            "GateWeight",
            ReverseGradTensor<float>.FromMatrix(new float[TrainingSetBuilder.ViewCount * features], TrainingSetBuilder.ViewCount, features, requiresGrad: true));

        var biasValues = MaskedFusion.LogWeights(initial ?? FusionWeights.Default);
        bias = new Parameter<float>(
            "GateBias",
            ReverseGradTensor<float>.FromMatrix(biasValues, TrainingSetBuilder.ViewCount, 1, requiresGrad: true));

        RegisterParameters(weight, bias);
    }

    public GateFeatureKind Kind => kind;

    public Parameter<float> Weight => weight;

    public Parameter<float> Bias => bias;

    public int FeatureCount => features;

    public IReadOnlyList<Parameter<float>> TrainableParameters => [weight, bias];

    /// <summary>
    /// Per-view mean and max over the candidate pool, skipping absent views in the mean. With
    /// <see cref="GateFeatureKind.QueryEmbedding"/> this is the raw query vector instead.
    /// </summary>
    public float[] FeatureVector(TrainingQuery query)
    {
        ArgumentNullException.ThrowIfNull(query);

        if (kind == GateFeatureKind.QueryEmbedding)
        {
            if (query.QueryVector.Length != features)
                throw new ArgumentException(
                    $"Query vector length ({query.QueryVector.Length}) does not match the gate width ({features}).",
                    nameof(query));
            return query.QueryVector;
        }

        var means = new float[TrainingSetBuilder.ViewCount];
        var counts = new int[TrainingSetBuilder.ViewCount];
        var maxima = new float[TrainingSetBuilder.ViewCount];
        Array.Fill(maxima, float.NegativeInfinity);

        for (var row = 0; row < query.RowCount; row++)
        {
            for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
            {
                var index = row * TrainingSetBuilder.ViewCount + view;
                if (!query.Present[index])
                    continue;

                var score = query.Scores[index];
                means[view] += score;
                counts[view]++;
                maxima[view] = MathF.Max(maxima[view], score);
            }
        }

        var vector = new float[TrainingSetBuilder.ViewCount * 2];
        for (var view = 0; view < TrainingSetBuilder.ViewCount; view++)
        {
            vector[view] = counts[view] == 0 ? 0f : means[view] / counts[view];
            vector[TrainingSetBuilder.ViewCount + view] = counts[view] == 0 ? 0f : maxima[view];
        }

        return vector;
    }

    /// <summary>Runs the gate on a <c>[features, 1]</c> input and returns its <c>[4, 1]</c> logits.</summary>
    public override ReverseGradTensor<float> Forward(ReverseGradTensor<float> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Length != features)
            throw new ArgumentException($"Gate input length ({input.Length}) must equal the gate width ({features}).", nameof(input));

        var logits = ReverseGradOperations.MatMul(weight.Tensor, input);
        return ReverseGradOperations.Add(logits, bias.Tensor);
    }

    public ReverseGradTensor<float> ForwardMasked(TrainingQuery query)
    {
        ArgumentNullException.ThrowIfNull(query);
        var scores = ReverseGradTensor<float>.FromMatrix(
            query.Scores, query.RowCount, TrainingSetBuilder.ViewCount, requiresGrad: false);
        var gateInput = ReverseGradTensor<float>.FromMatrix(
            FeatureVector(query), features, 1, requiresGrad: false);

        return MaskedFusion.Apply(scores, Activation.Softmax(Forward(gateInput), dim: 0), query.Present, query.RowCount);
    }

    public float[] WeightsFor(TrainingQuery query)
    {
        ArgumentNullException.ThrowIfNull(query);
        var gateInput = ReverseGradTensor<float>.FromMatrix(FeatureVector(query), features, 1, requiresGrad: false);
        return MaskedFusion.Weights(MaskedFusion.ToArray(Forward(gateInput)));
    }
}

/// <summary>Which ranking loss the trainer minimizes.</summary>
public enum LossKind
{
    /// <summary>
    /// Listwise softmax cross-entropy against a <c>2^grade</c> target. One graph per query, and it
    /// needs no row-gather op because the target and the log-probabilities are elementwise.
    /// </summary>
    Listwise,

    /// <summary>
    /// Pairwise logistic loss over every ordered pair with a grade difference. The pair logits are
    /// <c>A @ fused</c> with a constant ±1 matrix, which is also gather-free.
    /// </summary>
    Pairwise
}

/// <summary>Training hyperparameters. Defaults are sized for ~100 queries on CPU.</summary>
public sealed record TrainingOptions(
    int Epochs = 200,
    float LearningRate = 0.05f,
    float WeightDecay = 0.01f,
    int Patience = 20,
    int Seed = 42,
    LossKind Loss = LossKind.Listwise);

/// <summary>One epoch's recorded loss and validation score.</summary>
public sealed record EpochReport(int Epoch, double TrainLoss, double ValidationNdcg);

/// <summary>
/// Outcome of a training run. <see cref="BestEpoch"/> and the restored parameters correspond to the
/// best validation NDCG, not to the last epoch, so a report never describes an overfitted tail.
/// </summary>
public sealed record TrainingReport(
    IReadOnlyList<EpochReport> History,
    int BestEpoch,
    double BestValidationNdcg,
    double BestTrainLoss,
    int OptimizerSteps,
    int QueriesUsed,
    bool StoppedEarly)
{
    /// <summary>Final training loss, at the epoch training actually ended.</summary>
    public double FinalTrainLoss => History.Count == 0 ? double.NaN : History[^1].TrainLoss;
}

/// <summary>
/// Trains an <see cref="IFusionRanker"/> with Adam. Each usable query is one optimizer step, gradients
/// are zeroed between steps, and the best-validation checkpoint is restored at the end.
/// </summary>
public static class RetrievalTrainer
{
    /// <summary>Runs the epoch loop and leaves the ranker holding its best-validation parameters.</summary>
    public static TrainingReport Train(
        IFusionRanker ranker,
        IReadOnlyList<TrainingQuery> train,
        IReadOnlyList<TrainingQuery> validation,
        TrainingOptions? options = null,
        int topK = RetrievalEvaluation.DefaultK)
    {
        ArgumentNullException.ThrowIfNull(ranker);
        ArgumentNullException.ThrowIfNull(train);
        ArgumentNullException.ThrowIfNull(validation);

        options ??= new TrainingOptions();
        if (options.Epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Epochs must be positive.");

        var usable = train.Where(query => query.IsUsable).ToList();
        if (usable.Count == 0)
        {
            return new TrainingReport([], 0, double.NaN, double.NaN, 0, 0, false);
        }

        using var optimizer = new Adam<float>(options.LearningRate);
        foreach (var parameter in ranker.TrainableParameters)
            optimizer.AddParameterGroup(parameter, options.LearningRate, options.WeightDecay);

        var random = new Random(options.Seed);
        var order = new List<TrainingQuery>(usable);
        var history = new List<EpochReport>(options.Epochs);
        var steps = 0;

        var hasValidation = validation.Count > 0;
        var bestNdcg = double.NegativeInfinity;
        var bestEpoch = 0;
        var bestLoss = double.NaN;
        Dictionary<string, ReverseGradTensor<float>>? bestState = null;
        var sinceBest = 0;

        for (var epoch = 1; epoch <= options.Epochs; epoch++)
        {
            Shuffle(order, random);

            var lossSum = 0d;
            var lossCount = 0;
            foreach (var query in order)
            {
                using (GradientUtils.Grad())
                {
                    var loss = Loss(ranker, query, options.Loss);
                    if (loss is null)
                        continue;

                    loss.Backward();
                    optimizer.Step();
                    optimizer.ZeroGrad();
                    lossSum += loss.Data[0];
                    lossCount++;
                    steps++;
                }
            }

            var trainLoss = lossCount == 0 ? double.NaN : lossSum / lossCount;
            var ndcg = hasValidation
                ? RetrievalEvaluation.MeanNdcg(ranker, validation, topK)
                : double.NaN;
            history.Add(new EpochReport(epoch, trainLoss, ndcg));

            if (!hasValidation)
            {
                bestEpoch = epoch;
                bestLoss = trainLoss;
                continue;
            }

            if (ndcg > bestNdcg)
            {
                bestNdcg = ndcg;
                bestEpoch = epoch;
                bestLoss = trainLoss;
                bestState = ranker is Module<float> module ? module.StateDict() : null;
                sinceBest = 0;
            }
            else if (++sinceBest >= options.Patience)
            {
                break;
            }
        }

        if (bestState is not null && ranker is Module<float> bestModule)
            bestModule.LoadStateDict(bestState);

        return new TrainingReport(
            history,
            bestEpoch,
            hasValidation ? bestNdcg : double.NaN,
            bestLoss,
            steps,
            usable.Count,
            history.Count < options.Epochs);
    }

    /// <summary>Builds the loss graph for one query, or null when the query has nothing to rank.</summary>
    static ReverseGradTensor<float>? Loss(IFusionRanker ranker, TrainingQuery query, LossKind kind)
    {
        var fused = ranker.ForwardMasked(query);
        return kind == LossKind.Pairwise
            ? PairwiseLoss(fused, query)
            : ListwiseLoss(fused, query);
    }

    static ReverseGradTensor<float> ListwiseLoss(ReverseGradTensor<float> fused, TrainingQuery query)
    {
        var rows = query.RowCount;
        var target = ReverseGradTensor<float>.FromMatrix(query.LossTarget, rows, 1, requiresGrad: false);
        var logProbabilities = ReverseGradOperations.LogSoftmax(fused, dim: 0);
        var weighted = ReverseGradOperations.Multiply(logProbabilities, target);
        var total = ReverseGradOperations.Sum(weighted);
        var zero = ReverseGradTensor<float>.FromMatrix([0f], 1, 1, requiresGrad: false);
        return ReverseGradOperations.Subtract(zero, total);
    }

    static ReverseGradTensor<float>? PairwiseLoss(ReverseGradTensor<float> fused, TrainingQuery query)
    {
        var pairs = new List<(int Better, int Worse)>();
        for (var better = 0; better < query.RowCount; better++)
        {
            for (var worse = 0; worse < query.RowCount; worse++)
            {
                if (query.Candidates[better].Grade > query.Candidates[worse].Grade)
                    pairs.Add((better, worse));
            }
        }

        if (pairs.Count == 0)
            return null;

        var rows = query.RowCount;
        var differences = new float[pairs.Count * rows];
        for (var pair = 0; pair < pairs.Count; pair++)
        {
            differences[pair * rows + pairs[pair].Better] = 1f;
            differences[pair * rows + pairs[pair].Worse] = -1f;
        }

        var selector = ReverseGradTensor<float>.FromMatrix(differences, pairs.Count, rows, requiresGrad: false);
        var logits = ReverseGradOperations.MatMul(selector, fused);
        var targets = ReverseGradTensor<float>.FromMatrix(Enumerable.Repeat(1f, pairs.Count).ToArray(), pairs.Count, 1, requiresGrad: false);
        return new BCEWithLogitsLoss<float>(Reduction.Mean).Forward(logits, targets);
    }

    static void Shuffle(List<TrainingQuery> items, Random random)
    {
        for (var i = items.Count - 1; i > 0; i--)
        {
            var j = random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }
}
