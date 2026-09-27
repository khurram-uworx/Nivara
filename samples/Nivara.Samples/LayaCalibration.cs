namespace Nivara.Samples;

/// <summary>The temperature actually applied to one question, with the provenance of the value.</summary>
/// <param name="Scale">The clamped divisor applied to the logits. Always in [0.5, 5.0].</param>
/// <param name="Bucket">The <c>type:size</c> bucket key that was looked up.</param>
/// <param name="RawValue">
/// The configured value before clamping. Equal to <paramref name="Scale"/> unless the configured
/// value was out of range — which the shipped <c>choice:11+</c> bucket is.
/// </param>
/// <param name="FromBucketTable">
/// Whether the bucket table supplied the value, or the per-type fallback did. The bucket table wins
/// whenever it has an entry.
/// </param>
/// <param name="WasClamped">Whether <paramref name="RawValue"/> fell outside the allowed range.</param>
public readonly record struct LayaTemperature(
    double Scale,
    string Bucket,
    double RawValue,
    bool FromBucketTable,
    bool WasClamped);

/// <summary>
/// A decoded Laya answer: the typed decision plus the quantities a caller gates on.
/// </summary>
/// <remarks>
/// <see cref="Probabilities"/> is aligned with the question's option order, i.e. with
/// <see cref="LayaPromptBuilder.RenderOptions"/>, and is a temperature-softened distribution over
/// the first <c>k</c> logits.
/// </remarks>
public sealed record LayaDecision
{
    public required string QuestionId { get; init; }
    public required LayaQuestionType Type { get; init; }

    /// <summary>The selected option's key, for a <see cref="LayaQuestionType.Choice"/> question.</summary>
    public string? Choice { get; init; }

    /// <summary>
    /// The expectation over option indices, for a <see cref="LayaQuestionType.Score"/> question.
    /// </summary>
    public double? Score { get; init; }

    /// <summary>P(true), for a <see cref="LayaQuestionType.Noul"/> question.</summary>
    public double? NoulProbability { get; init; }

    public IReadOnlyList<double> Probabilities { get; init; } = [];

    /// <summary>
    /// Normalized-entropy confidence for <c>choice</c> and <c>score</c>; <c>max(p, 1-p)</c> for
    /// <c>noul</c>. <b>Not calibrated</b> — do not gate on this.
    /// </summary>
    public double Confidence { get; init; }

    /// <summary>
    /// The calibrated quantity: the probability mass on the answer being reported, so of the
    /// answers returned at confidence <c>c</c>, about <c>c</c> are right. This is what temperature
    /// scaling is fitted against and what ECE is measured on, on every question type.
    /// </summary>
    public double AnswerConfidence { get; init; }

    /// <summary>
    /// The <c>act_head</c> reading, class 0. Reported for completeness; see the caveat on
    /// <see cref="LayaCalibration.Decode"/>.
    /// </summary>
    public double ActionProbability { get; init; }

    public required LayaTemperature Temperature { get; init; }
}

/// <summary>
/// Temperature scaling, option-count bucketing, and answer decoding, ported from
/// <c>common.py</c> (<c>clamp_temperature</c>, <c>temp_bucket</c>, <c>answer_confidence</c>,
/// <c>confidence_from_probs</c>) and <c>agent.py</c> (<c>_decode_answers</c>).
/// </summary>
public sealed class LayaCalibration
{
    /// <summary>
    /// Lower bound on an applied temperature. A value below 1 sharpens logits rather than softening
    /// them, and the reference declines to ship one that hardens this much: the shipped
    /// <c>choice:11+</c> bucket of 0.1006 multiplies logits ~10x, so a 0.24 top probability would
    /// be published as 0.99 and a caller gating on confidence would be told a coin flip is a
    /// certainty. It lands on exactly the high-cardinality questions that already degrade.
    /// </summary>
    public const double TempMin = 0.5;

    /// <summary>Upper bound on an applied temperature.</summary>
    public const double TempMax = 5.0;

    readonly double[] temperaturePerType;
    readonly IReadOnlyDictionary<string, double> temperatureByOptions;

    public LayaCalibration(
        double[] temperaturePerType,
        IReadOnlyDictionary<string, double>? temperatureByOptions = null)
    {
        ArgumentNullException.ThrowIfNull(temperaturePerType);
        if (temperaturePerType.Length != 3)
            throw new ArgumentException(
                $"temperature must have one entry per question type (3), got {temperaturePerType.Length}",
                nameof(temperaturePerType));
        this.temperaturePerType = [.. temperaturePerType];
        this.temperatureByOptions = temperatureByOptions
            ?? new Dictionary<string, double>();
    }

    /// <summary>
    /// The <c>type:size</c> key a question's temperature is looked up under. Option-count buckets
    /// are what the shipped calibration refines per type.
    /// </summary>
    public static string TempBucket(LayaQuestionType type, int optionCount) => optionCount switch
    {
        <= 2 => string.Concat(LayaPromptBuilder.TypeName(type), ":2"),
        <= 5 => string.Concat(LayaPromptBuilder.TypeName(type), ":3-5"),
        <= 10 => string.Concat(LayaPromptBuilder.TypeName(type), ":6-10"),
        _ => string.Concat(LayaPromptBuilder.TypeName(type), ":11+")
    };

    /// <summary>
    /// Confines a configured temperature to [<paramref name="lo"/>, <paramref name="hi"/>], falling
    /// back to 1.0 for a value that is not a usable number.
    /// </summary>
    public static double ClampTemperature(double value, double lo = TempMin, double hi = TempMax)
    {
        if (double.IsNaN(value) || double.IsInfinity(value))
            return 1.0;
        return Math.Min(hi, Math.Max(lo, value));
    }

    /// <summary>Resolves the temperature to apply to one question, keeping the raw value visible.</summary>
    public LayaTemperature Resolve(LayaQuestionType type, int optionCount)
    {
        if (optionCount < 0)
            throw new ArgumentOutOfRangeException(nameof(optionCount), optionCount, null);

        string bucket = TempBucket(type, optionCount);
        double raw;
        if (temperatureByOptions.TryGetValue(bucket, out raw))
        {
            // The bucket table wins whenever it has an entry.
        }
        else
            raw = temperaturePerType[(int)type];

        double scale = ClampTemperature(raw);
        return new LayaTemperature(scale, bucket, raw, FromBucketTable: temperatureByOptions.ContainsKey(bucket), scale != raw);
    }

    /// <summary>
    /// The calibrated confidence: probability mass on the answer being reported.
    /// </summary>
    /// <remarks>
    /// A question with no options is vacuously fully decided, hence 1.0 for <c>k &lt; 1</c>.
    /// </remarks>
    public static double AnswerConfidence(IReadOnlyList<double> probabilities, int k)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        if (k < 1)
            return 1.0;

        double max = 0.0;
        for (int i = 0; i < Math.Min(k, probabilities.Count); i++)
            max = Math.Max(max, probabilities[i]);
        return Math.Clamp(max, 0.0, 1.0);
    }

    /// <summary>
    /// Normalized Shannon entropy: <c>1 - H(p) / log(k)</c>. How concentrated the whole
    /// distribution is. <b>Not calibrated</b> — it is neither what temperature scaling is fitted
    /// against nor what ECE measures, so it must not be compared against
    /// <see cref="AnswerConfidence"/>'s threshold.
    /// </summary>
    public static double ConfidenceFromProbs(IReadOnlyList<double> probabilities, int k)
    {
        ArgumentNullException.ThrowIfNull(probabilities);
        if (k < 2)
            return 1.0;

        double entropy = 0.0;
        for (int i = 0; i < Math.Min(k, probabilities.Count); i++)
        {
            double p = Math.Clamp(probabilities[i], 1e-12, 1.0);
            entropy -= p * Math.Log(p);
        }
        return Math.Clamp(1.0 - entropy / Math.Log(k), 0.0, 1.0);
    }

    /// <summary>
    /// Softmaxes the first <paramref name="k"/> logits at the given temperature, matching the
    /// reference's stabilized <c>exp(z - max) / sum</c> form.
    /// </summary>
    public static double[] SoftmaxAtTemperature(IReadOnlyList<float> logits, int k, double scale)
    {
        ArgumentNullException.ThrowIfNull(logits);
        if (k <= 0)
            return [];
        if (scale is 0.0 or double.NaN)
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "temperature must be non-zero");

        int n = Math.Min(k, logits.Count);
        var z = new double[n];
        double max = double.NegativeInfinity;
        for (int i = 0; i < n; i++)
        {
            z[i] = logits[i] / scale;
            if (z[i] > max)
                max = z[i];
        }

        var p = new double[n];
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            p[i] = Math.Exp(z[i] - max);
            sum += p[i];
        }
        for (int i = 0; i < n; i++)
            p[i] /= sum;
        return p;
    }

    /// <summary>
    /// Turns one question's logits into a typed decision, as <c>agent.py</c>'s
    /// <c>_decode_answers</c> does: divide by the resolved temperature, softmax, then read off the
    /// answer the question type defines.
    /// </summary>
    /// <param name="question">The question the logits belong to.</param>
    /// <param name="logits">The marker-scorer logits, one per option.</param>
    /// <param name="actProbability">
    /// The <c>act_head</c> class-0 probability.
    /// </param>
    /// <param name="roundTo">Decimal places for the reported figures. The reference rounds to 4.</param>
    /// <remarks>
    /// <b>On <see cref="LayaDecision.ActionProbability"/>.</b> It is reproduced faithfully and
    /// reported, but it is an upstream model limitation, not a Nivara defect: the head's action
    /// branch reads ~1.0 on the shipped checkpoint regardless of input, and the model card
    /// documents the same. Do not gate anything on it. The cheap check is whether it reads ~1.0 for
    /// arbitrary synthetic inputs — no eval data and no labels needed.
    /// </remarks>
    public LayaDecision Decode(
        LayaQuestion question,
        IReadOnlyList<float> logits,
        double actProbability,
        int roundTo = 4)
    {
        ArgumentNullException.ThrowIfNull(question);
        ArgumentNullException.ThrowIfNull(logits);

        var options = LayaPromptBuilder.RenderOptions(question);
        int k = options.Count;
        if (logits.Count < k)
            throw new ArgumentException(
                $"{question.Id}: {logits.Count} logits for {k} options", nameof(logits));

        var temperature = Resolve(question.Type, k);
        var p = SoftmaxAtTemperature(logits, k, temperature.Scale);
        double answerConfidence = Round(AnswerConfidence(p, k), roundTo);

        var decision = new LayaDecision
        {
            QuestionId = question.Id,
            Type = question.Type,
            AnswerConfidence = answerConfidence,
            ActionProbability = Round(actProbability, roundTo),
            Temperature = temperature
        };

        switch (question.Type)
        {
            case LayaQuestionType.Choice:
            {
                int best = 0;
                for (int i = 1; i < p.Length; i++)
                    if (p[i] > p[best])
                        best = i;
                // The key, not the rendered "key: description". The rendered text is prompt
                // content; the wheel's agent reports crit.keys()[argmax], and that is the answer
                // a caller acts on. They coincide only when the option has no description.
                return decision with
                {
                    Choice = question.Options[best].Key,
                    Probabilities = [.. p.Select(v => Round(v, roundTo))],
                    Confidence = Round(ConfidenceFromProbs(p, k), roundTo)
                };
            }

            case LayaQuestionType.Score:
            {
                double expected = 0.0;
                for (int i = 0; i < p.Length; i++)
                    expected += i * p[i];
                return decision with
                {
                    Score = Round(expected, roundTo),
                    Probabilities = [.. p.Select(v => Round(v, roundTo))],
                    Confidence = Round(ConfidenceFromProbs(p, k), roundTo)
                };
            }

            case LayaQuestionType.Noul:
            {
                // Two options by construction, so p[1] is P(true) and the semantic order is fixed.
                double noul = p.Length > 1 ? p[1] : 0.0;
                return decision with
                {
                    NoulProbability = Round(noul, roundTo),

                    // Over two options max(p, 1-p) is max(p), so this equals the answer
                    // confidence; the reference reports both to avoid changing one underneath
                    // existing callers.
                    Confidence = Round(Math.Max(noul, 1.0 - noul), roundTo)
                };
            }

            default:
                throw new ArgumentOutOfRangeException(nameof(question), question.Type, null);
        }
    }

    /// <summary>
    /// Rounds half-to-even, matching Python's <c>round()</c> for floats. C#'s
    /// <see cref="Math.Round(double, int)"/> already defaults to this; it is spelled out because a
    /// different midpoint mode would make the last digit disagree with the reference.
    /// </summary>
    static double Round(double value, int digits) =>
        Math.Round(value, digits, MidpointRounding.ToEven);
}
