using Nivara.Samples;
using NUnit.Framework;

namespace Nivara.Tests.AutoDiff;

/// <summary>
/// Pins temperature resolution and answer decoding against <c>common.py</c>
/// (<c>clamp_temperature</c>, <c>temp_bucket</c>, <c>answer_confidence</c>,
/// <c>confidence_from_probs</c>) and <c>agent.py</c>'s <c>_decode_answers</c>.
/// </summary>
[TestFixture]
public class LayaCalibrationTests
{
    /// <summary>The shipped per-type temperatures, from <c>rl_agent_config.json</c>.</summary>
    static readonly double[] ShippedPerType = [1.6369, 1.2514, 1.9834];

    /// <summary>The shipped bucket table. <c>choice:11+</c> is the out-of-range entry.</summary>
    static Dictionary<string, double> ShippedBuckets() => new()
    {
        ["choice:2"] = 1.9064,
        ["choice:3-5"] = 1.7602,
        ["choice:6-10"] = 1.00002,
        ["choice:11+"] = 0.10058,
        ["score:3-5"] = 1.2514,
        ["noul:2"] = 1.9834
    };

    [Test]
    public void TempBucket_MapsOptionCountsToTheShippedBucketNames()
    {
        Assert.Multiple(() =>
        {
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Choice, 2), Is.EqualTo("choice:2"));
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Choice, 3), Is.EqualTo("choice:3-5"));
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Choice, 5), Is.EqualTo("choice:3-5"));
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Choice, 6), Is.EqualTo("choice:6-10"));
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Choice, 10), Is.EqualTo("choice:6-10"));
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Choice, 11), Is.EqualTo("choice:11+"));
            Assert.That(LayaCalibration.TempBucket(LayaQuestionType.Noul, 2), Is.EqualTo("noul:2"));
        });
    }

    [Test]
    public void ClampTemperature_ConfinesToRangeAndFallsBackToOne()
    {
        Assert.Multiple(() =>
        {
            Assert.That(LayaCalibration.ClampTemperature(0.10058), Is.EqualTo(0.5),
                "the shipped choice:11+ bucket sharpens logits and must be clamped, not applied");
            Assert.That(LayaCalibration.ClampTemperature(0.1), Is.EqualTo(LayaCalibration.TempMin));
            Assert.That(LayaCalibration.ClampTemperature(7.0), Is.EqualTo(LayaCalibration.TempMax));
            Assert.That(LayaCalibration.ClampTemperature(1.0), Is.EqualTo(1.0), "in range passes through");
            Assert.That(LayaCalibration.ClampTemperature(double.NaN), Is.EqualTo(1.0));
            Assert.That(LayaCalibration.ClampTemperature(double.PositiveInfinity), Is.EqualTo(1.0));
            Assert.That(LayaCalibration.ClampTemperature(double.NegativeInfinity), Is.EqualTo(1.0));
        });
    }

    [Test]
    public void Resolve_PrefersTheBucketTableOverThePerTypeFallback()
    {
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());

        var bucketed = calibration.Resolve(LayaQuestionType.Choice, 4);
        var fallback = calibration.Resolve(LayaQuestionType.Score, 3);

        Assert.Multiple(() =>
        {
            Assert.That(bucketed.Bucket, Is.EqualTo("choice:3-5"));
            Assert.That(bucketed.FromBucketTable, Is.True);
            Assert.That(bucketed.Scale, Is.EqualTo(1.7602));

            // score:3-5 happens to equal the per-type value, so the distinguishing assertion is
            // the provenance, not the number.
            Assert.That(fallback.FromBucketTable, Is.True, "score:3-5 is in the table");

            var absent = calibration.Resolve(LayaQuestionType.Score, 7);
            Assert.That(absent.Bucket, Is.EqualTo("score:6-10"));
            Assert.That(absent.FromBucketTable, Is.False, "no score:6-10 entry, so it falls back");
            Assert.That(absent.Scale, Is.EqualTo(1.2514));
        });
    }

    [Test]
    public void Resolve_ClampsTheShippedHighCardinalityChoiceBucketAndKeepsTheRawValue()
    {
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());
        var t = calibration.Resolve(LayaQuestionType.Choice, 13);

        Assert.Multiple(() =>
        {
            Assert.That(t.Bucket, Is.EqualTo("choice:11+"));
            Assert.That(t.WasClamped, Is.True);
            Assert.That(t.Scale, Is.EqualTo(LayaCalibration.TempMin), "clamped up to the floor");
            Assert.That(t.RawValue, Is.EqualTo(0.10058).Within(1e-6),
                "the rejected value stays visible rather than being silently replaced");
        });
    }

    [Test]
    public void SoftmaxAtTemperature_ProducesAUnitDistributionAndShiftsWithScale()
    {
        var logits = new[] { 1.0f, 2.0f, 3.0f };

        var neutral = LayaCalibration.SoftmaxAtTemperature(logits, 3, 1.0);
        var softened = LayaCalibration.SoftmaxAtTemperature(logits, 3, 5.0);

        Assert.Multiple(() =>
        {
            Assert.That(neutral.Sum(), Is.EqualTo(1.0).Within(1e-12));
            Assert.That(neutral, Is.EqualTo(new[] { 0.09003057, 0.24472847, 0.66524094 })
                .Within(1e-6), "softmax of [1,2,3]");
            Assert.That(softened[0], Is.GreaterThan(neutral[0]),
                "a temperature above 1 pulls the distribution toward uniform");
        });
    }

    [Test]
    public void AnswerConfidence_IsTheProbabilityMassOnTheAnswer()
    {
        var p = new[] { 0.1, 0.2, 0.7 };

        Assert.Multiple(() =>
        {
            Assert.That(LayaCalibration.AnswerConfidence(p, 3), Is.EqualTo(0.7));
            Assert.That(LayaCalibration.AnswerConfidence(p, 2), Is.EqualTo(0.2),
                "only the first k are considered");
            Assert.That(LayaCalibration.AnswerConfidence(p, 0), Is.EqualTo(1.0), "k < 1 is vacuous");
        });
    }

    [Test]
    public void ConfidenceFromProbs_IsOneForAKnownAnswerAndZeroForAUniformDistribution()
    {
        Assert.Multiple(() =>
        {
            Assert.That(LayaCalibration.ConfidenceFromProbs([1.0], 1), Is.EqualTo(1.0));
            Assert.That(LayaCalibration.ConfidenceFromProbs([0.5, 0.5], 2),
                Is.EqualTo(1.0 - 1.0).Within(1e-12), "uniform over two is maximal entropy");
            Assert.That(LayaCalibration.ConfidenceFromProbs([0.5, 0.5], 1), Is.EqualTo(1.0), "k < 2");
            Assert.That(LayaCalibration.ConfidenceFromProbs([0.25, 0.25, 0.25, 0.25], 4),
                Is.EqualTo(0.0).Within(1e-12), "uniform over four is maximal entropy");
        });
    }

    [Test]
    public void Decode_Choice_ReportsTheArgmaxKeyAndItsCalibratedConfidence()
    {
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Choice,
            Instructions = "pick",
            Options = [new("alpha", null), new("beta", null), new("gamma", null)]
        };

        var d = calibration.Decode(q, [0.1f, 5.0f, 0.2f], actProbability: 0.99);

        Assert.Multiple(() =>
        {
            Assert.That(d.Choice, Is.EqualTo("beta"));
            Assert.That(d.Probabilities, Has.Count.EqualTo(3), "one probability per option");
            Assert.That(d.Probabilities.Sum(), Is.EqualTo(1.0).Within(1e-3), "rounded to 4dp");
            Assert.That(d.AnswerConfidence, Is.EqualTo(d.Probabilities.Max()));
            Assert.That(d.Temperature.Bucket, Is.EqualTo("choice:3-5"));
        });
    }

    [Test]
    public void Decode_Choice_ReportsTheKeyNotTheRenderedDescription()
    {
        // The wheel's agent reports crit.keys()[argmax]. "other: not a password question" is the
        // prompt text, and reporting it would make the typed decision disagree with the reference
        // on every described option.
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Choice,
            Instructions = "pick",
            Options = [new("reset", null), new("other", "not a password question")]
        };

        var d = calibration.Decode(q, [0.1f, 5.0f], actProbability: 0.5);

        Assert.That(d.Choice, Is.EqualTo("other"));
        Assert.That(d.Choice, Is.Not.EqualTo("other: not a password question"));
    }

    [Test]
    public void Decode_Score_ReportsTheExpectationOverOptionIndices()
    {
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Score,
            Instructions = "how urgent",
            ScoreCriteria = ["low", "medium", "high"]
        };

        var low = calibration.Decode(q, [2f, 0f, 0f], actProbability: 0.0);
        var high = calibration.Decode(q, [0f, 0f, 2f], actProbability: 0.0);

        // The shipped score temperature is 1.2514, so the distribution is softened rather than
        // one-hot: the expectation is a weighted mean, not the argmax.
        double weighted = 0.0;
        for (int i = 0; i < low.Probabilities.Count; i++)
            weighted += i * low.Probabilities[i];

        Assert.Multiple(() =>
        {
            Assert.That(low.Score, Is.EqualTo(weighted).Within(1e-3),
                "the score is the expectation over the reported probabilities");
            Assert.That(high.Score, Is.GreaterThan(low.Score!.Value), "a later peak scores higher");
            Assert.That(low.Score, Is.InRange(0.0, 2.0));
            Assert.That(low.Choice, Is.Null);
            Assert.That(low.Probabilities, Has.Count.EqualTo(3));
        });
    }

    [Test]
    public void Decode_Noul_ReportsPTrueAndConfidenceAsTheLargerSide()
    {
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());
        var q = new LayaQuestion { Id = "q", Type = LayaQuestionType.Noul, Instructions = "resolved" };

        var d = calibration.Decode(q, [0f, 3f], actProbability: 0.0);
        var flipped = calibration.Decode(q, [3f, 0f], actProbability: 0.0);

        Assert.Multiple(() =>
        {
            Assert.That(d.NoulProbability, Is.GreaterThan(0.5), "the higher logit is option 1 = true");
            Assert.That(d.Confidence, Is.EqualTo(d.AnswerConfidence),
                "over two options max(p, 1-p) is max(p), so the two agree by construction");
            Assert.That(flipped.NoulProbability, Is.LessThan(0.5));
            Assert.That(d.Temperature.Bucket, Is.EqualTo("noul:2"));
        });
    }

    [Test]
    public void Decode_RejectsFewerLogitsThanOptions()
    {
        var calibration = new LayaCalibration(ShippedPerType, ShippedBuckets());
        var q = new LayaQuestion
        {
            Id = "q",
            Type = LayaQuestionType.Choice,
            Instructions = "pick",
            Options = [new("a", null), new("b", null), new("c", null)]
        };

        Assert.Throws<ArgumentException>(() => calibration.Decode(q, [0f, 1f], 0.0));
    }

    [Test]
    public void Constructor_RejectsAPerTypeTableOfTheWrongLength()
    {
        Assert.Throws<ArgumentException>(() => new LayaCalibration([1.0, 1.0]));
    }
}
