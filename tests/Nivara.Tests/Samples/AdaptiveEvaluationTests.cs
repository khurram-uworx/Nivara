using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveEvaluationTests
{
    static AdaptiveObservation Obs(int grade, bool choice, double conf, bool accepted = false) =>
        new("q", "c", grade, choice, accepted, conf);

    [Test]
    public void BinIndex_ConfidenceOne_LandsInLastBin()
    {
        Assert.That(AdaptiveEvaluation.BinIndex(1.0, 10), Is.EqualTo(9));
        Assert.That(AdaptiveEvaluation.BinIndex(0.0, 10), Is.EqualTo(0));
        Assert.That(AdaptiveEvaluation.BinIndex(0.95, 10), Is.EqualTo(9));
    }

    [Test]
    public void BinIndex_RejectsOutOfRange()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => AdaptiveEvaluation.BinIndex(1.1, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => AdaptiveEvaluation.BinIndex(-0.1, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => AdaptiveEvaluation.BinIndex(0.5, 0));
    }

    [Test]
    public void Calibrate_PerfectlyCalibrated_ReturnsZeroEce()
    {
        var obs = new List<AdaptiveObservation>();
        for (var i = 0; i < 5; i++) obs.Add(Obs(1, true, 0.8));
        for (var i = 0; i < 5; i++) obs.Add(Obs(0, false, 0.2));
        // force two bins: 0.8 bin accuracy 1.0, 0.2 bin accuracy 0.0 (not_relevant correct when grade 0)
        // Rebuild so each bin is internally consistent: conf 0.8 with 100% correct
        obs = [.. Enumerable.Range(0, 10).Select(_ => Obs(1, true, 1.0))];
        var cal = AdaptiveEvaluation.Calibrate(obs);
        Assert.That(cal.Ece, Is.EqualTo(0.0).Within(1e-9));
        Assert.That(cal.Brier, Is.EqualTo(0.0).Within(1e-9));
    }

    [Test]
    public void Calibrate_Overconfident_ReturnsPositiveEce()
    {
        var obs = Enumerable.Range(0, 10).Select(_ => Obs(0, true, 0.9)).ToList();
        var cal = AdaptiveEvaluation.Calibrate(obs);
        Assert.That(cal.Ece, Is.GreaterThan(0.8));
        Assert.That(cal.Brier, Is.GreaterThan(0.8));
    }

    [Test]
    public void Calibrate_IgnoresEmptyBins()
    {
        var obs = Enumerable.Range(0, 5).Select(_ => Obs(1, true, 0.3)).ToList();
        var cal = AdaptiveEvaluation.Calibrate(obs, bins: 10);
        Assert.That(cal.Bins.Count(b => b.Count == 0), Is.GreaterThan(0));
        Assert.That(cal.Bins.Where(b => b.Count == 0).All(b => b.Gap == 0.0), Is.True);
        Assert.That(double.IsFinite(cal.Ece), Is.True);
    }

    [Test]
    public void Calibrate_CountsObservations()
    {
        var obs = Enumerable.Range(0, 7).Select(_ => Obs(1, true, 0.5)).ToList();
        var cal = AdaptiveEvaluation.Calibrate(obs);
        Assert.That(cal.Count, Is.EqualTo(7));
        Assert.That(cal.Bins.Sum(b => b.Count), Is.EqualTo(7));
    }

    [Test]
    public void Decisions_SeparateChoiceFromAccept()
    {
        var obs = new List<AdaptiveObservation>
        {
            new("q", "a", 1, true, true, 0.9),
            new("q", "b", 0, true, false, 0.4),   // choice wrong, not accepted
            new("q", "c", 1, false, false, 0.3),   // choice wrong
        };
        var m = AdaptiveEvaluation.Decisions(obs);
        Assert.That(m.Count, Is.EqualTo(3));
        Assert.That(m.Correct, Is.EqualTo(1));
        Assert.That(m.Accuracy, Is.EqualTo(1.0 / 3.0).Within(1e-9));
    }

    [Test]
    public void Calibrate_RejectsEmptyInput()
    {
        Assert.Throws<ArgumentException>(() => AdaptiveEvaluation.Calibrate([]));
    }
}
