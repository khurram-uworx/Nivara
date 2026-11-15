using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class AdaptiveDecisionScoringTests
{
    static AdaptiveCandidate C(int row, double fused, bool accepted, double conf) =>
        new(row, $"chunk-{row}", fused, accepted, conf);

    [Test]
    public void Rerank_AcceptedFirstThenConfidenceThenFusedThenRow()
    {
        var cands = new[]
        {
            C(0, 0.9, false, 0.0),
            C(1, 0.5, true, 0.7),
            C(2, 0.4, true, 0.9),
            C(3, 0.8, false, 0.0),
        };
        var order = AdaptiveDecisionScoring.Rerank(cands);
        Assert.That(order, Is.EqualTo(new[] { 2, 1, 0, 3 }));
    }

    [Test]
    public void Rerank_AllTied_OrdersByRow()
    {
        var cands = new[] { C(2, 0.5, false, 0.0), C(0, 0.5, false, 0.0), C(1, 0.5, false, 0.0) };
        var order = AdaptiveDecisionScoring.Rerank(cands);
        Assert.That(order, Is.EqualTo(new[] { 0, 1, 2 }));
    }

    [Test]
    public void Rerank_PreservesMembership()
    {
        var cands = new[] { C(0, 0.9, true, 0.8), C(1, 0.5, false, 0.2), C(2, 0.4, true, 0.6) };
        var order = AdaptiveDecisionScoring.Rerank(cands);
        Assert.That(order.OrderBy(x => x).ToArray(), Is.EqualTo(new[] { 0, 1, 2 }));
    }

    [Test]
    public void Filter_DropsRejected()
    {
        var cands = new[] { C(0, 0.9, true, 0.8), C(1, 0.5, false, 0.2), C(2, 0.4, true, 0.6) };
        var order = AdaptiveDecisionScoring.Filter(cands);
        Assert.That(order, Is.EqualTo(new[] { 0, 2 }));
    }

    [Test]
    public void Filter_NoAccepted_ReturnsEmpty()
    {
        var cands = new[] { C(0, 0.9, false, 0.2), C(1, 0.5, false, 0.1) };
        var order = AdaptiveDecisionScoring.Filter(cands);
        Assert.That(order, Is.Empty);
    }
}
