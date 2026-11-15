using NivaraChat.Helpers;
using NUnit.Framework;

namespace Nivara.Tests.Samples;

[TestFixture]
public sealed class RagDecisionRoutingTests
{
    static LayaChunkDecision Decision(
        string chunkId,
        double answerConfidence,
        double retrievalScore,
        bool accepted)
        => new(
            chunkId,
            "source",
            "title",
            "content",
            accepted ? "accept" : "fallback",
            answerConfidence,
            answerConfidence,
            retrievalScore,
            accepted,
            !accepted,
            "reason");

    [Test]
    public void SelectBestDecisionFrom_PrefersAcceptedDecisionWithHighestConfidence()
    {
        var fallback = Decision("ranked-first", 0.99, 10, false);
        var lowerConfidence = Decision("accepted-lower", 0.7, 9, true);
        var highestConfidence = Decision("accepted-highest", 0.8, 1, true);

        var selected = RagDecisionRouting.SelectBestDecisionFrom(
            [fallback, lowerConfidence, highestConfidence]);

        Assert.That(selected, Is.SameAs(highestConfidence));
    }

    [Test]
    public void SelectBestDecisionFrom_UsesRetrievalScoreWhenNoDecisionIsAccepted()
    {
        var lowerRank = Decision("lower-rank", 0.9, 0.5, false);
        var higherRank = Decision("higher-rank", 0.2, 0.8, false);

        var selected = RagDecisionRouting.SelectBestDecisionFrom([lowerRank, higherRank]);

        Assert.That(selected, Is.SameAs(higherRank));
    }

    [Test]
    public void SelectBestDecisionFrom_RejectsEmptyDecisions()
        => Assert.Throws<ArgumentException>(() => RagDecisionRouting.SelectBestDecisionFrom([]));
}
