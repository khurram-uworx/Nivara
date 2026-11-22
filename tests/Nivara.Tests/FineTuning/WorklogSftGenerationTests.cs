using NivaraFineTuning;
using NUnit.Framework;

namespace Nivara.Tests.FineTuning;

[TestFixture]
public sealed class WorklogSftGenerationTests
{
    [Test]
    public void GenerateGreedyCached_StopsOnStopToken()
    {
        bool resetCalled = false;
        int[] prompt = [1, 2, 3];
        int[] prefillSeq = [];
        var cached = new List<(int Token, int Pos)>();

        int[] result = WorklogSftGeneration.GenerateGreedyCached(
            prefillArgmax: p => { prefillSeq = p.ToArray(); return 10; },
            cachedArgmax: (token, pos) => { cached.Add((token, pos)); return 99; },
            resetCache: () => resetCalled = true,
            promptIds: prompt,
            maxNewTokens: 8,
            stopIds: [99],
            maxSequenceLength: 128);

        Assert.That(resetCalled, Is.True);
        Assert.That(prefillSeq, Is.EqualTo(prompt));
        Assert.That(result, Is.EqualTo(new[] { 10 }));
        Assert.That(cached, Has.Count.EqualTo(1));
        Assert.That(cached[0], Is.EqualTo((10, 3)));
    }

    [Test]
    public void GenerateGreedyCached_GeneratesUpToMaxNewTokens()
    {
        int next = 10;
        var cached = new List<(int Token, int Pos)>();

        int[] result = WorklogSftGeneration.GenerateGreedyCached(
            prefillArgmax: _ => next++,
            cachedArgmax: (token, pos) => { cached.Add((token, pos)); return next++; },
            resetCache: () => { },
            promptIds: [1, 2],
            maxNewTokens: 4,
            stopIds: [],
            maxSequenceLength: 128);

        Assert.That(result, Is.EqualTo(new[] { 10, 11, 12, 13 }));
        Assert.That(cached.Select(c => c.Pos), Is.EqualTo(new[] { 2, 3, 4 }));
        Assert.That(cached.Select(c => c.Token), Is.EqualTo(new[] { 10, 11, 12 }));
    }

    [Test]
    public void GenerateGreedyCached_RespectsMaxSequenceLength()
    {
        int next = 5;
        int[] result = WorklogSftGeneration.GenerateGreedyCached(
            prefillArgmax: _ => next++,
            cachedArgmax: (_, _) => next++,
            resetCache: () => { },
            promptIds: [1, 2, 3],
            maxNewTokens: 10,
            stopIds: [],
            maxSequenceLength: 5);

        Assert.That(result.Length, Is.LessThanOrEqualTo(2));
        Assert.That(result, Is.EqualTo(new[] { 5, 6 }));
    }

    [Test]
    public void GenerateGreedyCached_EmptyPromptOrZeroMax_ReturnsEmpty()
    {
        int[] emptyPrompt = WorklogSftGeneration.GenerateGreedyCached(
            prefillArgmax: _ => 1,
            cachedArgmax: (_, _) => 2,
            resetCache: () => { },
            promptIds: [],
            maxNewTokens: 4,
            stopIds: [],
            maxSequenceLength: 128);

        int[] zeroMax = WorklogSftGeneration.GenerateGreedyCached(
            prefillArgmax: _ => 1,
            cachedArgmax: (_, _) => 2,
            resetCache: () => { },
            promptIds: [1],
            maxNewTokens: 0,
            stopIds: [],
            maxSequenceLength: 128);

        Assert.That(emptyPrompt, Is.Empty);
        Assert.That(zeroMax, Is.Empty);
    }

    [Test]
    public void GenerateGreedyCached_PrefillStopToken_ReturnsEmpty()
    {
        bool cachedCalled = false;
        int[] result = WorklogSftGeneration.GenerateGreedyCached(
            prefillArgmax: _ => 7,
            cachedArgmax: (_, _) => { cachedCalled = true; return 8; },
            resetCache: () => { },
            promptIds: [1, 2],
            maxNewTokens: 4,
            stopIds: [7],
            maxSequenceLength: 128);

        Assert.That(result, Is.Empty);
        Assert.That(cachedCalled, Is.False);
    }
}
