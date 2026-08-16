using Nivara.Execution;
using Nivara.Query;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class StreamingBackpressureTests
{
    [Test]
    public void CalculateChannelCapacity_ClampsToMinimumTwo()
    {
        Assert.That(StreamingExecutionStrategy.CalculateChannelCapacity(1, 1_000_000), Is.EqualTo(2));
        Assert.That(StreamingExecutionStrategy.CalculateChannelCapacity(0, 1_000), Is.EqualTo(2));
    }

    [Test]
    public void CalculateChannelCapacity_ClampsToMaximumSixteen()
    {
        const long budget = 10_000_000;
        const int chunkSize = 1_000;

        Assert.That(StreamingExecutionStrategy.CalculateChannelCapacity(budget, chunkSize), Is.EqualTo(16));
    }

    [Test]
    public void CalculateChannelCapacity_ShrinkingBudget_NeverIncreasesCapacity()
    {
        const int chunkSize = 1_000;
        long[] budgets = { 16_000_000, 8_000_000, 4_000_000, 2_000_000, 1_000_000, 500_000, 200_000, 100_000 };

        int previous = int.MaxValue;
        foreach (var budget in budgets)
        {
            var capacity = StreamingExecutionStrategy.CalculateChannelCapacity(budget, chunkSize);
            Assert.That(capacity, Is.LessThanOrEqualTo(previous), $"capacity increased when budget shrank to {budget}");
            previous = capacity;
        }
    }

    [Test]
    public void CalculateChannelCapacity_GrowingChunkSize_NeverIncreasesCapacity()
    {
        const long budget = 16_000_000;
        int[] chunkSizes = { 1_000, 2_000, 4_000, 8_000, 16_000 };

        int previous = int.MaxValue;
        foreach (var chunkSize in chunkSizes)
        {
            var capacity = StreamingExecutionStrategy.CalculateChannelCapacity(budget, chunkSize);
            Assert.That(capacity, Is.LessThanOrEqualTo(previous), $"capacity increased when chunk size grew to {chunkSize}");
            previous = capacity;
        }
    }

    [Test]
    public void CalculateChannelCapacity_BothKnobsInfluenceBound()
    {
        const int chunkSize = 1_000;

        Assert.That(StreamingExecutionStrategy.CalculateChannelCapacity(800_000, chunkSize), Is.EqualTo(8));
        Assert.That(StreamingExecutionStrategy.CalculateChannelCapacity(400_000, chunkSize), Is.EqualTo(4));
        Assert.That(StreamingExecutionStrategy.CalculateChannelCapacity(800_000, 2_000), Is.EqualTo(4));
    }

    [Test]
    public async Task CreateBoundChannel_UnderLoad_ProducerBlocksAndNeverExceedsCapacity()
    {
        const long budget = 800_000;
        const int chunkSize = 1_000;
        var capacity = StreamingExecutionStrategy.CalculateChannelCapacity(budget, chunkSize);
        Assert.That(capacity, Is.EqualTo(8));

        var channel = StreamingExecutionStrategy.CreateBoundChannel(budget, chunkSize);
        const int frameCount = 64;

        var capacityReached = new TaskCompletionSource<bool>();
        int accepted = 0;

        var producer = Task.Run(async () =>
        {
            for (int i = 0; i < frameCount; i++)
            {
                await channel.Writer.WriteAsync(CreateFrame(i)).ConfigureAwait(false);
                if (Interlocked.Increment(ref accepted) == capacity)
                    capacityReached.TrySetResult(true);
            }
            channel.Writer.TryComplete();
        });

        // Deterministic backpressure proof: with no consumer running yet, the fast
        // producer fills the channel to exactly its capacity and then blocks on the
        // (capacity+1)-th write. No timing assumption — the handshake fires only after
        // `capacity` writes have been accepted.
        await capacityReached.Task.WaitAsync(TimeSpan.FromSeconds(10)).ConfigureAwait(false);
        Assert.That(channel.Reader.Count, Is.EqualTo(capacity));
        Assert.That(producer.IsCompleted, Is.False);

        // Drain against the still-active producer (fast producer + slow consumer):
        // Reader.Count is the channel's own authoritative in-flight measure and must
        // never exceed the bound.
        var consumed = new List<int>();
        var peakBuffered = 0;
        await foreach (var frame in channel.Reader.ReadAllAsync())
        {
            await Task.Delay(5).ConfigureAwait(false);
            consumed.Add((int)frame.GetColumn("A").GetValue(0)!);
            peakBuffered = Math.Max(peakBuffered, channel.Reader.Count);
            frame.Dispose();
        }

        await producer.ConfigureAwait(false);

        Assert.That(consumed, Is.EqualTo(Enumerable.Range(0, frameCount)));
        Assert.That(peakBuffered, Is.LessThanOrEqualTo(capacity));
    }

    [Test]
    public async Task ExecuteAsync_ChunkedSource_TinyBudget_ReturnsFullResult()
    {
        var strategy = new StreamingExecutionStrategy();
        var plan = new QueryPlan(
            new StubChunkedQuerySource(5000),
            new IQueryOperation[] { new StubQueryOperation(OperationType.Filter) });
        var context = new NivaraExecutionContext(ExecutionStrategy.Streaming)
        {
            MemoryBudget = 500_000,
            ChunkSize = 1_000,
        };

        using var result = await strategy.ExecuteAsync(plan, context);

        Assert.That(result.RowCount, Is.EqualTo(5000));
        Assert.That(result.GetColumn("A").GetValue(0), Is.EqualTo(0));
        Assert.That(result.GetColumn("A").GetValue(4999), Is.EqualTo(4999));
    }

    static NivaraFrame CreateFrame(int id)
        => NivaraFrame.Create(("A", NivaraColumn<int>.Create(new[] { id })));
}
