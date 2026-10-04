using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class StreamingChunkSizeTests
{
    [Test]
    public void CalculateChunkSize_DefaultOneGigabyteBudget_ClampsToMaximum()
    {
        var chunkSize = StreamingExecutionStrategy.CalculateChunkSize(1024L * 1024 * 1024);

        Assert.That(chunkSize, Is.EqualTo(100_000));
    }

    [Test]
    public void CalculateChunkSize_BudgetInsideClampRange_MatchesFormulaExactly()
    {
        var chunkSize = StreamingExecutionStrategy.CalculateChunkSize(67_108_864);

        Assert.That(chunkSize, Is.EqualTo(67_108));
    }

    [Test]
    public void CalculateChunkSize_TinyBudget_ClampsToMinimum()
    {
        Assert.That(StreamingExecutionStrategy.CalculateChunkSize(1), Is.EqualTo(1_000));
        Assert.That(StreamingExecutionStrategy.CalculateChunkSize(0), Is.EqualTo(1_000));
    }

    [Test]
    public void CalculateChunkSize_MaxValueBudget_ClampsToMaximum()
    {
        var chunkSize = StreamingExecutionStrategy.CalculateChunkSize(long.MaxValue);

        Assert.That(chunkSize, Is.EqualTo(100_000));
    }

    [Test]
    public void CalculateChunkSize_GrowingBudget_NeverDecreasesChunkSize()
    {
        long[] budgets = { 1, 1_000_000, 16_777_216, 67_108_864, 100_000_000, 2_147_483_647_000, 4_294_967_296_000, long.MaxValue };

        int previous = int.MinValue;
        foreach (var budget in budgets)
        {
            var chunkSize = StreamingExecutionStrategy.CalculateChunkSize(budget);
            Assert.That(chunkSize, Is.GreaterThanOrEqualTo(previous), $"chunk size decreased when budget grew to {budget}");
            previous = chunkSize;
        }
    }
}