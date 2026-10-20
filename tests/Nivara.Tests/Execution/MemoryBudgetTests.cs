using Nivara.Execution;
using NUnit.Framework;

namespace Nivara.Tests.Execution;

[TestFixture]
public class MemoryBudgetTests
{
    [Test]
    public void Constructor_InvalidCapacity_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new MemoryBudget(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new MemoryBudget(-1));
    }

    [Test]
    public async Task AcquireAsync_SingleRequestExceedingCapacity_Throws()
    {
        var budget = new MemoryBudget(100);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await budget.AcquireAsync(200));
    }

    [Test]
    public async Task AcquireRelease_HappyPath_UpdatesCurrentUsage()
    {
        using var budget = new MemoryBudget(100);
        Assert.That(budget.CurrentUsage, Is.EqualTo(0));
        Assert.That(budget.Capacity, Is.EqualTo(100));

        await budget.AcquireAsync(40);
        Assert.That(budget.CurrentUsage, Is.EqualTo(40));

        budget.Release(40);
        Assert.That(budget.CurrentUsage, Is.EqualTo(0));
    }

    [Test]
    public async Task AcquireAsync_BlocksUntilRelease()
    {
        using var budget = new MemoryBudget(100);
        await budget.AcquireAsync(100);

        var acquired = new TaskCompletionSource<bool>();
        var waiter = Task.Run(async () =>
        {
            await budget.AcquireAsync(50);
            acquired.TrySetResult(true);
        });

        // Give the waiter a chance to start; it should not have acquired.
        await Task.Delay(50);
        Assert.That(acquired.Task.IsCompleted, Is.False);
        Assert.That(budget.CurrentUsage, Is.EqualTo(100));

        budget.Release(100);
        await waiter.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.That(acquired.Task.IsCompleted, Is.True);
        Assert.That(budget.CurrentUsage, Is.EqualTo(50));
    }

    [Test]
    public async Task AcquireAsync_Cancellation_UnblocksAndWaiterRemoved()
    {
        using var budget = new MemoryBudget(100);
        await budget.AcquireAsync(100);

        using var cts = new CancellationTokenSource();
        var task = budget.AcquireAsync(50, cts.Token).AsTask();

        cts.Cancel();
        await Assert.ThrowsAsync<TaskCanceledException>(async () => await task);

        // Capacity unchanged; the cancelled waiter must not still hold a reservation.
        Assert.That(budget.CurrentUsage, Is.EqualTo(100));
        budget.Release(100);
        Assert.That(budget.CurrentUsage, Is.EqualTo(0));
    }

    [Test]
    public void Release_Negative_Throws()
    {
        using var budget = new MemoryBudget(100);
        Assert.Throws<ArgumentOutOfRangeException>(() => budget.Release(-1));
    }

    [Test]
    public async Task Dispose_UnblocksWaitersWithObjectDisposed()
    {
        var budget = new MemoryBudget(100);
        budget.AcquireAsync(100).GetAwaiter().GetResult();

        var task = budget.AcquireAsync(50).AsTask();
        budget.Dispose();

        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await task);
    }

    [Test]
    public async Task AcquireAsync_AfterDispose_Throws()
    {
        var budget = new MemoryBudget(100);
        budget.Dispose();
        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await budget.AcquireAsync(10));
    }
}
