using System.Collections.Concurrent;
using System.Threading;
using Nivara.Linq;
using NUnit.Framework;

namespace Nivara.Tests.Query;

[TestFixture]
public class AsyncDeadlockTests
{
    sealed class TestRow
    {
        public int X { get; set; }
        public string Y { get; set; } = string.Empty;
    }

    sealed class SingleThreadSynchronizationContext : SynchronizationContext
    {
        readonly BlockingCollection<(SendOrPostCallback Callback, object? State)> queue = new();

        public override void Post(SendOrPostCallback d, object? state)
            => queue.Add((d, state));

        public override void Send(SendOrPostCallback d, object? state)
            => d(state);
    }

    [Test]
    public void Collect_DoesNotDeadlock_UnderSingleThreadedSynchronizationContext()
    {
        var frame = Nivara.NivaraFrame.Create(
            ("X", Nivara.NivaraColumn<int>.Create(new[] { 1, 2, 3, 4 })),
            ("Y", Nivara.NivaraColumn<string>.Create(new[] { "a", "b", "c", "d" })));

        var query = frame.Query<TestRow>().Where(r => r.X > 1);

        var context = new SingleThreadSynchronizationContext();
        var task = Task.Factory.StartNew(() =>
        {
            SynchronizationContext.SetSynchronizationContext(context);
            return query.Collect();
        }, CancellationToken.None, TaskCreationOptions.None, TaskScheduler.Default);

        Assert.That(task.Wait(TimeSpan.FromSeconds(5)), Is.True, "Collect() deadlocked under a single-threaded SynchronizationContext");
        Assert.That(task.Result.RowCount, Is.EqualTo(3));
    }
}
