using System.Threading;

namespace Nivara.Execution;

/// <summary>
/// A byte-aware async resource governor. Unlike <see cref="StreamingBudgetTracker"/>,
/// which only observes accumulated memory, this primitive <em>controls</em> whether a
/// chunk may be allocated: callers must <see cref="AcquireAsync"/> capacity before
/// producing data and <see cref="Release"/> it once ownership transfers away.
/// </summary>
public sealed class MemoryBudget : IDisposable
{
    readonly long capacity;
    long current;
    readonly object gate = new();
    readonly List<Waiter> waiters = new();
    bool disposed;

    /// <summary>
    /// Initializes a new budget with the given capacity in bytes.
    /// </summary>
    /// <param name="capacityBytes">Maximum in-flight bytes. Must be positive.</param>
    public MemoryBudget(long capacityBytes)
    {
        if (capacityBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(capacityBytes), capacityBytes, "Capacity must be positive.");

        capacity = capacityBytes;
    }

    /// <summary>Maximum capacity in bytes.</summary>
    public long Capacity => capacity;

    /// <summary>Current estimated in-flight bytes.</summary>
    public long CurrentUsage
    {
        get
        {
            lock (gate) { return current; }
        }
    }

    /// <summary>
    /// Waits until the budget has room for <paramref name="requestedBytes"/>, then
    /// reserves that amount. Returns a completed <see cref="ValueTask"/> when capacity
    /// is immediately available (no allocation on the happy path).
    /// </summary>
    public ValueTask AcquireAsync(long requestedBytes, CancellationToken cancellationToken = default)
    {
        if (requestedBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(requestedBytes), requestedBytes, "Requested bytes must be positive.");
        if (requestedBytes > capacity)
            throw new ArgumentOutOfRangeException(nameof(requestedBytes), requestedBytes, "Requested bytes exceed budget capacity.");

        if (cancellationToken.IsCancellationRequested)
            return ValueTask.FromCanceled(cancellationToken);

        lock (gate)
        {
            ThrowIfDisposed();

            if (current + requestedBytes <= capacity)
            {
                current += requestedBytes;
                return ValueTask.CompletedTask;
            }

            var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var waiter = new Waiter(requestedBytes, tcs);

            waiters.Add(waiter);

            if (cancellationToken.CanBeCanceled)
            {
                var registration = cancellationToken.Register(static state =>
                {
                    var (budget, w, ct) = ((MemoryBudget, Waiter, CancellationToken))state!;
                    lock (budget.gate)
                    {
                        if (budget.waiters.Remove(w))
                            w.Completion.TrySetCanceled(ct);
                    }
                }, (this, waiter, cancellationToken));

                waiter.Registration = registration;
            }

            return new ValueTask(tcs.Task);
        }
    }

    /// <summary>
    /// Returns previously acquired bytes to the budget, unblocking any waiting acquirer
    /// whose request now fits.
    /// </summary>
    public void Release(long releasedBytes)
    {
        if (releasedBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(releasedBytes), releasedBytes, "Released bytes cannot be negative.");

        lock (gate)
        {
            if (disposed)
                return;

            current = Math.Max(0, current - releasedBytes);

            // Wake waiters in FIFO order; stop when the next request no longer fits so
            // that a large waiter cannot be starved by a stream of small ones.
            int i = 0;
            while (i < waiters.Count)
            {
                var waiter = waiters[i];
                if (current + waiter.RequestedBytes <= capacity)
                {
                    current += waiter.RequestedBytes;
                    waiters.RemoveAt(i);
                    waiter.Registration?.Dispose();
                    waiter.Completion.TrySetResult();
                }
                else
                {
                    break;
                }
            }
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
                return;
            disposed = true;

            foreach (var waiter in waiters)
            {
                waiter.Registration?.Dispose();
                waiter.Completion.TrySetException(new ObjectDisposedException(nameof(MemoryBudget)));
            }
            waiters.Clear();
            current = 0;
        }
    }

    void ThrowIfDisposed()
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(MemoryBudget));
    }

    sealed class Waiter
    {
        public Waiter(long requestedBytes, TaskCompletionSource completion)
        {
            RequestedBytes = requestedBytes;
            Completion = completion;
        }

        public long RequestedBytes { get; }
        public TaskCompletionSource Completion { get; }
        public CancellationTokenRegistration? Registration { get; set; }
    }
}