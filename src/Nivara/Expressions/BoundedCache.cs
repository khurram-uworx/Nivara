using System.Collections.Concurrent;

namespace Nivara.Expressions;

/// <summary>
/// A bounded, thread-safe cache with an approximate LRU eviction policy implemented as two
/// generations of <see cref="ConcurrentDictionary{TKey, TValue}"/>: <c>current</c> and
/// <c>previous</c>. On a hit in <c>previous</c> the entry is promoted into <c>current</c>;
/// when <c>current</c> exceeds <see cref="Capacity"/> it becomes the new <c>previous</c> and a
/// fresh <c>current</c> is started, so the cache never holds more than
/// <c>2 × Capacity</c> entries. Rotations happen roughly once per <see cref="Capacity"/>
/// inserts and are the only operation that takes a lock; lookups are lock-free.
/// </summary>
/// <remarks>
/// The <paramref name="factory"/> passed to <see cref="GetOrAdd"/> may run more than once under
/// contention, and a delegate that loses the race is discarded. This is the same contract as
/// <see cref="ConcurrentDictionary{TKey, TValue}.GetOrAdd(TKey, Func{TKey, TValue})"/> and is
/// acceptable for caches of compiled delegates, where a duplicate build is wasted work rather
/// than a correctness problem.
/// </remarks>
/// <typeparam name="TKey">The cache key type (must not be null).</typeparam>
/// <typeparam name="TValue">The cached value type.</typeparam>
internal sealed class BoundedCache<TKey, TValue> where TKey : notnull
{
    readonly int capacity;
    readonly object rotationGate = new();

    ConcurrentDictionary<TKey, TValue> current = new();
    ConcurrentDictionary<TKey, TValue> previous = new();
    long overflowCount;

    /// <summary>
    /// Initializes a new bounded cache with the given per-generation capacity.
    /// </summary>
    /// <param name="capacity">Maximum number of entries held in one generation.</param>
    public BoundedCache(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        this.capacity = capacity;
    }

    /// <summary>
    /// Gets the capacity of a single generation.
    /// </summary>
    public int Capacity => capacity;

    /// <summary>
    /// Gets the total number of entries currently held across both generations.
    /// </summary>
    public int Count => Volatile.Read(ref current).Count + Volatile.Read(ref previous).Count;

    /// <summary>
    /// Gets the number of times the cache has rotated (overflowed) since creation or last
    /// <see cref="Clear"/>.
    /// </summary>
    public long OverflowCount => Interlocked.Read(ref overflowCount);

    /// <summary>
    /// Returns the cached value for <paramref name="key"/>, creating it via
    /// <paramref name="factory"/> on a miss.
    /// </summary>
    public TValue GetOrAdd(TKey key, Func<TKey, TValue> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);

        var live = Volatile.Read(ref current);
        if (live.TryGetValue(key, out var value))
            return value;

        var stale = Volatile.Read(ref previous);
        if (stale.TryGetValue(key, out value))
        {
            live.TryAdd(key, value);
            return value;
        }

        value = factory(key);
        live.TryAdd(key, value);
        RotateIfFull();
        return value;
    }

    /// <summary>
    /// Removes every entry and resets the overflow counter.
    /// </summary>
    public void Clear()
    {
        lock (rotationGate)
        {
            Volatile.Write(ref current, new ConcurrentDictionary<TKey, TValue>());
            Volatile.Write(ref previous, new ConcurrentDictionary<TKey, TValue>());
            Interlocked.Exchange(ref overflowCount, 0);
        }
    }

    void RotateIfFull()
    {
        if (Volatile.Read(ref current).Count <= capacity)
            return;

        lock (rotationGate)
        {
            if (current.Count <= capacity)
                return;

            Volatile.Write(ref previous, current);
            Volatile.Write(ref current, new ConcurrentDictionary<TKey, TValue>());
            Interlocked.Increment(ref overflowCount);
        }
    }
}