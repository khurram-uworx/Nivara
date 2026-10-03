using Nivara.Helpers;

namespace Nivara.Query;

/// <summary>
/// Owns the single <see cref="IQuerySource"/> behind a chain of <see cref="QueryFrame"/> instances.
/// </summary>
/// <remarks>
/// A derived frame is a temporary produced inside an expression (<c>q.Where(...).Collect()</c>) and
/// is normally never disposed, so ownership cannot be counted per frame - the count would never
/// return to zero and the source would leak. Instead exactly one handle exists per source and every
/// frame in the chain shares it, giving one resource one release, the same contract
/// <see cref="System.IO.FileStream"/> has. Disposing any frame releases the source for the whole
/// chain.
/// </remarks>
internal sealed class QuerySourceHandle
{
    readonly IQuerySource source;
    int released;

    /// <summary>
    /// Initializes a new instance wrapping <paramref name="source"/>, registering it for
    /// abandoned-resource cleanup when the source is lazy.
    /// </summary>
    /// <param name="source">The source to own</param>
    /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
    internal QuerySourceHandle(IQuerySource source)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));

        // Track lazy sources for abandoned resource cleanup (opt-in via NivaraResourceManager.Enable).
        // The closure captures `source` and must never capture `this`: the tracked cleanup actions
        // live in a static dictionary, so anything they capture stays reachable forever and the weak
        // reference could never clear, silently disabling cleanup.
        if (source.IsLazy && NivaraResourceManager.IsEnabled)
        {
            NivaraResourceManager.TrackResource(this, "LazyQuerySource", 0, () =>
            {
                try
                {
                    source.Dispose();
                }
                catch
                {
                    // Ignore disposal errors for abandoned resources
                }
            });
        }
    }

    /// <summary>
    /// Gets the owned source. Remains readable after release; callers guard on
    /// <see cref="Released"/> before using it.
    /// </summary>
    internal IQuerySource Source => source;

    /// <summary>
    /// Gets a value indicating whether the source has been released.
    /// </summary>
    internal bool Released => Volatile.Read(ref released) != 0;

    /// <summary>
    /// Releases the source. Idempotent; disposal errors are swallowed.
    /// </summary>
    internal void Release()
    {
        if (Interlocked.Exchange(ref released, 1) != 0) return;

        NivaraResourceManager.UntrackResource(this);

        try
        {
            source.Dispose();
        }
        catch
        {
            // Ignore disposal errors, mirroring the abandoned-resource cleanup path.
        }
    }

    /// <summary>
    /// Releases the source asynchronously. Idempotent; disposal errors propagate.
    /// </summary>
    internal async ValueTask ReleaseAsync()
    {
        if (Interlocked.Exchange(ref released, 1) != 0) return;

        NivaraResourceManager.UntrackResource(this);

        if (source is IAsyncDisposable asyncDisposable)
            await asyncDisposable.DisposeAsync().ConfigureAwait(false);
        else
            source.Dispose();
    }
}