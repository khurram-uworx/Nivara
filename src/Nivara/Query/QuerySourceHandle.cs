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
    /// Initializes a new instance wrapping <paramref name="source"/>.
    /// </summary>
    /// <param name="source">The source to own</param>
    /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
    internal QuerySourceHandle(IQuerySource source)
    {
        this.source = source ?? throw new ArgumentNullException(nameof(source));

        // Abandoned-resource cleanup is no longer tracked here (see #1168): explicit Dispose/Release
        // is the only release path for a lazy source, matching the one-source-one-release contract.
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
    /// Releases the source. Idempotent: a second call is a no-op. The release is
    /// considered complete even when IQuerySource.Dispose throws, so a retry is
    /// not attempted; the disposal failure propagates to the caller.
    /// </summary>
    internal void Release()
    {
        if (Interlocked.Exchange(ref released, 1) != 0) return;

        source.Dispose();
    }

    /// <summary>
    /// Releases the source asynchronously. Idempotent: a second call is a no-op.
    /// When source implements IAsyncDisposable the awaited DisposeAsync is used;
    /// otherwise the synchronous Dispose fallback runs. Whichever path runs, a
    /// disposal failure propagates to the caller; there is no fallback from a
    /// failed DisposeAsync to the synchronous Dispose.
    /// </summary>
    internal async ValueTask ReleaseAsync()
    {
        if (Interlocked.Exchange(ref released, 1) != 0) return;

        if (source is IAsyncDisposable asyncDisposable)
            await asyncDisposable.DisposeAsync().ConfigureAwait(false);
        else
            source.Dispose();
    }
}
