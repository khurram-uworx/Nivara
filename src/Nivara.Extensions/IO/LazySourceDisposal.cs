using System.Runtime.ExceptionServices;

namespace Nivara.IO;

/// <summary>
/// Shared helper for releasing a lazily-opened resource (e.g. a Parquet reader)
/// while always releasing the gate that serialises access to it. A close failure
/// propagates to the caller instead of being swallowed.
/// </summary>
internal static class LazySourceDisposal
{
    /// <summary>
    /// Closes <paramref name="closeReader"/> and releases <paramref name="releaseGate"/>
    /// afterwards. <paramref name="releaseGate"/> always runs, even when the close
    /// throws; the close failure (if any) is rethrown via
    /// <see cref="ExceptionDispatchInfo"/>.
    /// </summary>
    internal static void DisposeReader(Func<ValueTask> closeReader, Action releaseGate)
    {
        ArgumentNullException.ThrowIfNull(closeReader);
        ArgumentNullException.ThrowIfNull(releaseGate);

        Exception? failure = null;
        try
        {
            closeReader().AsTask().GetAwaiter().GetResult();
        }
        catch (Exception ex)
        {
            failure = ex;
        }

        releaseGate();

        if (failure is not null)
            ExceptionDispatchInfo.Capture(failure).Throw();
    }
}
