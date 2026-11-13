using System.Threading;

namespace Nivara.Execution;

/// <summary>
/// Observes the producer task during streaming teardown so its failure is not
/// silently lost. Returns the producer's exception when it is a genuine fault
/// (to be surfaced alongside the consumer's), and <c>null</c> when it is an
/// expected cancellation or the very same exception that already reached the
/// consumer.
/// </summary>
internal static class StreamingProducerCleanup
{
    /// <summary>
    /// Key under which a secondary producer failure can be attached to a primary
    /// exception's <see cref="Exception.Data"/>. Exposed for tests/diagnostics.
    /// </summary>
    internal const string ProducerFailureDataKey = "Nivara.StreamingProducerFailure";

    /// <summary>
    /// Awaits <paramref name="producer"/>. Returns its exception when it is a
    /// genuine fault that should be surfaced alongside <paramref name="primary"/>;
    /// otherwise <c>null</c> (expected cancellation, or the same instance the
    /// consumer already observed).
    /// </summary>
    internal static async Task<Exception?> ObserveAsync(Task producer, Exception primary, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(producer);
        ArgumentNullException.ThrowIfNull(primary);

        Exception? failure;
        try
        {
            await producer.ConfigureAwait(false);
            return null;
        }
        catch (Exception ex)
        {
            failure = ex;
        }

        if (ReferenceEquals(failure, primary))
            return null;

        if (failure is OperationCanceledException &&
            (primary is OperationCanceledException || ct.IsCancellationRequested))
            return null;

        return failure;
    }
}
