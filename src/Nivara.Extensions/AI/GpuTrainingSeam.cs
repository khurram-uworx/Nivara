namespace Nivara.AI;

/// <summary>
/// Seam for a future GPU training backend for the NivaraClassifier sample models.
/// The concrete implementation is intentionally deferred (see GitHub issue #883). No
/// runtime currently implements it; the HTTP/CLI surface rejects <c>--silk</c> until one does.
/// </summary>
public interface ITextClassifierGpuTrainer : IDisposable
{
    string DeviceName { get; }
}

/// <summary>
/// Seam for a future GPU prediction backend for the NivaraClassifier sample models.
/// </summary>
public interface ITextClassifierGpuPredictor : IDisposable
{
    string DeviceName { get; }
}
