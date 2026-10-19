namespace Nivara.Samples.Incident;

/// <summary>
/// Records the outcome of a profitability-gated GPU offload decision for a single kernel.
/// </summary>
public sealed record GpuOffloadDecision(
    bool Offloaded,
    string Reason,
    double CpuMs,
    double GpuMs,
    long TransferBytes,
    string Kernel);
