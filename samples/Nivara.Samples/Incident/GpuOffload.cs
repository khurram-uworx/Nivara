namespace Nivara.Samples.Incident;

/// <summary>
/// Records the outcome of a profitability-gated GPU offload decision for a single kernel.
/// </summary>
/// <param name="Offloaded">True when the kernel actually ran on the GPU.</param>
/// <param name="Reason">Human-readable decision rationale or failure cause.</param>
/// <param name="CpuMs">
/// Measured CPU time when the CPU path ran. When <paramref name="Offloaded"/> is true this is
/// the CPU cost the profitability gate predicted for the work the GPU path replaced, so the
/// gate's estimate is visible next to the GPU time it was compared against.
/// </param>
/// <param name="GpuMs">Measured GPU time, or 0 when the GPU path did not run.</param>
/// <param name="TransferBytes">Host-to-device input bytes the GPU path would move.</param>
/// <param name="Kernel">Identifier of the offloaded kernel.</param>
public sealed record GpuOffloadDecision(
    bool Offloaded,
    string Reason,
    double CpuMs,
    double GpuMs,
    long TransferBytes,
    string Kernel);
