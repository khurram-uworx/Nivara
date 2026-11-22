namespace Nivara.Silk;

/// <summary>
/// Accumulated profiling snapshot for Silk cached-decode execution. Phase timings are wall-clock
/// milliseconds measured with <see cref="System.Diagnostics.Stopwatch"/> and
/// <see cref="SilkRuntime.Finish"/> boundaries, so they include device execution and queue
/// synchronization overhead for the profiled phase. Values are accumulated across profiled tokens
/// until <see cref="SilkSmolLMGpuRunner.ResetDecodeProfile"/> /
/// <see cref="SilkQwenGpuRunner.ResetDecodeProfile"/> is called.
/// </summary>
public readonly record struct SilkDecodeProfile(
    int TokenCount,
    int KernelDispatches,
    int QueueCopies,
    double EmbedMs,
    double QkvMs,
    double RotaryMs,
    double CacheCopyMs,
    double AttentionOProjMs,
    double MlpMs,
    double FinalNormLmHeadMs,
    double ArgmaxMs)
{
    /// <summary>Total profiled phase milliseconds.</summary>
    public double TotalMs => EmbedMs + QkvMs + RotaryMs + CacheCopyMs + AttentionOProjMs + MlpMs + FinalNormLmHeadMs + ArgmaxMs;

    /// <summary>Average profiled milliseconds per token, or 0 when no tokens were profiled.</summary>
    public double MsPerToken => TokenCount > 0 ? TotalMs / TokenCount : 0d;

    /// <summary>Average kernel dispatches per token, or 0 when no tokens were profiled.</summary>
    public double DispatchesPerToken => TokenCount > 0 ? (double)KernelDispatches / TokenCount : 0d;
}
