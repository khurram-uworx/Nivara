using System.Diagnostics;

namespace Nivara.Samples.Incident;

/// <summary>
/// Profitability gate for the grouped hash-reduction GPU offload. The decision is made
/// before paying for the GPU transfer, using a CPU sample to estimate the CPU cost and a
/// simple cost model for the GPU cost.
/// </summary>
public static class GpuProfitability
{
    /// <summary>GPU must be at most this fraction of the estimated CPU time to win.</summary>
    public const double Margin = 0.9;

    /// <summary>Rows used to calibrate the CPU cost.</summary>
    public const int SampleSize = 100_000;

    /// <summary>Fixed launch + kernel overhead in milliseconds.</summary>
    public const double EstimatedGpuFixedMs = 1.0;

    /// <summary>Assumed transfer bandwidth in bytes per millisecond (~5 GB/s).</summary>
    public const double EstimatedGpuBytesPerMs = 5_000_000;

    /// <summary>Estimated GPU cost per row in milliseconds (calibrate from a measured run).</summary>
    public const double EstimatedGpuMsPerRow = 0.0003;

    public static double CalibrateCpuMs(
        int[] serviceIds, int[] statusCodes, float[] durations, int groupCount, int sampleSize)
    {
        if (serviceIds is null) throw new ArgumentNullException(nameof(serviceIds));
        if (statusCodes is null) throw new ArgumentNullException(nameof(statusCodes));
        if (durations is null) throw new ArgumentNullException(nameof(durations));
        if (serviceIds.Length != statusCodes.Length || serviceIds.Length != durations.Length)
            throw new ArgumentException("input arrays must have the same length");

        int n = Math.Min(sampleSize, serviceIds.Length);
        if (n <= 0 || groupCount <= 0) return 0;

        // The accumulators are only here to give the loop observable work to do; the sample's
        // result is discarded because only its duration feeds the estimate.
        var counts = new int[groupCount];
        var errors = new int[groupCount];
        var sums = new double[groupCount];

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < n; i++)
        {
            int g = serviceIds[i];
            if (g < 0 || g >= groupCount) continue;
            counts[g]++;
            if (statusCodes[i] >= 500) errors[g]++;
            sums[g] += durations[i];
        }
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    public static double EstimateCpuMs(double sampleMs, int sampleSize, int rowCount)
    {
        if (sampleSize <= 0 || rowCount <= 0) return 0;
        return sampleMs * (double)rowCount / sampleSize;
    }

    public static double EstimateGpuMs(int rowCount, long transferBytes)
    {
        if (rowCount < 0) rowCount = 0;
        if (transferBytes < 0) transferBytes = 0;
        return EstimatedGpuFixedMs + (double)transferBytes / EstimatedGpuBytesPerMs
               + (double)rowCount * EstimatedGpuMsPerRow;
    }

    public static (bool Offload, string Reason, double EstimatedCpuMs, double EstimatedGpuMs) Decide(
        bool useGpu, bool gpuForce, double estimatedCpuMs, double estimatedGpuMs)
    {
        if (!useGpu && !gpuForce)
            return (false, "gpu not requested", estimatedCpuMs, estimatedGpuMs);
        if (gpuForce)
            return (true, "forced (--gpu-force)", estimatedCpuMs, estimatedGpuMs);
        if (estimatedGpuMs < estimatedCpuMs * Margin)
            return (true, $"profitable (est. gpu {estimatedGpuMs:F1} ms < cpu {estimatedCpuMs:F1} ms)", estimatedCpuMs, estimatedGpuMs);
        return (false, $"not profitable (est. gpu {estimatedGpuMs:F1} ms >= cpu {estimatedCpuMs:F1} ms)", estimatedCpuMs, estimatedGpuMs);
    }
}
