using System;

namespace Nivara.Silk;

/// <summary>
/// Host-side driver for the analytics kernels (OpenCL). Pilot: grouped hash-reduction
/// for the NivaraIncident grouped aggregation analysis.
/// </summary>
public static class SilkAnalytics
{
    const int ThreadsPerGroup = 256;
    const int MaxGroups = 64;

    /// <summary>
    /// Reduces <paramref name="serviceIds"/> / <paramref name="statusCodes"/> /
    /// <paramref name="durations"/> into per-group counts, error counts, and duration sums.
    /// </summary>
    public static (int[] Counts, int[] Errors, float[] Sums) GroupedHashReduce(
        SilkRuntime runtime,
        int[] serviceIds,
        int[] statusCodes,
        float[] durations,
        int groupCount)
    {
        if (serviceIds is null) throw new ArgumentNullException(nameof(serviceIds));
        if (statusCodes is null) throw new ArgumentNullException(nameof(statusCodes));
        if (durations is null) throw new ArgumentNullException(nameof(durations));
        if (serviceIds.Length != statusCodes.Length || serviceIds.Length != durations.Length)
            throw new ArgumentException("input arrays must have the same length");
        if (groupCount <= 0 || groupCount > MaxGroups)
            throw new ArgumentOutOfRangeException(nameof(groupCount));

        int n = serviceIds.Length;
        int workGroups = Math.Min(1024, Math.Max(1, (n + ThreadsPerGroup - 1) / ThreadsPerGroup));
        int globalSize = workGroups;

        nint inService = runtime.AllocateBytes((ulong)n * sizeof(int));
        nint inStatus = runtime.AllocateBytes((ulong)n * sizeof(int));
        nint inDuration = runtime.Allocate(n); // float count

        nint partialCounts = runtime.AllocateBytes((ulong)workGroups * (ulong)groupCount * sizeof(int));
        nint partialErrors = runtime.AllocateBytes((ulong)workGroups * (ulong)groupCount * sizeof(int));
        nint partialSums = runtime.Allocate(workGroups * groupCount);

        try
        {
            runtime.Upload(inService, serviceIds.AsSpan());
            runtime.Upload(inStatus, statusCodes.AsSpan());
            runtime.Upload(inDuration, durations.AsSpan());

            // zero partials
            var zeroCounts = new int[workGroups * groupCount];
            var zeroErrors = new int[workGroups * groupCount];
            var zeroSums = new float[workGroups * groupCount];
            runtime.Upload(partialCounts, zeroCounts.AsSpan());
            runtime.Upload(partialErrors, zeroErrors.AsSpan());
            runtime.Upload(partialSums, zeroSums.AsSpan());

            runtime.SetBuffer(SilkKernels.GroupedHashReducePartial, 0, inService);
            runtime.SetBuffer(SilkKernels.GroupedHashReducePartial, 1, inStatus);
            runtime.SetBuffer(SilkKernels.GroupedHashReducePartial, 2, inDuration);
            runtime.SetBuffer(SilkKernels.GroupedHashReducePartial, 3, partialCounts);
            runtime.SetBuffer(SilkKernels.GroupedHashReducePartial, 4, partialErrors);
            runtime.SetBuffer(SilkKernels.GroupedHashReducePartial, 5, partialSums);
            runtime.SetUInt32(SilkKernels.GroupedHashReducePartial, 6, (uint)n);
            runtime.SetUInt32(SilkKernels.GroupedHashReducePartial, 7, (uint)groupCount);

            runtime.Run(SilkKernels.GroupedHashReducePartial,
                new nuint[] { (nuint)globalSize },
                new nuint[] { 1 });
            runtime.Finish();

            int[] pc = runtime.DownloadInt(partialCounts, workGroups * groupCount);
            int[] pe = runtime.DownloadInt(partialErrors, workGroups * groupCount);
            float[] ps = runtime.Download(partialSums, workGroups * groupCount);

            var counts = new int[groupCount];
            var errors = new int[groupCount];
            var sums = new float[groupCount];
            for (int w = 0; w < workGroups; w++)
            {
                int baseIdx = w * groupCount;
                for (int g = 0; g < groupCount; g++)
                {
                    counts[g] += pc[baseIdx + g];
                    errors[g] += pe[baseIdx + g];
                    sums[g] += ps[baseIdx + g];
                }
            }

            return (counts, errors, sums);
        }
        finally
        {
            runtime.Release(inService);
            runtime.Release(inStatus);
            runtime.Release(inDuration);
            runtime.Release(partialCounts);
            runtime.Release(partialErrors);
            runtime.Release(partialSums);
        }
    }
}
