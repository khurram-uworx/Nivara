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

    public static (float[] Means, bool[] Valid) RollingMean(
        SilkRuntime runtime,
        ReadOnlySpan<float> values,
        int windowSize,
        int minPeriods = 0)
    {
        if (runtime is null)
            throw new ArgumentNullException(nameof(runtime));
        if (windowSize < 1)
            throw new ArgumentOutOfRangeException(nameof(windowSize));
        if (values.Length == 0)
            return (Array.Empty<float>(), Array.Empty<bool>());

        if (minPeriods <= 0)
            minPeriods = windowSize;

        int n = values.Length;
        nint inBuf = runtime.Allocate(n);
        nint outBuf = runtime.Allocate(n);
        nint validBuf = runtime.Allocate(n);

        try
        {
            runtime.Upload(inBuf, values);

            var zeros = new int[n];
            runtime.Upload(validBuf, zeros.AsSpan());

            var ones = new float[n];
            Array.Fill(ones, 999f);
            runtime.Upload(outBuf, ones.AsSpan());

            runtime.SetBuffer(SilkKernels.RollingMean, 0, inBuf);
            runtime.SetBuffer(SilkKernels.RollingMean, 1, outBuf);
            runtime.SetBuffer(SilkKernels.RollingMean, 2, validBuf);
            runtime.SetUInt32(SilkKernels.RollingMean, 3, (uint)n);
            runtime.SetUInt32(SilkKernels.RollingMean, 4, (uint)windowSize);
            runtime.SetUInt32(SilkKernels.RollingMean, 5, (uint)minPeriods);

            runtime.Run(SilkKernels.RollingMean, new nuint[] { 64 }, new nuint[] { 1 });
            runtime.Finish();

            float[] means = runtime.Download(outBuf, n);
            int[] validInts = runtime.DownloadInt(validBuf, n);
            var valid = new bool[n];
            for (int i = 0; i < n; i++)
                valid[i] = validInts[i] != 0;

            return (means, valid);
        }
        finally
        {
            runtime.Release(inBuf);
            runtime.Release(outBuf);
            runtime.Release(validBuf);
        }
    }

    public static (float[] Values, bool[] Valid) RollingMin(
        SilkRuntime runtime,
        ReadOnlySpan<float> values,
        int windowSize,
        int minPeriods = 0)
    {
        if (runtime is null)
            throw new ArgumentNullException(nameof(runtime));
        if (windowSize < 1)
            throw new ArgumentOutOfRangeException(nameof(windowSize));
        if (values.Length == 0)
            return (Array.Empty<float>(), Array.Empty<bool>());

        if (minPeriods <= 0)
            minPeriods = windowSize;

        int n = values.Length;
        nint inBuf = runtime.Allocate(n);
        nint outBuf = runtime.Allocate(n);
        nint validBuf = runtime.Allocate(n);

        try
        {
            runtime.Upload(inBuf, values);

            var zeros = new int[n];
            runtime.Upload(validBuf, zeros.AsSpan());

            runtime.SetBuffer(SilkKernels.RollingMin, 0, inBuf);
            runtime.SetBuffer(SilkKernels.RollingMin, 1, outBuf);
            runtime.SetBuffer(SilkKernels.RollingMin, 2, validBuf);
            runtime.SetUInt32(SilkKernels.RollingMin, 3, (uint)n);
            runtime.SetUInt32(SilkKernels.RollingMin, 4, (uint)windowSize);
            runtime.SetUInt32(SilkKernels.RollingMin, 5, (uint)minPeriods);

            runtime.Run(SilkKernels.RollingMin, new nuint[] { 64 }, new nuint[] { 1 });
            runtime.Finish();

            float[] valuesOut = runtime.Download(outBuf, n);
            int[] validInts = runtime.DownloadInt(validBuf, n);
            var valid = new bool[n];
            for (int i = 0; i < n; i++)
                valid[i] = validInts[i] != 0;

            return (valuesOut, valid);
        }
        finally
        {
            runtime.Release(inBuf);
            runtime.Release(outBuf);
            runtime.Release(validBuf);
        }
    }

    public static (float[] Values, bool[] Valid) RollingMax(
        SilkRuntime runtime,
        ReadOnlySpan<float> values,
        int windowSize,
        int minPeriods = 0)
    {
        if (runtime is null)
            throw new ArgumentNullException(nameof(runtime));
        if (windowSize < 1)
            throw new ArgumentOutOfRangeException(nameof(windowSize));
        if (values.Length == 0)
            return (Array.Empty<float>(), Array.Empty<bool>());

        if (minPeriods <= 0)
            minPeriods = windowSize;

        int n = values.Length;
        nint inBuf = runtime.Allocate(n);
        nint outBuf = runtime.Allocate(n);
        nint validBuf = runtime.Allocate(n);

        try
        {
            runtime.Upload(inBuf, values);

            var zeros = new int[n];
            runtime.Upload(validBuf, zeros.AsSpan());

            runtime.SetBuffer(SilkKernels.RollingMax, 0, inBuf);
            runtime.SetBuffer(SilkKernels.RollingMax, 1, outBuf);
            runtime.SetBuffer(SilkKernels.RollingMax, 2, validBuf);
            runtime.SetUInt32(SilkKernels.RollingMax, 3, (uint)n);
            runtime.SetUInt32(SilkKernels.RollingMax, 4, (uint)windowSize);
            runtime.SetUInt32(SilkKernels.RollingMax, 5, (uint)minPeriods);

            runtime.Run(SilkKernels.RollingMax, new nuint[] { 64 }, new nuint[] { 1 });
            runtime.Finish();

            float[] valuesOut = runtime.Download(outBuf, n);
            int[] validInts = runtime.DownloadInt(validBuf, n);
            var valid = new bool[n];
            for (int i = 0; i < n; i++)
                valid[i] = validInts[i] != 0;

            return (valuesOut, valid);
        }
        finally
        {
            runtime.Release(inBuf);
            runtime.Release(outBuf);
            runtime.Release(validBuf);
        }
    }

    public static (int[] Counts, int[] Errors, float[] Sums, float[] Sumsq, float[] Mins, float[] Maxs) GroupedMultiStatReduce(
        SilkRuntime runtime,
        int[] serviceIds,
        int[] statusCodes,
        float[] durations,
        int groupCount)
    {
        const int ThreadsPerGroup = 256;
        const int MaxGroups = 64;
        if (serviceIds is null) throw new ArgumentNullException(nameof(serviceIds));
        if (statusCodes is null) throw new ArgumentNullException(nameof(statusCodes));
        if (durations is null) throw new ArgumentNullException(nameof(durations));
        if (serviceIds.Length != statusCodes.Length || serviceIds.Length != durations.Length)
            throw new ArgumentException("input arrays must have the same length");
        if (groupCount <= 0 || groupCount > MaxGroups)
            throw new ArgumentOutOfRangeException(nameof(groupCount));

        int n = serviceIds.Length;
        int workGroups = Math.Min(1024, Math.Max(1, (n + ThreadsPerGroup - 1) / ThreadsPerGroup));
        nint inService = runtime.AllocateBytes((ulong)n * sizeof(int));
        nint inStatus = runtime.AllocateBytes((ulong)n * sizeof(int));
        nint inDuration = runtime.Allocate(n);
        nint partialCounts = runtime.AllocateBytes((ulong)workGroups * (ulong)groupCount * sizeof(int));
        nint partialErrors = runtime.AllocateBytes((ulong)workGroups * (ulong)groupCount * sizeof(int));
        nint partialSums = runtime.Allocate(workGroups * groupCount);
        nint partialSumsq = runtime.Allocate(workGroups * groupCount);
        nint partialMins = runtime.Allocate(workGroups * groupCount);
        nint partialMaxs = runtime.Allocate(workGroups * groupCount);

        try
        {
            runtime.Upload(inService, serviceIds.AsSpan());
            runtime.Upload(inStatus, statusCodes.AsSpan());
            runtime.Upload(inDuration, durations.AsSpan());

            var zeroCounts = new int[workGroups * groupCount];
            var zeroErrors = new int[workGroups * groupCount];
            var zeroSums = new float[workGroups * groupCount];
            var zeroSumsq = new float[workGroups * groupCount];
            var zeroMins = new float[workGroups * groupCount];
            var zeroMaxs = new float[workGroups * groupCount];
            const float floatMax = 3.40282346638528860e+38f;
            const float floatMin = -3.40282346638528860e+38f;
            for (int i = 0; i < zeroMins.Length; i++) { zeroMins[i] = floatMax; zeroMaxs[i] = floatMin; }

            runtime.Upload(partialCounts, zeroCounts.AsSpan());
            runtime.Upload(partialErrors, zeroErrors.AsSpan());
            runtime.Upload(partialSums, zeroSums.AsSpan());
            runtime.Upload(partialSumsq, zeroSumsq.AsSpan());
            runtime.Upload(partialMins, zeroMins.AsSpan());
            runtime.Upload(partialMaxs, zeroMaxs.AsSpan());

            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 0, inService);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 1, inStatus);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 2, inDuration);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 3, partialCounts);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 4, partialErrors);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 5, partialSums);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 6, partialSumsq);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 7, partialMins);
            runtime.SetBuffer(SilkKernels.GroupedMultiStatReducePartial, 8, partialMaxs);
            runtime.SetUInt32(SilkKernels.GroupedMultiStatReducePartial, 9, (uint)n);
            runtime.SetUInt32(SilkKernels.GroupedMultiStatReducePartial, 10, (uint)groupCount);

            runtime.Run(SilkKernels.GroupedMultiStatReducePartial,
                new nuint[] { (nuint)workGroups },
                new nuint[] { 1 });
            runtime.Finish();

            int[] pc = runtime.DownloadInt(partialCounts, workGroups * groupCount);
            int[] pe = runtime.DownloadInt(partialErrors, workGroups * groupCount);
            float[] ps = runtime.Download(partialSums, workGroups * groupCount);
            float[] pss = runtime.Download(partialSumsq, workGroups * groupCount);
            float[] pmins = runtime.Download(partialMins, workGroups * groupCount);
            float[] pmaxs = runtime.Download(partialMaxs, workGroups * groupCount);

            var counts = new int[groupCount];
            var errors = new int[groupCount];
            var sums = new float[groupCount];
            var sumsq = new float[groupCount];
            var mins = new float[groupCount];
            var maxs = new float[groupCount];
            for (int g = 0; g < groupCount; g++)
            {
                mins[g] = floatMax;
                maxs[g] = floatMin;
            }
            for (int w = 0; w < workGroups; w++)
            {
                int baseIdx = w * groupCount;
                for (int g = 0; g < groupCount; g++)
                {
                    int c = pc[baseIdx + g];
                    counts[g] += c;
                    errors[g] += pe[baseIdx + g];
                    sums[g] += ps[baseIdx + g];
                    sumsq[g] += pss[baseIdx + g];
                    if (c > 0)
                    {
                        if (pmins[baseIdx + g] < mins[g]) mins[g] = pmins[baseIdx + g];
                        if (pmaxs[baseIdx + g] > maxs[g]) maxs[g] = pmaxs[baseIdx + g];
                    }
                }
            }
            for (int g = 0; g < groupCount; g++)
            {
                if (counts[g] == 0)
                {
                    mins[g] = 0;
                    maxs[g] = 0;
                }
            }

            return (counts, errors, sums, sumsq, mins, maxs);
        }
        finally
        {
            runtime.Release(inService);
            runtime.Release(inStatus);
            runtime.Release(inDuration);
            runtime.Release(partialCounts);
            runtime.Release(partialErrors);
            runtime.Release(partialSums);
            runtime.Release(partialSumsq);
            runtime.Release(partialMins);
            runtime.Release(partialMaxs);
        }
    }
}
