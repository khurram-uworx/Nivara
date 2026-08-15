using System.Buffers;
using System.Numerics;

namespace Nivara.Tensors;

/// <summary>
/// Window-function extensions on <see cref="NivaraColumn{T}"/>: rolling aggregates,
/// cumulative aggregates, and shift/lead.
/// <para>
/// Nulls follow the project's explicit null-mask model. Rolling and cumulative aggregates
/// ignore nulls by default and gate output on a minimum valid count; an optional
/// <paramref name="nullHandler"/> replaces each null so it participates in the computation.
/// </para>
/// </summary>
public static class WindowFunctions
{
    // ── Cumulative ──

    /// <summary>
    /// Computes the cumulative sum over non-null elements. Null positions stay null and the
    /// accumulated value carries forward. With <paramref name="nullHandler"/> set, nulls are
    /// replaced and the output has no nulls.
    /// </summary>
    public static NivaraColumn<T> CumulativeSum<T>(this NivaraColumn<T> column, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        return cumulativeScan(column, nullHandler, isSum: true, isMax: false, isProduct: false);
    }

    /// <summary>
    /// Computes the cumulative maximum over non-null elements. Null positions stay null and the
    /// accumulated value carries forward. With <paramref name="nullHandler"/> set, nulls are
    /// replaced and the output has no nulls.
    /// </summary>
    public static NivaraColumn<T> CumulativeMax<T>(this NivaraColumn<T> column, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        return cumulativeScan(column, nullHandler, isSum: false, isMax: true, isProduct: false);
    }

    /// <summary>
    /// Computes the cumulative minimum over non-null elements. Null positions stay null and the
    /// accumulated value carries forward. With <paramref name="nullHandler"/> set, nulls are
    /// replaced and the output has no nulls.
    /// </summary>
    public static NivaraColumn<T> CumulativeMin<T>(this NivaraColumn<T> column, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        return cumulativeScan(column, nullHandler, isSum: false, isMax: false, isProduct: false);
    }

    /// <summary>
    /// Computes the cumulative product over non-null elements. Null positions stay null and the
    /// accumulated value carries forward. With <paramref name="nullHandler"/> set, nulls are
    /// replaced and the output has no nulls.
    /// </summary>
    public static NivaraColumn<T> CumulativeProduct<T>(this NivaraColumn<T> column, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        return cumulativeScan(column, nullHandler, isSum: false, isMax: false, isProduct: true);
    }

    /// <summary>
    /// Computes the running count of non-null elements. Null positions stay null.
    /// </summary>
    public static NivaraColumn<long> CumulativeCount<T>(this NivaraColumn<T> column)
    {
        ArgumentNullException.ThrowIfNull(column);
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<long>.Create(Array.Empty<long>());

        var result = new long[length];
        var resultMask = new bool[length];
        long count = 0;

        for (int i = 0; i < length; i++)
        {
            if (column.IsNull(i))
            {
                resultMask[i] = true;
            }
            else
            {
                count++;
                result[i] = count;
            }
        }

        return NivaraColumn<long>.CreateFromSpans(result, resultMask);
    }

    // ── Rolling ──

    /// <summary>
    /// Computes the rolling sum over a fixed trailing window. Nulls inside the window are
    /// ignored; the output is null until the window holds at least <paramref name="minPeriods"/>
    /// valid values (default: the full window). With <paramref name="nullHandler"/> set, nulls are
    /// replaced and every position satisfies the window.
    /// </summary>
    public static NivaraColumn<T> RollingSum<T>(this NivaraColumn<T> column, int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        var min = resolveWindowArgs(windowSize, minPeriods, relaxWhenHandler: nullHandler != null);
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<T>.Create(Array.Empty<T>());

        var result = new T[length];
        var resultMask = new bool[length];

        if (!column.HasNulls && column.TryGetSpan(out var span))
            return isIntFamily<T>()
                ? rollingSumFromSpanInt(span, windowSize, min, result, resultMask)
                : rollingSumFromSpan(span, windowSize, min, result, resultMask);

        var (effective, valid) = buildEffective(column, nullHandler);

        if (isIntFamily<T>())
        {
            var (prefixSum, prefixCount) = buildWidenedPrefix(effective, valid);
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                long windowSum = prefixSum[i] - (lo >= 1 ? prefixSum[lo - 1] : 0L);
                int windowCount = prefixCount[i] - (lo >= 1 ? prefixCount[lo - 1] : 0);

                if (windowCount >= min)
                    result[i] = T.CreateChecked(windowSum);
                else
                {
                    resultMask[i] = true;
                }
            }
        }
        else
        {
            var (prefixSum, prefixCount) = buildPrefix(effective, valid);
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                T windowSum = prefixSum[i] - (lo >= 1 ? prefixSum[lo - 1] : T.Zero);
                int windowCount = prefixCount[i] - (lo >= 1 ? prefixCount[lo - 1] : 0);

                if (windowCount >= min)
                    result[i] = windowSum;
                else
                {
                    resultMask[i] = true;
                }
            }
        }

        return NivaraColumn<T>.CreateFromSpans(result, resultMask);
    }

    /// <summary>
    /// Computes the rolling mean over a fixed trailing window. Nulls inside the window are
    /// ignored; the output is null until the window holds at least <paramref name="minPeriods"/>
    /// valid values (default: the full window). With <paramref name="nullHandler"/> set, nulls are
    /// replaced and every position satisfies the window. Returns a double column.
    /// </summary>
    public static NivaraColumn<double> RollingMean<T>(this NivaraColumn<T> column, int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        var min = resolveWindowArgs(windowSize, minPeriods, relaxWhenHandler: nullHandler != null);
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<double>.Create(Array.Empty<double>());

        var result = new double[length];
        var resultMask = new bool[length];

        if (!column.HasNulls && column.TryGetSpan(out var span))
            return isIntFamily<T>()
                ? rollingMeanFromSpanInt(span, windowSize, min, result, resultMask)
                : rollingMeanFromSpan(span, windowSize, min, result, resultMask);

        var (effective, valid) = buildEffective(column, nullHandler);

        if (isIntFamily<T>())
        {
            var (prefixSum, prefixCount) = buildWidenedPrefix(effective, valid);
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                long windowSum = prefixSum[i] - (lo >= 1 ? prefixSum[lo - 1] : 0L);
                int windowCount = prefixCount[i] - (lo >= 1 ? prefixCount[lo - 1] : 0);

                if (windowCount >= min)
                    result[i] = double.CreateChecked(windowSum) / windowCount;
                else
                {
                    resultMask[i] = true;
                }
            }
        }
        else
        {
            var (prefixSum, prefixCount) = buildPrefix(effective, valid);
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                T windowSum = prefixSum[i] - (lo >= 1 ? prefixSum[lo - 1] : T.Zero);
                int windowCount = prefixCount[i] - (lo >= 1 ? prefixCount[lo - 1] : 0);

                if (windowCount >= min)
                    result[i] = double.CreateChecked(windowSum) / windowCount;
                else
                {
                    resultMask[i] = true;
                }
            }
        }

        return NivaraColumn<double>.CreateFromSpans(result, resultMask);
    }

    /// <summary>
    /// Computes the rolling maximum over a fixed trailing window. Nulls inside the window are
    /// ignored; the output is null until the window holds at least <paramref name="minPeriods"/>
    /// valid values (default: the full window). With <paramref name="nullHandler"/> set, nulls are
    /// replaced and every position satisfies the window.
    /// </summary>
    public static NivaraColumn<T> RollingMax<T>(this NivaraColumn<T> column, int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        var min = resolveWindowArgs(windowSize, minPeriods, relaxWhenHandler: nullHandler != null);
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<T>.Create(Array.Empty<T>());

        if (!column.HasNulls && column.TryGetSpan(out var span))
            return rollingExtremeFromSpan(span, windowSize, min, takeMax: true);

        var (effective, valid) = buildEffective(column, nullHandler);

        return rollingExtreme(effective, valid, windowSize, min, takeMax: true);
    }

    /// <summary>
    /// Computes the rolling minimum over a fixed trailing window. Nulls inside the window are
    /// ignored; the output is null until the window holds at least <paramref name="minPeriods"/>
    /// valid values (default: the full window). With <paramref name="nullHandler"/> set, nulls are
    /// replaced and every position satisfies the window.
    /// </summary>
    public static NivaraColumn<T> RollingMin<T>(this NivaraColumn<T> column, int windowSize, int? minPeriods = null, Func<T>? nullHandler = null)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        var min = resolveWindowArgs(windowSize, minPeriods, relaxWhenHandler: nullHandler != null);
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<T>.Create(Array.Empty<T>());

        if (!column.HasNulls && column.TryGetSpan(out var span))
            return rollingExtremeFromSpan(span, windowSize, min, takeMax: false);

        var (effective, valid) = buildEffective(column, nullHandler);

        return rollingExtreme(effective, valid, windowSize, min, takeMax: false);
    }

    // ── Shift / Lead ──

    /// <summary>
    /// Shifts values forward by <paramref name="periods"/> positions (positive = lag): output at
    /// <c>i</c> is the input at <c>i - periods</c>. Positions moved in from outside the column are
    /// null. In-range nulls are preserved.
    /// </summary>
    public static NivaraColumn<T> Shift<T>(this NivaraColumn<T> column, int periods)
    {
        ArgumentNullException.ThrowIfNull(column);
        return shiftCore(column, periods, default(T)!, hasFill: false);
    }

    /// <summary>
    /// Shifts values forward by <paramref name="periods"/> positions (positive = lag): output at
    /// <c>i</c> is the input at <c>i - periods</c>. Positions moved in from outside the column are
    /// set to <paramref name="fillValue"/>. In-range nulls are preserved.
    /// </summary>
    public static NivaraColumn<T> Shift<T>(this NivaraColumn<T> column, int periods, T fillValue)
    {
        ArgumentNullException.ThrowIfNull(column);
        return shiftCore(column, periods, fillValue, hasFill: true);
    }

    /// <summary>
    /// Shifts values backward by <paramref name="periods"/> positions (negative = lead): output at
    /// <c>i</c> is the input at <c>i + periods</c>. Positions moved in from outside the column are
    /// null. In-range nulls are preserved.
    /// </summary>
    public static NivaraColumn<T> Lead<T>(this NivaraColumn<T> column, int periods)
    {
        ArgumentNullException.ThrowIfNull(column);
        return shiftCore(column, -periods, default(T)!, hasFill: false);
    }

    /// <summary>
    /// Shifts values backward by <paramref name="periods"/> positions (negative = lead): output at
    /// <c>i</c> is the input at <c>i + periods</c>. Positions moved in from outside the column are
    /// set to <paramref name="fillValue"/>. In-range nulls are preserved.
    /// </summary>
    public static NivaraColumn<T> Lead<T>(this NivaraColumn<T> column, int periods, T fillValue)
    {
        ArgumentNullException.ThrowIfNull(column);
        return shiftCore(column, -periods, fillValue, hasFill: true);
    }

    // ── Shared kernels ──

    static bool isIntFamily<T>()
        where T : struct, INumber<T>
        => typeof(T) is var t
            && (t == typeof(sbyte) || t == typeof(byte) || t == typeof(short) || t == typeof(ushort)
                || t == typeof(int) || t == typeof(uint) || t == typeof(char));

    static int resolveWindowArgs(int windowSize, int? minPeriods, bool relaxWhenHandler = false)
    {
        if (windowSize < 1)
            throw new ArgumentOutOfRangeException(nameof(windowSize), "Window size must be at least 1");

        if (minPeriods is int min)
        {
            if (min < 1 || min > windowSize)
                throw new ArgumentOutOfRangeException(nameof(minPeriods), "minPeriods must be in [1, windowSize]");

            return min;
        }

        return relaxWhenHandler ? 1 : windowSize;
    }

    static (T[] Effective, bool[] Valid) buildEffective<T>(NivaraColumn<T> column, Func<T>? nullHandler)
        where T : struct
    {
        var effective = new T[column.Length];
        var valid = new bool[column.Length];

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                span.CopyTo(effective);
            else
                for (int i = 0; i < column.Length; i++)
                    effective[i] = column[i];

            Array.Fill(valid, true);
            return (effective, valid);
        }

        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
            {
                effective[i] = column[i];
                valid[i] = true;
            }
            else if (nullHandler != null)
            {
                effective[i] = nullHandler();
                valid[i] = true;
            }
        }

        return (effective, valid);
    }

    static NivaraColumn<T> cumulativeScan<T>(NivaraColumn<T> column, Func<T>? nullHandler, bool isSum, bool isMax, bool isProduct)
        where T : struct, INumber<T>
    {
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<T>.Create(Array.Empty<T>());

        var result = new T[length];
        var resultMask = new bool[length];

        if (!column.HasNulls && column.TryGetSpan(out var span))
        {
            if (isIntFamily<T>() && (isSum || isProduct))
            {
                long accumulator = long.CreateChecked(span[0]);
                result[0] = T.CreateChecked(accumulator);
                for (int i = 1; i < length; i++)
                {
                    // Checked: a product of several large int-family values can overflow the widened
                    // long accumulator itself (e.g. [int.MaxValue] * 3). Without checked, the wrap
                    // could land inside the result type's range and be silently returned instead of
                    // throwing (issue #248).
                    checked
                    {
                        accumulator = isSum
                            ? accumulator + long.CreateChecked(span[i])
                            : accumulator * long.CreateChecked(span[i]);
                    }

                    result[i] = T.CreateChecked(accumulator);
                }
            }
            else
            {
                T accumulator = span[0];
                result[0] = accumulator;
                for (int i = 1; i < length; i++)
                    result[i] = accumulator = isSum ? accumulator + span[i]
                        : isProduct ? accumulator * span[i]
                        : isMax ? T.Max(accumulator, span[i])
                        : T.Min(accumulator, span[i]);
            }

            return NivaraColumn<T>.CreateFromSpans(result, resultMask);
        }

        var (effective, valid) = buildEffective(column, nullHandler);

        if (isIntFamily<T>() && (isSum || isProduct))
        {
            bool hasAccumulator = false;
            long accumulator = 0;

            for (int i = 0; i < length; i++)
            {
                if (!valid[i])
                {
                    resultMask[i] = true;
                    continue;
                }

                if (!hasAccumulator)
                {
                    accumulator = long.CreateChecked(effective[i]);
                    hasAccumulator = true;
                }
                else
                {
                    // Checked: a product of several large int-family values can overflow the widened
                    // long accumulator itself (e.g. [int.MaxValue] * 3). Without checked, the wrap
                    // could land inside the result type's range and be silently returned instead of
                    // throwing (issue #248).
                    checked
                    {
                        accumulator = isSum
                            ? accumulator + long.CreateChecked(effective[i])
                            : accumulator * long.CreateChecked(effective[i]);
                    }
                }

                result[i] = T.CreateChecked(accumulator);
            }

            return NivaraColumn<T>.CreateFromSpans(result, resultMask);
        }

        bool hasTAccumulator = false;
        T tAccumulator = default;

        for (int i = 0; i < length; i++)
        {
            if (!valid[i])
            {
                resultMask[i] = true;
                continue;
            }

            if (!hasTAccumulator)
            {
                tAccumulator = effective[i];
                hasTAccumulator = true;
            }
            else
            {
                tAccumulator = isSum ? tAccumulator + effective[i]
                    : isProduct ? tAccumulator * effective[i]
                    : isMax ? T.Max(tAccumulator, effective[i])
                    : T.Min(tAccumulator, effective[i]);
            }

            result[i] = tAccumulator;
        }

        return NivaraColumn<T>.CreateFromSpans(result, resultMask);
    }

    static (long[] PrefixSum, int[] PrefixCount) buildWidenedPrefix<T>(T[] effective, bool[] valid)
        where T : struct, INumber<T>
    {
        var length = effective.Length;
        var prefixSum = new long[length];
        var prefixCount = new int[length];
        long runningSum = 0;
        int runningCount = 0;

        for (int i = 0; i < length; i++)
        {
            if (valid[i])
            {
                checked { runningSum += long.CreateChecked(effective[i]); }
                runningCount++;
            }

            prefixSum[i] = runningSum;
            prefixCount[i] = runningCount;
        }

        return (prefixSum, prefixCount);
    }

    static (T[] PrefixSum, int[] PrefixCount) buildPrefix<T>(T[] effective, bool[] valid)
        where T : struct, INumber<T>
    {
        var length = effective.Length;
        var prefixSum = new T[length];
        var prefixCount = new int[length];
        T runningSum = T.Zero;
        int runningCount = 0;

        for (int i = 0; i < length; i++)
        {
            if (valid[i])
            {
                runningSum += effective[i];
                runningCount++;
            }

            prefixSum[i] = runningSum;
            prefixCount[i] = runningCount;
        }

        return (prefixSum, prefixCount);
    }

    static (T[] PrefixSum, bool Rented) buildPrefixFromSpan<T>(ReadOnlySpan<T> span)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var rented = length > 1024;
        var prefix = rented ? ArrayPool<T>.Shared.Rent(length) : new T[length];
        T running = T.Zero;

        for (int i = 0; i < length; i++)
        {
            running += span[i];
            prefix[i] = running;
        }

        return (prefix, rented);
    }

    static (long[] PrefixSum, bool Rented) buildWidenedPrefixFromSpan<T>(ReadOnlySpan<T> span)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var rented = length > 1024;
        var prefix = rented ? ArrayPool<long>.Shared.Rent(length) : new long[length];
        long running = 0;

        for (int i = 0; i < length; i++)
        {
            checked { running += long.CreateChecked(span[i]); }
            prefix[i] = running;
        }

        return (prefix, rented);
    }

    static NivaraColumn<T> rollingSumFromSpan<T>(ReadOnlySpan<T> span, int windowSize, int min, T[] result, bool[] resultMask)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var (prefix, rented) = buildPrefixFromSpan(span);

        try
        {
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                T windowSum = prefix[i] - (lo >= 1 ? prefix[lo - 1] : T.Zero);
                int windowCount = Math.Min(i + 1, windowSize);

                if (windowCount >= min)
                    result[i] = windowSum;
                else
                    resultMask[i] = true;
            }

            return NivaraColumn<T>.CreateFromSpans(result, resultMask);
        }
        finally
        {
            if (rented)
                ArrayPool<T>.Shared.Return(prefix);
        }
    }

    static NivaraColumn<T> rollingSumFromSpanInt<T>(ReadOnlySpan<T> span, int windowSize, int min, T[] result, bool[] resultMask)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var (prefix, rented) = buildWidenedPrefixFromSpan(span);

        try
        {
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                long windowSum = prefix[i] - (lo >= 1 ? prefix[lo - 1] : 0L);
                int windowCount = Math.Min(i + 1, windowSize);

                if (windowCount >= min)
                    result[i] = T.CreateChecked(windowSum);
                else
                    resultMask[i] = true;
            }

            return NivaraColumn<T>.CreateFromSpans(result, resultMask);
        }
        finally
        {
            if (rented)
                ArrayPool<long>.Shared.Return(prefix);
        }
    }

    static NivaraColumn<double> rollingMeanFromSpan<T>(ReadOnlySpan<T> span, int windowSize, int min, double[] result, bool[] resultMask)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var (prefix, rented) = buildPrefixFromSpan(span);

        try
        {
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                T windowSum = prefix[i] - (lo >= 1 ? prefix[lo - 1] : T.Zero);
                int windowCount = Math.Min(i + 1, windowSize);

                if (windowCount >= min)
                    result[i] = double.CreateChecked(windowSum) / windowCount;
                else
                    resultMask[i] = true;
            }

            return NivaraColumn<double>.CreateFromSpans(result, resultMask);
        }
        finally
        {
            if (rented)
                ArrayPool<T>.Shared.Return(prefix);
        }
    }

    static NivaraColumn<double> rollingMeanFromSpanInt<T>(ReadOnlySpan<T> span, int windowSize, int min, double[] result, bool[] resultMask)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var (prefix, rented) = buildWidenedPrefixFromSpan(span);

        try
        {
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;
                long windowSum = prefix[i] - (lo >= 1 ? prefix[lo - 1] : 0L);
                int windowCount = Math.Min(i + 1, windowSize);

                if (windowCount >= min)
                    result[i] = double.CreateChecked(windowSum) / windowCount;
                else
                    resultMask[i] = true;
            }

            return NivaraColumn<double>.CreateFromSpans(result, resultMask);
        }
        finally
        {
            if (rented)
                ArrayPool<long>.Shared.Return(prefix);
        }
    }

    static NivaraColumn<T> rollingExtremeFromSpan<T>(ReadOnlySpan<T> span, int windowSize, int minPeriods, bool takeMax)
        where T : struct, INumber<T>
    {
        var length = span.Length;
        var rented = length > 1024;
        var deque = rented ? ArrayPool<int>.Shared.Rent(length) : new int[length];
        int head = 0;
        int tail = 0;
        var result = new T[length];
        var resultMask = new bool[length];

        try
        {
            for (int i = 0; i < length; i++)
            {
                int lo = i - windowSize + 1;

                while (head < tail && deque[head] < lo)
                    head++;

                var current = span[i];
                while (head < tail && (takeMax ? span[deque[tail - 1]] <= current : span[deque[tail - 1]] >= current))
                    tail--;

                deque[tail++] = i;

                if (Math.Min(i + 1, windowSize) >= minPeriods)
                    result[i] = span[deque[head]];
                else
                    resultMask[i] = true;
            }

            return NivaraColumn<T>.CreateFromSpans(result, resultMask);
        }
        finally
        {
            if (rented)
                ArrayPool<int>.Shared.Return(deque);
        }
    }

    static NivaraColumn<T> rollingExtreme<T>(T[] effective, bool[] valid, int windowSize, int minPeriods, bool takeMax)
        where T : struct, INumber<T>
    {
        var length = effective.Length;
        var deque = new int[length];
        int head = 0;
        int tail = 0;
        int validInWindow = 0;
        var result = new T[length];
        var resultMask = new bool[length];

        for (int i = 0; i < length; i++)
        {
            int lo = i - windowSize + 1;

            while (head < tail && deque[head] < lo)
                head++;

            if (lo - 1 >= 0 && valid[lo - 1])
                validInWindow--;

            if (valid[i])
            {
                validInWindow++;
                var current = effective[i];

                while (head < tail && (takeMax ? effective[deque[tail - 1]] <= current : effective[deque[tail - 1]] >= current))
                    tail--;

                deque[tail++] = i;
            }

            if (validInWindow >= minPeriods)
                result[i] = effective[deque[head]];
            else
            {
                resultMask[i] = true;
            }
        }

        return NivaraColumn<T>.CreateFromSpans(result, resultMask);
    }

    static NivaraColumn<T> shiftCore<T>(NivaraColumn<T> column, int periods, T fill, bool hasFill)
    {
        var length = column.Length;
        if (length == 0)
            return NivaraColumn<T>.Create(Array.Empty<T>());

        var hasInputNulls = column.HasNulls;
        var result = new T[length];
        var resultMask = new bool[length];

        if (!hasInputNulls)
        {
            column.TryGetSpan(out var span);

            for (int i = 0; i < length; i++)
            {
                int src = i - periods;
                if (src >= 0 && src < length)
                    result[i] = span[src];
                else if (hasFill)
                    result[i] = fill;
                else
                {
                    resultMask[i] = true;
                }
            }
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                int src = i - periods;
                if (src >= 0 && src < length)
                {
                    result[i] = column[src];
                    if (column.IsNull(src))
                    {
                        resultMask[i] = true;
                    }
                }
                else if (hasFill)
                    result[i] = fill;
                else
                {
                    resultMask[i] = true;
                }
            }
        }

        return NivaraColumn<T>.CreateFromSpans(result, resultMask);
    }
}
