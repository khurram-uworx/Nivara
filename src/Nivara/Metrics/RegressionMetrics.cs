using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Runtime.ExceptionServices;

namespace Nivara.Metrics;

/// <summary>
/// Regression summary frames. Values are converted to <see cref="double"/> once, then the entire
/// calculation runs through <see cref="TensorPrimitives"/> over compacted, pooled buffers.
/// </summary>
internal static class RegressionMetrics
{
    const int PoolThreshold = 1024;

    static readonly HashSet<Type> numericTypes =
    [
        typeof(byte),
        typeof(sbyte),
        typeof(short),
        typeof(ushort),
        typeof(int),
        typeof(uint),
        typeof(long),
        typeof(ulong),
        typeof(nint),
        typeof(nuint),
        typeof(float),
        typeof(double),
        typeof(decimal),
        typeof(Half),
        typeof(BFloat16),
    ];

    static readonly MethodInfo buildMethod = typeof(RegressionMetrics)
        .GetMethod(nameof(buildSummary), BindingFlags.NonPublic | BindingFlags.Static)!;

    static readonly ConcurrentDictionary<(Type Predicted, Type Actual), MethodInfo> kernels = new();

    internal static bool IsNumeric(Type type) => numericTypes.Contains(type);

    internal static MetricsResult BuildSummary(IColumn predicted, IColumn actual)
    {
        var kernel = kernels.GetOrAdd(
            (predicted.ElementType, actual.ElementType),
            static key => buildMethod.MakeGenericMethod(key.Predicted, key.Actual));

        try
        {
            return (MetricsResult)kernel.Invoke(null, [predicted, actual])!;
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
    }

    static MetricsResult buildSummary<TPredicted, TActual>(IColumn predicted, IColumn actual)
        where TPredicted : struct, INumberBase<TPredicted>
        where TActual : struct, INumberBase<TActual>
    {
        var predictedColumn = (NivaraColumn<TPredicted>)predicted;
        var actualColumn = (NivaraColumn<TActual>)actual;

        if (predictedColumn is NivaraColumn<double> predictedDoubles
            && actualColumn is NivaraColumn<double> actualDoubles
            && predictedDoubles.TryGetSpan(out var predictedSpan)
            && actualDoubles.TryGetSpan(out var actualSpan))
        {
            ensureFinite(predictedSpan, "predicted");
            ensureFinite(actualSpan, "actual");
            return summarize(predictedSpan, actualSpan);
        }

        int length = predictedColumn.Length;
        bool predictedHasNulls = predictedColumn.HasNulls;
        bool actualHasNulls = actualColumn.HasNulls;

        var predictedValues = rent(length);
        var actualValues = rent(length);
        try
        {
            int validPairs = 0;
            for (int i = 0; i < length; i++)
            {
                if (predictedHasNulls && predictedColumn.IsNull(i))
                    continue;
                if (actualHasNulls && actualColumn.IsNull(i))
                    continue;

                double predictedValue = double.CreateChecked(predictedColumn[i]);
                if (!double.IsFinite(predictedValue))
                    throw new ArgumentException(
                        $"Regression requires finite values; non-finite value at row {i}, column predicted.",
                        "predicted");

                double actualValue = double.CreateChecked(actualColumn[i]);
                if (!double.IsFinite(actualValue))
                    throw new ArgumentException(
                        $"Regression requires finite values; non-finite value at row {i}, column actual.",
                        "actual");

                predictedValues[validPairs] = predictedValue;
                actualValues[validPairs] = actualValue;
                validPairs++;
            }

            return summarize(
                predictedValues.AsSpan(0, validPairs),
                actualValues.AsSpan(0, validPairs));
        }
        finally
        {
            returnBuffer(predictedValues, length);
            returnBuffer(actualValues, length);
        }
    }

    static MetricsResult summarize(ReadOnlySpan<double> predicted, ReadOnlySpan<double> actual)
    {
        int validPairs = predicted.Length;
        if (validPairs == 0)
            throw new ArgumentException(
                "No row has a value in both predicted and actual; regression metrics need at least "
                + "one non-null pair.",
                nameof(predicted));

        var residual = rent(validPairs);
        try
        {
            var residualSpan = residual.AsSpan(0, validPairs);
            TensorPrimitives.Subtract(predicted, actual, residualSpan);
            var residualSumOfSquares = TensorPrimitives.SumOfSquares<double>(residualSpan);
            double mae = TensorPrimitives.SumOfMagnitudes<double>(residualSpan) / validPairs;
            double mse = residualSumOfSquares / validPairs;
            double rmse = Math.Sqrt(mse);

            double mean = TensorPrimitives.Average<double>(actual);
            TensorPrimitives.Subtract(actual, mean, residualSpan);
            double totalSumOfSquares = TensorPrimitives.SumOfSquares<double>(residualSpan);
            double r2 = totalSumOfSquares == 0
                ? (residualSumOfSquares == 0 ? 1.0 : 0.0)
                : 1.0 - (residualSumOfSquares / totalSumOfSquares);

            var frame = NivaraFrame.Create(
                ("Count", (IColumn)NivaraColumn<long>.Create(new long[] { validPairs })),
                ("Mae", (IColumn)NivaraColumn<double>.Create(new double[] { mae })),
                ("Mse", (IColumn)NivaraColumn<double>.Create(new double[] { mse })),
                ("Rmse", (IColumn)NivaraColumn<double>.Create(new double[] { rmse })),
                ("R2", (IColumn)NivaraColumn<double>.Create(new double[] { r2 })));

            return new MetricsResult(frame, validPairs);
        }
        finally
        {
            returnBuffer(residual, validPairs);
        }
    }

    static void ensureFinite(ReadOnlySpan<double> values, string column)
    {
        if (TensorPrimitives.IsFiniteAll<double>(values))
            return;

        for (int i = 0; i < values.Length; i++)
        {
            if (!double.IsFinite(values[i]))
                throw new ArgumentException(
                    $"Regression requires finite values; non-finite value at row {i}, column {column}.",
                    column);
        }
    }

    static double[] rent(int length)
        => length > PoolThreshold ? ArrayPool<double>.Shared.Rent(length) : new double[length];

    static void returnBuffer(double[] buffer, int length)
    {
        if (length > PoolThreshold)
            ArrayPool<double>.Shared.Return(buffer);
    }
}
