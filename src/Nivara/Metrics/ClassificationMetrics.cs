using System.Buffers;
using System.Collections.Concurrent;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

namespace Nivara.Metrics;

/// <summary>
/// Confusion-matrix and per-class accuracy frames. Counting is a single typed hash pass, so the
/// result is reported as the scalar kernel rather than pretending SIMD applies to class counting.
/// </summary>
internal static class ClassificationMetrics
{
    const int PoolThreshold = 1024;

    static readonly MethodInfo buildMatrixMethod = typeof(ClassificationMetrics)
        .GetMethod(nameof(buildMatrix), BindingFlags.NonPublic | BindingFlags.Static)!;

    static readonly MethodInfo buildPerClassMethod = typeof(ClassificationMetrics)
        .GetMethod(nameof(buildPerClass), BindingFlags.NonPublic | BindingFlags.Static)!;

    static readonly ConcurrentDictionary<Type, MethodInfo> matrixKernels = new();
    static readonly ConcurrentDictionary<Type, MethodInfo> perClassKernels = new();

    internal static MetricsResult BuildConfusionMatrix(IColumn predicted, IColumn actual)
    {
        ensureComparableLabels(predicted);
        var kernel = matrixKernels.GetOrAdd(
            predicted.ElementType,
            static type => buildMatrixMethod.MakeGenericMethod(type));
        return invoke(kernel, predicted, actual);
    }

    internal static MetricsResult BuildPerClass(IColumn predicted, IColumn actual)
    {
        ensureComparableLabels(predicted);
        var kernel = perClassKernels.GetOrAdd(
            predicted.ElementType,
            static type => buildPerClassMethod.MakeGenericMethod(type));
        return invoke(kernel, predicted, actual);
    }

    internal static bool IsComparable(Type type)
        => type == typeof(string)
            || typeof(IComparable).IsAssignableFrom(type)
            || typeof(IComparable<>).MakeGenericType(type).IsAssignableFrom(type);

    static void ensureComparableLabels(IColumn predicted)
    {
        if (!IsComparable(predicted.ElementType))
            throw new ArgumentException(
                $"Classification labels of type '{predicted.ElementType.Name}' must be comparable so "
                + "the class order is deterministic.",
                nameof(predicted));
    }

    static MetricsResult invoke(MethodInfo kernel, IColumn predicted, IColumn actual)
    {
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

    static MetricsResult buildMatrix<T>(IColumn predicted, IColumn actual) where T : notnull
    {
        var counts = countColumns<T>(predicted, actual, out var values);
        try
        {
            return new MetricsResult(buildMatrixFrame(counts, predicted), counts.ValidPairs);
        }
        finally
        {
            values.Release();
        }
    }

    static MetricsResult buildPerClass<T>(IColumn predicted, IColumn actual) where T : notnull
    {
        var counts = countColumns<T>(predicted, actual, out var values);
        try
        {
            return new MetricsResult(buildPerClassFrame(counts), counts.ValidPairs);
        }
        finally
        {
            values.Release();
        }
    }

    static ClassificationCounts<T> countColumns<T>(
        IColumn predicted,
        IColumn actual,
        out ColumnBuffers<T> values) where T : notnull
    {
        int length = predicted.Length;
        var predictedValues = readValues((NivaraColumn<T>)predicted, length, out var predictedMask);
        T[] actualValues;
        bool[]? actualMask;
        try
        {
            actualValues = readValues((NivaraColumn<T>)actual, length, out actualMask);
        }
        catch
        {
            release(predictedValues, predictedMask, length);
            throw;
        }

        values = new ColumnBuffers<T>(predictedValues, predictedMask, actualValues, actualMask, length);
        try
        {
            return count(predictedValues, predictedMask, actualValues, actualMask, length);
        }
        catch
        {
            values.Release();
            throw;
        }
    }

    static T[] readValues<T>(NivaraColumn<T> column, int length, out bool[]? nullMask)
    {
        bool hasNulls = column.HasNulls;
        var values = length > PoolThreshold ? ArrayPool<T>.Shared.Rent(length) : new T[length];
        if (hasNulls)
        {
            nullMask = length > PoolThreshold ? ArrayPool<bool>.Shared.Rent(length) : new bool[length];
            column.CopyTo(values.AsSpan(0, length), default!, nullMask.AsSpan(0, length));
        }
        else
        {
            nullMask = null;
            column.CopyTo(values.AsSpan(0, length), default!);
        }

        return values;
    }

    static void release<T>(T[] values, bool[]? nullMask, int length)
    {
        if (length > PoolThreshold)
            ArrayPool<T>.Shared.Return(values, clearArray: RuntimeHelpers.IsReferenceOrContainsReferences<T>());

        if (nullMask is not null && length > PoolThreshold)
            ArrayPool<bool>.Shared.Return(nullMask);
    }

    static ClassificationCounts<T> count<T>(
        T[] predicted,
        bool[]? predictedMask,
        T[] actual,
        bool[]? actualMask,
        int length) where T : notnull
    {
        var seen = new HashSet<T>(EqualityComparer<T>.Default);
        var labels = new List<T>();

        for (int i = 0; i < length; i++)
        {
            if (!isValid(predictedMask, actualMask, i))
                continue;

            if (seen.Add(predicted[i]))
                labels.Add(predicted[i]);
            if (seen.Add(actual[i]))
                labels.Add(actual[i]);
        }

        if (seen.Count == 0)
            throw new ArgumentException(
                "No row has a value in both predicted and actual; classification metrics need at "
                + "least one non-null pair.",
                nameof(predicted));

        labels.Sort(sortComparer<T>());

        int classes = labels.Count;
        var index = new Dictionary<T, int>(classes, EqualityComparer<T>.Default);
        for (int i = 0; i < classes; i++)
            index[labels[i]] = i;

        var matrix = new long[classes * classes];
        int validPairs = 0;

        for (int i = 0; i < length; i++)
        {
            if (!isValid(predictedMask, actualMask, i))
                continue;

            matrix[(index[predicted[i]] * classes) + index[actual[i]]]++;
            validPairs++;
        }

        return new ClassificationCounts<T>(labels.ToArray(), matrix, validPairs);
    }

    static bool isValid(bool[]? predictedMask, bool[]? actualMask, int row)
        => (predictedMask is null || !predictedMask[row]) && (actualMask is null || !actualMask[row]);

    static IComparer<T> sortComparer<T>()
        => typeof(T) == typeof(string)
            ? (IComparer<T>)(object)StringComparer.Ordinal
            : Comparer<T>.Default;

    static NivaraFrame buildMatrixFrame<T>(ClassificationCounts<T> counts, IColumn predicted) where T : notnull
    {
        int classes = counts.ClassCount;
        var perActual = new long[classes][];
        var support = new long[classes];
        var correct = new long[classes];

        for (int c = 0; c < classes; c++)
            perActual[c] = new long[classes];

        for (int row = 0; row < classes; row++)
        {
            long rowTotal = 0;
            for (int column = 0; column < classes; column++)
            {
                var value = counts.Matrix[(row * classes) + column];
                perActual[column][row] = value;
                rowTotal += value;
            }

            support[row] = rowTotal;
            correct[row] = counts.Matrix[(row * classes) + row];
        }

        var columns = new List<(string Name, IColumn Column)>(classes + 3);
        var names = new Dictionary<string, T>(StringComparer.OrdinalIgnoreCase);
        var actualNames = new string[classes];
        for (int c = 0; c < classes; c++)
        {
            var name = "Actual:" + labelText(counts.Labels[c]);
            if (names.TryGetValue(name, out var existing))
                throw new ArgumentException(
                    $"Class labels '{existing}' and '{counts.Labels[c]}' produce the same column name "
                    + $"'{name}' ignoring case; labels must stay distinct because frame column names do.",
                    nameof(predicted));

            names[name] = counts.Labels[c];
            actualNames[c] = name;
        }

        columns.Add(("Predicted", (IColumn)NivaraColumn<T>.Create(counts.Labels)));
        for (int c = 0; c < classes; c++)
            columns.Add((actualNames[c], (IColumn)NivaraColumn<long>.Create(perActual[c])));

        columns.Add(("Correct", (IColumn)NivaraColumn<long>.Create(correct)));
        columns.Add(("Support", (IColumn)NivaraColumn<long>.Create(support)));
        return NivaraFrame.Create(columns.ToArray());
    }

    static NivaraFrame buildPerClassFrame<T>(ClassificationCounts<T> counts) where T : notnull
    {
        int classes = counts.ClassCount;
        var tp = new long[classes];
        var fp = new long[classes];
        var fn = new long[classes];
        var precision = new double?[classes];
        var recall = new double?[classes];
        var f1 = new double?[classes];

        for (int c = 0; c < classes; c++)
        {
            long predictedTotal = 0;
            long actualTotal = 0;
            for (int other = 0; other < classes; other++)
            {
                predictedTotal += counts.Matrix[(c * classes) + other];
                actualTotal += counts.Matrix[(other * classes) + c];
            }

            var diagonal = counts.Matrix[(c * classes) + c];
            tp[c] = diagonal;
            fp[c] = predictedTotal - diagonal;
            fn[c] = actualTotal - diagonal;

            double? classPrecision = predictedTotal == 0 ? null : (double)diagonal / predictedTotal;
            double? classRecall = actualTotal == 0 ? null : (double)diagonal / actualTotal;
            precision[c] = classPrecision;
            recall[c] = classRecall;
            f1[c] = classPrecision is null || classRecall is null
                || classPrecision.Value + classRecall.Value == 0.0
                    ? null
                    : 2.0 * classPrecision.Value * classRecall.Value
                        / (classPrecision.Value + classRecall.Value);
        }

        return NivaraFrame.Create(
            ("Class", (IColumn)NivaraColumn<T>.Create(counts.Labels)),
            ("Tp", (IColumn)NivaraColumn<long>.Create(tp)),
            ("Fp", (IColumn)NivaraColumn<long>.Create(fp)),
            ("Fn", (IColumn)NivaraColumn<long>.Create(fn)),
            ("Precision", (IColumn)NivaraColumnFactory.CreateFromNullable<double>(precision)),
            ("Recall", (IColumn)NivaraColumnFactory.CreateFromNullable<double>(recall)),
            ("F1", (IColumn)NivaraColumnFactory.CreateFromNullable<double>(f1)));
    }

    static string labelText<T>(T label) where T : notnull => label.ToString() ?? string.Empty;

    readonly struct ColumnBuffers<T>(T[] predicted, bool[]? predictedMask, T[] actual, bool[]? actualMask, int length)
    {
        public void Release()
        {
            release(predicted, predictedMask, length);
            release(actual, actualMask, length);
        }
    }

    sealed class ClassificationCounts<T>(T[] labels, long[] matrix, int validPairs)
    {
        public T[] Labels { get; } = labels;

        public long[] Matrix { get; } = matrix;

        public int ValidPairs { get; } = validPairs;

        public int ClassCount => Labels.Length;
    }
}
