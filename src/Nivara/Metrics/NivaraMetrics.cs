using Nivara.Diagnostics;

namespace Nivara.Metrics;

/// <summary>
/// Frame-producing metric operations over a predicted/actual column pair.
/// </summary>
/// <remarks>
/// These are data operations that return frames and hold no state. Pipeline-level and
/// feature-level evaluation remains ML.NET's surface; the boundary is recorded in
/// <c>docs/METRICS.md</c>.
/// Null positions never enter a denominator: a row where either column is null is excluded
/// from every metric and from every count.
/// </remarks>
public static class NivaraMetrics
{
    /// <summary>
    /// Builds a confusion matrix. Each row is a predicted class; each <c>Actual:{label}</c>
    /// column counts the rows whose actual class is that label.
    /// </summary>
    /// <param name="predicted">The predicted class column.</param>
    /// <param name="actual">The actual class column. Must have the same element type as
    /// <paramref name="predicted"/>.</param>
    /// <returns>
    /// A frame with columns <c>Predicted</c> (the class label), one <c>Actual:{label}</c>
    /// column per class, <c>Correct</c> (the diagonal cell) and <c>Support</c> (the
    /// predicted-row total). Classes are ordered by the label comparator: ordinal for
    /// <see cref="string"/>, default ordering otherwise.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="predicted"/> or
    /// <paramref name="actual"/> is null.</exception>
    /// <exception cref="ArgumentException">The columns have different lengths, different
    /// element types, are not <see cref="NivaraColumn{T}"/>-backed, the label type is not
    /// comparable, or two labels collide when written as column names ignoring case.</exception>
    public static NivaraFrame ConfusionMatrixFrame(IColumn predicted, IColumn actual)
    {
        validatePair(predicted, actual, requireSameElementType: true);

        var measurement = DiagnosticsTracker.StartMeasurement();
        var (frame, validPairs) = ClassificationMetrics.BuildConfusionMatrix(predicted, actual);
        measurement.Record(
            "Metrics.ConfusionMatrix",
            KernelType.Scalar,
            predicted.Length,
            predicted.ElementType,
            hasNulls(predicted, actual),
            usedPairs(validPairs, predicted.Length));
        return frame;
    }

    /// <summary>
    /// Builds one row per class with the confusion counts and derived precision, recall and F1.
    /// </summary>
    /// <param name="predicted">The predicted class column.</param>
    /// <param name="actual">The actual class column. Must have the same element type as
    /// <paramref name="predicted"/>.</param>
    /// <returns>
    /// A frame with columns <c>Class</c>, <c>Tp</c>, <c>Fp</c>, <c>Fn</c> (counts) and
    /// <c>Precision</c>, <c>Recall</c>, <c>F1</c>. A metric whose denominator is zero is null:
    /// precision when the class is never predicted, recall when it is never actual, and F1
    /// when either is null or both are zero.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="predicted"/> or
    /// <paramref name="actual"/> is null.</exception>
    /// <exception cref="ArgumentException">The columns have different lengths, different
    /// element types, are not <see cref="NivaraColumn{T}"/>-backed, or the label type is not
    /// comparable.</exception>
    public static NivaraFrame PerClassAccuracyFrame(IColumn predicted, IColumn actual)
    {
        validatePair(predicted, actual, requireSameElementType: true);

        var measurement = DiagnosticsTracker.StartMeasurement();
        var (frame, validPairs) = ClassificationMetrics.BuildPerClass(predicted, actual);
        measurement.Record(
            "Metrics.PerClassAccuracy",
            KernelType.Scalar,
            predicted.Length,
            predicted.ElementType,
            hasNulls(predicted, actual),
            usedPairs(validPairs, predicted.Length));
        return frame;
    }

    /// <summary>
    /// Builds a one-row regression summary: <c>Count</c>, <c>Mae</c>, <c>Mse</c>, <c>Rmse</c>
    /// and <c>R2</c>, accumulated in <see cref="double"/>.
    /// </summary>
    /// <param name="predicted">The predicted value column.</param>
    /// <param name="actual">The actual value column. May use a different numeric element type
    /// than <paramref name="predicted"/>.</param>
    /// <returns>
    /// A one-row frame. For a constant target, <c>R2</c> is 1 when every residual is zero and
    /// 0 otherwise.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="predicted"/> or
    /// <paramref name="actual"/> is null.</exception>
    /// <exception cref="ArgumentException">The columns have different lengths, are not
    /// <see cref="NivaraColumn{T}"/>-backed, an element type is not numeric, or no row has a
    /// value in both columns.</exception>
    public static NivaraFrame RegressionSummaryFrame(IColumn predicted, IColumn actual)
    {
        validatePair(predicted, actual, requireSameElementType: false);

        if (!RegressionMetrics.IsNumeric(predicted.ElementType))
            throw new ArgumentException(
                $"Regression requires numeric columns; '{predicted.ElementType.Name}' is not numeric.",
                nameof(predicted));

        if (!RegressionMetrics.IsNumeric(actual.ElementType))
            throw new ArgumentException(
                $"Regression requires numeric columns; '{actual.ElementType.Name}' is not numeric.",
                nameof(actual));

        var measurement = DiagnosticsTracker.StartMeasurement();
        var (frame, validPairs) = RegressionMetrics.BuildSummary(predicted, actual);
        measurement.Record(
            "Metrics.RegressionSummary",
            KernelSelector.DetermineKernelType<double>(validPairs),
            predicted.Length,
            predicted.ElementType,
            hasNulls(predicted, actual),
            usedPairs(validPairs, predicted.Length));
        return frame;
    }

    static void validatePair(IColumn predicted, IColumn actual, bool requireSameElementType)
    {
        ArgumentNullException.ThrowIfNull(predicted);
        ArgumentNullException.ThrowIfNull(actual);

        requireNivaraColumn(predicted, nameof(predicted));
        requireNivaraColumn(actual, nameof(actual));

        if (predicted.Length != actual.Length)
            throw new ArgumentException(
                $"predicted and actual must have the same length; got {predicted.Length} and {actual.Length}.",
                nameof(actual));

        if (requireSameElementType && predicted.ElementType != actual.ElementType)
            throw new ArgumentException(
                "Classification requires predicted and actual to have the same element type; "
                + $"got '{predicted.ElementType.Name}' and '{actual.ElementType.Name}'.",
                nameof(actual));
    }

    static void requireNivaraColumn(IColumn column, string parameterName)
    {
        var expected = typeof(NivaraColumn<>).MakeGenericType(column.ElementType);
        if (!expected.IsInstanceOfType(column))
            throw new ArgumentException(
                $"Expected a {expected.Name} instance; metadata requires column-backed data, but got "
                + $"'{column.GetType().Name}'.",
                parameterName);
    }

    static bool hasNulls(IColumn predicted, IColumn actual) => predicted.HasNulls || actual.HasNulls;

    static string usedPairs(int validPairs, int total)
        => $"{validPairs} of {total} pairs used";
}

/// <summary>
/// A metric frame plus the number of row pairs that survived null exclusion, so diagnostics can
/// report how much of the input the result was computed from.
/// </summary>
internal readonly record struct MetricsResult(NivaraFrame Frame, int ValidPairs);
