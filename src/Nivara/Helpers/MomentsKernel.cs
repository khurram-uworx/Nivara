using System.Numerics.Tensors;

namespace Nivara.Helpers;

/// <summary>
/// Shared variance/standard-deviation computation for the aggregation domain. Both follow the
/// standard definition: variance is the mean squared deviation from the mean and the standard
/// deviation is its square root. The <c>ddof</c> parameter is the delta degrees of freedom
/// (numpy <c>ddof</c> semantics): 0 is population (divide by n), 1 is sample (divide by n - 1).
/// </summary>
internal static class MomentsKernel
{
    /// <summary>
    /// Computes the variance of a value span with the given delta degrees of freedom.
    /// </summary>
    /// <param name="values">The values to compute the variance over (non-empty)</param>
    /// <param name="ddof">Delta degrees of freedom (0 = population, 1 = sample)</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="ddof"/> is negative</exception>
    /// <exception cref="InvalidOperationException">Thrown when the span is empty or has fewer than <c>ddof + 1</c> elements</exception>
    public static double Variance(ReadOnlySpan<double> values, int ddof)
    {
        if (ddof < 0)
            throw new ArgumentOutOfRangeException(nameof(ddof), "ddof must be >= 0.");
        if (values.Length <= ddof)
            throw new InvalidOperationException($"Cannot compute variance with ddof={ddof} over {values.Length} value(s); need at least {ddof + 1}.");

        var mean = TensorPrimitives.Average<double>(values);
        double sumSquaredDeviation = 0d;
        foreach (var value in values)
        {
            var deviation = value - mean;
            sumSquaredDeviation += deviation * deviation;
        }
        return sumSquaredDeviation / (values.Length - ddof);
    }

    /// <summary>
    /// Computes the standard deviation of a value span with the given delta degrees of freedom.
    /// </summary>
    public static double StdDev(ReadOnlySpan<double> values, int ddof)
        => Math.Sqrt(Variance(values, ddof));

    /// <summary>
    /// Computes the variance from boxed valid values, converting each across the full 17-type
    /// numeric aggregation domain to double first.
    /// </summary>
    /// <returns>The variance, or null when <paramref name="validValues"/> is empty</returns>
    public static double? ComputeVarianceFromBoxed(IReadOnlyList<object> validValues, int ddof)
    {
        if (validValues.Count == 0)
            return null;

        var values = new double[validValues.Count];
        for (int i = 0; i < values.Length; i++)
            values[i] = QuantileKernel.ToDouble(validValues[i]);

        return Variance(values, ddof);
    }

    /// <summary>
    /// Computes the standard deviation from boxed valid values.
    /// </summary>
    /// <returns>The standard deviation, or null when <paramref name="validValues"/> is empty</returns>
    public static double? ComputeStdDevFromBoxed(IReadOnlyList<object> validValues, int ddof)
    {
        if (validValues.Count == 0)
            return null;

        var values = new double[validValues.Count];
        for (int i = 0; i < values.Length; i++)
            values[i] = QuantileKernel.ToDouble(validValues[i]);

        return StdDev(values, ddof);
    }
}
