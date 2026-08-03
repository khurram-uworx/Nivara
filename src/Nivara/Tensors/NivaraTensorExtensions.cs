using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Tensors;

/// <summary>
/// Null-aware column reductions.
/// </summary>
public static class NivaraTensorExtensions
{
    // ── Column-level terminal reductions ──

    /// <summary>
    /// Computes the sum of all non-null elements in the column.
    /// Returns T.Zero if all elements are null.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to sum</param>
    /// <returns>The sum of non-null elements</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty</exception>
    public static T Sum<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Sum on an empty column");

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                return TensorPrimitives.Sum(span);

            T sum = T.Zero;
            for (int i = 0; i < column.Length; i++)
                sum += column[i];
            return sum;
        }

        T result = T.Zero;
        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
                result += column[i];
        }
        return result;
    }

    /// <summary>
    /// Computes the mean (average) of all non-null elements in the column.
    /// Returns NaN if all elements are null.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to average</param>
    /// <returns>The mean of non-null elements as a double</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty</exception>
    public static double Mean<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Mean on an empty column");

        T sum = T.Zero;
        int count = 0;

        if (!column.HasNulls)
        {
            count = column.Length;
            if (column.TryGetSpan(out var span))
                return double.CreateChecked(TensorPrimitives.Sum(span)) / count;

            for (int i = 0; i < column.Length; i++)
                sum += column[i];
        }
        else
        {
            for (int i = 0; i < column.Length; i++)
            {
                if (!column.IsNull(i))
                {
                    sum += column[i];
                    count++;
                }
            }
        }

        return count > 0
            ? double.CreateChecked(sum) / count
            : double.NaN;
    }

    /// <summary>
    /// Returns the minimum non-null value in the column.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to find the minimum of</param>
    /// <returns>The minimum non-null value</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty or all values are null</exception>
    public static T Min<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Min on an empty column");

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                return TensorPrimitives.Min(span);

            T min = column[0];
            for (int i = 1; i < column.Length; i++)
                min = T.Min(min, column[i]);
            return min;
        }

        bool found = false;
        T result = T.Zero;
        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
            {
                if (!found)
                {
                    result = column[i];
                    found = true;
                }
                else
                {
                    result = T.Min(result, column[i]);
                }
            }
        }

        if (!found)
            throw new InvalidOperationException("Cannot compute Min on a column where all values are null");
        return result;
    }

    /// <summary>
    /// Returns the maximum non-null value in the column.
    /// </summary>
    /// <typeparam name="T">The numeric element type</typeparam>
    /// <param name="column">The column to find the maximum of</param>
    /// <returns>The maximum non-null value</returns>
    /// <exception cref="InvalidOperationException">Thrown when the column is empty or all values are null</exception>
    public static T Max<T>(this NivaraColumn<T> column)
        where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(column);
        if (column.Length == 0)
            throw new InvalidOperationException("Cannot compute Max on an empty column");

        if (!column.HasNulls)
        {
            if (column.TryGetSpan(out var span))
                return TensorPrimitives.Max(span);

            T max = column[0];
            for (int i = 1; i < column.Length; i++)
                max = T.Max(max, column[i]);
            return max;
        }

        bool found = false;
        T result = T.Zero;
        for (int i = 0; i < column.Length; i++)
        {
            if (!column.IsNull(i))
            {
                if (!found)
                {
                    result = column[i];
                    found = true;
                }
                else
                {
                    result = T.Max(result, column[i]);
                }
            }
        }

        if (!found)
            throw new InvalidOperationException("Cannot compute Max on a column where all values are null");
        return result;
    }
}
