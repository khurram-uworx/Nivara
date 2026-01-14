using System.Numerics;
using Nivara.Extensions.AutoDiff.Exceptions;

namespace Nivara.Extensions.AutoDiff.Utilities;

/// <summary>
/// Provides type conversion utilities for automatic differentiation operations.
/// Enables safe conversion between different numeric types while preserving gradient tracking.
/// </summary>
public static class TypeConverter
{
    /// <summary>
    /// Converts a GradTensor from one numeric type to another.
    /// </summary>
    /// <typeparam name="TSource">The source numeric type</typeparam>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <param name="source">The source tensor to convert</param>
    /// <param name="requiresGrad">Whether the converted tensor should track gradients</param>
    /// <returns>A new GradTensor with the converted type</returns>
    /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
    /// <exception cref="AutoGradException">Thrown when conversion is not supported</exception>
    public static GradTensor<TTarget> Convert<TSource, TTarget>(
        GradTensor<TSource> source, 
        bool? requiresGrad = null)
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        // Validate that both types are supported
        TypeValidator.ValidateNumericType<TSource>();
        TypeValidator.ValidateNumericType<TTarget>();

        // Determine if the result should require gradients
        bool resultRequiresGrad = requiresGrad ?? source.RequiresGrad;

        // Convert the data
        var sourceData = source.Data;
        var targetData = new TTarget[sourceData.Length];

        for (int i = 0; i < sourceData.Length; i++)
        {
            if (!sourceData.IsNull(i))
            {
                targetData[i] = ConvertValue<TSource, TTarget>(sourceData[i]);
            }
        }

        // Create the target column with null handling
        var targetColumn = NivaraColumn<TTarget>.Create(targetData);

        // Copy null mask if present
        if (sourceData.HasNulls)
        {
            var nullMask = new bool[sourceData.Length];
            for (int i = 0; i < sourceData.Length; i++)
            {
                nullMask[i] = sourceData.IsNull(i);
            }
            // Note: We would need to create a column with explicit null mask
            // For now, we'll create a new column and rely on the null detection
        }

        return new GradTensor<TTarget>(targetColumn, resultRequiresGrad);
    }

    /// <summary>
    /// Converts a single value from one numeric type to another.
    /// </summary>
    /// <typeparam name="TSource">The source numeric type</typeparam>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <param name="value">The value to convert</param>
    /// <returns>The converted value</returns>
    private static TTarget ConvertValue<TSource, TTarget>(TSource value)
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        // Use double as an intermediate type for conversion
        // This works for float and double, which are our supported types
        var doubleValue = double.CreateChecked(value);
        return TTarget.CreateChecked(doubleValue);
    }

    /// <summary>
    /// Converts a GradTensor to float (single-precision).
    /// </summary>
    /// <typeparam name="T">The source numeric type</typeparam>
    /// <param name="source">The source tensor to convert</param>
    /// <param name="requiresGrad">Whether the converted tensor should track gradients</param>
    /// <returns>A new GradTensor with float type</returns>
    public static GradTensor<float> ToFloat<T>(GradTensor<T> source, bool? requiresGrad = null)
        where T : struct, INumber<T>
    {
        return Convert<T, float>(source, requiresGrad);
    }

    /// <summary>
    /// Converts a GradTensor to double (double-precision).
    /// </summary>
    /// <typeparam name="T">The source numeric type</typeparam>
    /// <param name="source">The source tensor to convert</param>
    /// <param name="requiresGrad">Whether the converted tensor should track gradients</param>
    /// <returns>A new GradTensor with double type</returns>
    public static GradTensor<double> ToDouble<T>(GradTensor<T> source, bool? requiresGrad = null)
        where T : struct, INumber<T>
    {
        return Convert<T, double>(source, requiresGrad);
    }

    /// <summary>
    /// Attempts to convert a GradTensor from one type to another, returning null on failure.
    /// </summary>
    /// <typeparam name="TSource">The source numeric type</typeparam>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <param name="source">The source tensor to convert</param>
    /// <param name="requiresGrad">Whether the converted tensor should track gradients</param>
    /// <returns>A converted GradTensor, or null if conversion fails</returns>
    public static GradTensor<TTarget>? TryConvert<TSource, TTarget>(
        GradTensor<TSource> source,
        bool? requiresGrad = null)
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        try
        {
            return Convert<TSource, TTarget>(source, requiresGrad);
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Checks if conversion between two types is supported.
    /// </summary>
    /// <typeparam name="TSource">The source numeric type</typeparam>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <returns>True if conversion is supported; otherwise, false</returns>
    public static bool CanConvert<TSource, TTarget>()
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        return TypeValidator.IsSupported<TSource>() && TypeValidator.IsSupported<TTarget>();
    }
}
