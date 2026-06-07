using Nivara.Extensions.AutoDiff.Exceptions;
using System.Numerics;

namespace Nivara.Extensions.AutoDiff.Utilities;

/// <summary>
/// Provides type conversion utilities for reverse-mode automatic differentiation operations.
/// Enables safe conversion between different numeric types while preserving gradient tracking.
/// </summary>
public static class TypeConverter
{
    /// <summary>
    /// Converts a ReverseGradTensor from one numeric type to another.
    /// </summary>
    /// <typeparam name="TSource">The source numeric type</typeparam>
    /// <typeparam name="TTarget">The target numeric type</typeparam>
    /// <param name="source">The source tensor to convert</param>
    /// <param name="requiresGrad">Whether the converted tensor should track gradients</param>
    /// <returns>A new ReverseGradTensor with the converted type</returns>
    /// <exception cref="ArgumentNullException">Thrown when source is null</exception>
    /// <exception cref="AutoGradException">Thrown when conversion is not supported</exception>
    public static ReverseGradTensor<TTarget> Convert<TSource, TTarget>(
        ReverseGradTensor<TSource> source,
        bool? requiresGrad = null)
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        if (source == null)
            throw new ArgumentNullException(nameof(source));

        TypeValidator.ValidateNumericType<TSource>();
        TypeValidator.ValidateNumericType<TTarget>();

        bool resultRequiresGrad = requiresGrad ?? source.RequiresGrad;

        var sourceData = source.Data;
        var targetData = new TTarget[sourceData.Length];

        for (int i = 0; i < sourceData.Length; i++)
        {
            if (!sourceData.IsNull(i))
            {
                targetData[i] = ConvertValue<TSource, TTarget>(sourceData[i]);
            }
        }

        var targetColumn = NivaraColumn<TTarget>.Create(targetData);

        if (sourceData.HasNulls)
        {
            var nullMask = new bool[sourceData.Length];
            for (int i = 0; i < sourceData.Length; i++)
            {
                nullMask[i] = sourceData.IsNull(i);
            }
        }

        return new ReverseGradTensor<TTarget>(targetColumn, resultRequiresGrad);
    }

    /// <summary>
    /// Converts a single value from one numeric type to another.
    /// </summary>
    private static TTarget ConvertValue<TSource, TTarget>(TSource value)
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        var doubleValue = double.CreateChecked(value);
        return TTarget.CreateChecked(doubleValue);
    }

    /// <summary>
    /// Converts a ReverseGradTensor to float (single-precision).
    /// </summary>
    public static ReverseGradTensor<float> ToFloat<T>(ReverseGradTensor<T> source, bool? requiresGrad = null)
        where T : struct, INumber<T>
    {
        return Convert<T, float>(source, requiresGrad);
    }

    /// <summary>
    /// Converts a ReverseGradTensor to double (double-precision).
    /// </summary>
    public static ReverseGradTensor<double> ToDouble<T>(ReverseGradTensor<T> source, bool? requiresGrad = null)
        where T : struct, INumber<T>
    {
        return Convert<T, double>(source, requiresGrad);
    }

    /// <summary>
    /// Attempts to convert a ReverseGradTensor from one type to another, returning null on failure.
    /// </summary>
    public static ReverseGradTensor<TTarget>? TryConvert<TSource, TTarget>(
        ReverseGradTensor<TSource> source,
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
    public static bool CanConvert<TSource, TTarget>()
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        return TypeValidator.IsSupported<TSource>() && TypeValidator.IsSupported<TTarget>();
    }
}
