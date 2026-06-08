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
        int n = sourceData.Length;

        if (!sourceData.HasNulls)
        {
            sourceData.TryGetSpan(out var span);
            var targetData = new TTarget[n];
            for (int i = 0; i < n; i++)
                targetData[i] = TTarget.CreateChecked(span[i]);
            return new ReverseGradTensor<TTarget>(NivaraColumn<TTarget>.Create(targetData), resultRequiresGrad, source.shape);
        }

        var buf = System.Buffers.ArrayPool<TSource>.Shared.Rent(n);
        var nullMask = new bool[n]; // allocated fresh, already zeroed
        try
        {
            sourceData.CopyTo(buf.AsSpan(0, n), default);
            sourceData.TryGetNullMask(out var mask);
            mask.CopyTo(nullMask);

            var targetData = new TTarget[n];
            for (int i = 0; i < n; i++)
            {
                if (!nullMask[i])
                    targetData[i] = TTarget.CreateChecked(buf[i]);
            }

            return new ReverseGradTensor<TTarget>(
                NivaraColumn<TTarget>.CreateFromSpans(targetData, nullMask),
                resultRequiresGrad, source.shape);
        }
        finally
        {
            System.Buffers.ArrayPool<TSource>.Shared.Return(buf, clearArray: true);
        }
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
