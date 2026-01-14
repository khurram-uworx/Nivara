using System.Numerics;
using Nivara;
using Nivara.Extensions.AutoDiff.Utilities;

namespace Nivara.Extensions.AutoDiff.Extensions;

/// <summary>
/// Extension methods that provide seamless integration 
/// between Nivara types (Column, Series, Frame) and automatic differentiation.
/// Includes type validation to ensure only supported numeric types are used.
/// </summary>
public static class NivaraAutoGradExtensions
{
    /// <summary>
    /// Converts a NivaraColumn to a GradTensor with type validation.
    /// </summary>
    /// <typeparam name="T">The numeric type (must be float or double)</typeparam>
    /// <param name="column">The column to convert</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor wrapping the column</returns>
    /// <exception cref="ArgumentNullException">Thrown when column is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type</exception>
    public static GradTensor<T> ToGradTensor<T>(this NivaraColumn<T> column, bool requiresGrad = false)
        where T : struct, INumber<T>
    {
        if (column == null)
            throw new ArgumentNullException(nameof(column));

        // Type validation is performed in GradTensor constructor
        return GradTensor<T>.FromColumn(column, requiresGrad);
    }

    /// <summary>
    /// Converts a NivaraSeries to a GradTensor with type validation.
    /// </summary>
    /// <typeparam name="T">The numeric type (must be float or double)</typeparam>
    /// <param name="series">The series to convert</param>
    /// <param name="requiresGrad">Whether the tensor should track gradients</param>
    /// <returns>A new GradTensor wrapping the series values</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="AutoGradException">Thrown when T is not a supported type</exception>
    public static GradTensor<T> ToGradTensor<T>(this NivaraSeries<T> series, bool requiresGrad = false)
        where T : struct, INumber<T>
    {
        if (series == null)
            throw new ArgumentNullException(nameof(series));

        // Type validation is performed in GradTensor constructor
        return GradTensor<T>.FromSeries(series, requiresGrad);
    }

    /// <summary>
    /// Checks if a type is supported for automatic differentiation.
    /// </summary>
    /// <typeparam name="T">The type to check</typeparam>
    /// <returns>True if the type is supported; otherwise, false</returns>
    public static bool IsAutoGradSupported<T>() where T : struct, INumber<T>
    {
        return TypeValidator.IsSupported<T>();
    }

    /// <summary>
    /// Gets the list of types supported for automatic differentiation.
    /// </summary>
    /// <returns>An array of supported types</returns>
    public static Type[] GetSupportedAutoGradTypes()
    {
        return TypeValidator.GetSupportedTypes();
    }
}
