using Apache.Arrow;
using Apache.Arrow.Arrays;
using Nivara.IO;

namespace Nivara.IO;

/// <summary>
/// Extension methods for NivaraSeries providing fluent API for Arrow operations
/// </summary>
/// <remarks>
/// These extension methods provide idiomatic C# integration with Apache Arrow array operations,
/// following .NET naming conventions and ensuring proper generic type constraints.
/// </remarks>
public static class NivaraSeriesExtensions
{
    /// <summary>
    /// Converts the NivaraSeries to an Apache Arrow Array
    /// </summary>
    /// <typeparam name="T">The type of data in the series</typeparam>
    /// <param name="series">The NivaraSeries to convert</param>
    /// <param name="options">Optional Arrow conversion options</param>
    /// <returns>An Apache Arrow Array</returns>
    /// <exception cref="ArgumentNullException">Thrown when series is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the series type is not supported</exception>
    public static IArrowArray ToArrowArray<T>(this NivaraSeries<T> series, ArrowConversionOptions? options = null)
    {
        return ArrowInterop.ToArrowArray(series, options);
    }

    /// <summary>
    /// Creates a NivaraSeries from an Apache Arrow Array
    /// </summary>
    /// <typeparam name="T">The type of data for the series</typeparam>
    /// <param name="arrowArray">The Apache Arrow Array to convert</param>
    /// <param name="options">Optional Arrow conversion options</param>
    /// <returns>A NivaraSeries</returns>
    /// <exception cref="ArgumentNullException">Thrown when arrowArray is null</exception>
    /// <exception cref="UnsupportedTypeException">Thrown when the Arrow array type is not supported</exception>
    public static NivaraSeries<T> ToNivaraSeries<T>(this IArrowArray arrowArray, ArrowConversionOptions? options = null)
    {
        return ArrowInterop.FromArrowArray<T>(arrowArray, options);
    }
}