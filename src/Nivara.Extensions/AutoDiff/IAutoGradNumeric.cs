using System.Numerics;

namespace Nivara.Extensions.AutoDiff;

/// <summary>
/// Interface for numeric types that support automatic differentiation.
/// This is a simplified interface for the initial setup - full implementation will be added in later tasks.
/// </summary>
/// <typeparam name="T">The numeric type</typeparam>
public interface IAutoGradNumeric<T> where T : struct, INumber<T>
{
    /// <summary>
    /// Gets the zero value for the type
    /// </summary>
    static abstract T Zero { get; }

    /// <summary>
    /// Gets the one (identity) value for the type
    /// </summary>
    static abstract T One { get; }

    /// <summary>
    /// Creates a value from a double (for gradient initialization)
    /// </summary>
    /// <param name="value">The double value to convert</param>
    /// <returns>The converted value of type T</returns>
    static abstract T FromDouble(double value);

    /// <summary>
    /// Converts the value to a double (for numerical computations)
    /// </summary>
    /// <param name="value">The value to convert</param>
    /// <returns>The value as a double</returns>
    static abstract double ToDouble(T value);
}