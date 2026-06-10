namespace Nivara.AutoDiff;

/// <summary>
/// Concrete implementation of IAutoGradNumeric for float (single-precision floating point).
/// Provides automatic differentiation support for 32-bit floating point operations.
/// </summary>
public readonly struct Float32 : IAutoGradNumeric<float>
{
    /// <inheritdoc />
    public static float Zero => 0.0f;

    /// <inheritdoc />
    public static float One => 1.0f;

    /// <inheritdoc />
    public static float FromDouble(double value) => (float)value;

    /// <inheritdoc />
    public static double ToDouble(float value) => (double)value;
}

/// <summary>
/// Concrete implementation of IAutoGradNumeric for double (double-precision floating point).
/// Provides automatic differentiation support for 64-bit floating point operations.
/// </summary>
public readonly struct Float64 : IAutoGradNumeric<double>
{
    /// <inheritdoc />
    public static double Zero => 0.0;

    /// <inheritdoc />
    public static double One => 1.0;

    /// <inheritdoc />
    public static double FromDouble(double value) => value;

    /// <inheritdoc />
    public static double ToDouble(double value) => value;
}
