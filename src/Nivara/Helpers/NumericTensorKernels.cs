using System.Numerics;
using System.Numerics.Tensors;
using Nivara.Primitives;

namespace Nivara.Helpers;

/// <summary>
/// Constrained generic kernels for element-wise column arithmetic and comparison.
/// <typeparamref name="T"/> must satisfy the generic math operator interfaces, which
/// enables generic <see cref="TensorPrimitives"/> SIMD paths for every numeric primitive
/// (not just float/double) and direct operator-based comparison loops.
/// <see cref="Nivara.NivaraColumn{T}"/> is unconstrained, so it dispatches to concrete
/// instantiations of this type after a runtime <c>typeof(T)</c> check.
/// </summary>
static class NumericTensorKernels<T>
    where T : struct,
        IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>,
        IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>,
        IEqualityOperators<T, T, bool>, IComparisonOperators<T, T, bool>,
        INumber<T>
{
    public static void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => WidenPrimitives.Add(x, y, destination);

    public static void Add(ReadOnlySpan<T> x, T y, Span<T> destination)
        => TensorPrimitives.Add(x, y, destination);

    public static void Subtract(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => WidenPrimitives.Subtract(x, y, destination);

    public static void Subtract(ReadOnlySpan<T> x, T y, Span<T> destination)
        => TensorPrimitives.Subtract(x, y, destination);

    /// <summary>
    /// Computes <c>scalar - y[i]</c> element-wise. <see cref="TensorPrimitives"/> has no
    /// scalar-first subtract overload on the in-repo BCL version, so this uses a direct
    /// <see cref="INumber{T}"/> loop (functionally identical for the numeric domain).
    /// </summary>
    public static void SubtractFrom(T scalar, ReadOnlySpan<T> y, Span<T> destination)
    {
        for (int i = 0; i < y.Length; i++)
        {
            destination[i] = scalar - y[i];
        }
    }

    public static void Divide(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => WidenPrimitives.Divide(x, y, destination);

    public static void Divide(ReadOnlySpan<T> x, T y, Span<T> destination)
        => TensorPrimitives.Divide(x, y, destination);

    /// <summary>
    /// Computes <c>scalar / y[i]</c> element-wise. <see cref="TensorPrimitives"/> has no
    /// scalar-first divide overload on the in-repo BCL version, so this uses a direct
    /// <see cref="INumber{T}"/> loop (functionally identical for the numeric domain).
    /// </summary>
    public static void DivideBy(T scalar, ReadOnlySpan<T> y, Span<T> destination)
    {
        for (int i = 0; i < y.Length; i++)
        {
            destination[i] = scalar / y[i];
        }
    }

    public static void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => WidenPrimitives.Multiply(x, y, destination);

    public static void Multiply(ReadOnlySpan<T> x, T y, Span<T> destination)
        => TensorPrimitives.Multiply(x, y, destination);

    public static void Equals(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] == y[i];
        }
    }

    public static void Equals(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] == y;
        }
    }

    public static void GreaterThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] > y[i];
        }
    }

    public static void GreaterThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] > y;
        }
    }

    public static void LessThan(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] < y[i];
        }
    }

    public static void LessThan(ReadOnlySpan<T> x, T y, Span<bool> destination)
    {
        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] < y;
        }
    }

    public static T Sum(ReadOnlySpan<T> values)
        => TensorPrimitives.Sum(values);

    /// <summary>
    /// Computes <c>sum / count</c>. <c>CreateChecked</c> matches the previous per-type
    /// <c>Unsafe.As</c> casts: truncating integer division for the integer types and true
    /// division for float/double/Half/decimal (including <c>char</c>, whose generic
    /// division truncates the promoted quotient identically).
    /// </summary>
    public static T DivideByCount(T sum, int count)
        => sum / T.CreateChecked(count);

    public static T Min(ReadOnlySpan<T> values)
        => TensorPrimitives.Min(values);

    public static T Max(ReadOnlySpan<T> values)
        => TensorPrimitives.Max(values);
}
