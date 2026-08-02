using System.Numerics;
using System.Numerics.Tensors;

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
    where T :
        IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>,
        IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>,
        IEqualityOperators<T, T, bool>, IComparisonOperators<T, T, bool>
{
    public static void Add(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => TensorPrimitives.Add(x, y, destination);

    public static void Add(ReadOnlySpan<T> x, T y, Span<T> destination)
        => TensorPrimitives.Add(x, y, destination);

    public static void Multiply(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        => TensorPrimitives.Multiply(x, y, destination);

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
}
