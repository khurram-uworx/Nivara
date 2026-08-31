using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;

namespace Nivara.Primitives;

/// <summary>
/// Dispatch surface for the narrow-float SIMD layer. Widens <c>Half</c>/
/// <c>BFloat16</c> to <c>float</c>, runs the genuinely-SIMD float
/// <c>TensorPrimitives</c> kernels, and narrows back. <c>float</c>/<c>double</c>
/// pass straight through to <c>TensorPrimitives</c> with no conversion.
/// </summary>
/// <remarks>
/// Constrained to <see cref="INumber{T}"/> (not just
/// <see cref="IFloatingPointIeee754{T}"/>) so both the column layer
/// (<c>NumericTensorKernels</c>, <c>INumber</c>-constrained) and AutoDiff
/// (<c>IFloatingPointIeee754</c> ⊆ <c>INumber</c>) consume the same surface.
/// The widen path operates on raw bit patterns, so it needs no floating-point
/// interface methods.
///
/// The widen branch is selected only when <see cref="ShouldWiden{T}(int)"/> says
/// so (toggle on + hardware accelerated + length above threshold + narrow type).
/// When it is not selected — the default, toggle-off state — each method falls
/// through to the exact <c>TensorPrimitives</c> call, bit-identical to the
/// pre-Phase-1 behavior.
/// </remarks>
public static class WidenPrimitives
{
    /// <summary>
    /// Length gate: widening is only worthwhile above this threshold — the probe
    /// shows narrow-float dot is slower for tiny vectors (n &lt; 128), so small
    /// buffers stay scalar.
    /// </summary>
    public static bool ShouldWiden<T>(int length) where T : struct, INumber<T>
        => ShouldWiden(typeof(T), length);

    /// <summary>
    /// Length gate operating on a runtime <paramref name="type"/>, for callers
    /// without a <c>struct</c>/<c>INumber</c> constraint.
    /// </summary>
    public static bool ShouldWiden(Type type, int length)
        => NivaraPrimitives.UseWidenSimd
           && Vector.IsHardwareAccelerated
           && IsNarrowFloat(type)
           && length >= Vector<byte>.Count * 4;

    /// <summary>
    /// Returns <c>true</c> for the narrow 16-bit float types this layer widens.
    /// </summary>
    public static bool IsNarrowFloat<T>() where T : struct, INumber<T>
        => IsNarrowFloat(typeof(T));

    /// <summary>
    /// Returns <c>true</c> for the narrow 16-bit float types this layer widens,
    /// operating on a runtime <paramref name="type"/>.
    /// </summary>
    public static bool IsNarrowFloat(Type type)
        => type == typeof(Half) || type == typeof(BFloat16);

    /// <summary>
    /// Computes the dot product of <paramref name="x"/> and <paramref name="y"/>.
    /// For narrow floats this widens to float, uses the float SIMD backend, and
    /// returns a result narrowed back to <c>T</c>.
    /// </summary>
    public static T Dot<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
        where T : struct, INumber<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            if (typeof(T) == typeof(BFloat16))
                return (T)(object)NarrowFloatKernels.Dot(MemoryMarshal.Cast<T, BFloat16>(x), MemoryMarshal.Cast<T, BFloat16>(y));
            if (typeof(T) == typeof(Half))
                return (T)(object)NarrowFloatKernels.Dot(MemoryMarshal.Cast<T, Half>(x), MemoryMarshal.Cast<T, Half>(y));
        }

        return TensorPrimitives.Dot(x, y);
    }

    /// <summary>
    /// Element-wise add. For narrow floats this widens both operands, adds in
    /// float, and narrows back to <c>T</c>.
    /// </summary>
    public static void Add<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            if (typeof(T) == typeof(BFloat16))
            {
                NarrowFloatKernels.Add(MemoryMarshal.Cast<T, BFloat16>(x), MemoryMarshal.Cast<T, BFloat16>(y), MemoryMarshal.Cast<T, BFloat16>(destination));
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                NarrowFloatKernels.Add(MemoryMarshal.Cast<T, Half>(x), MemoryMarshal.Cast<T, Half>(y), MemoryMarshal.Cast<T, Half>(destination));
                return;
            }
        }

        TensorPrimitives.Add(x, y, destination);
    }

    /// <summary>
    /// Element-wise subtract. For narrow floats this widens both operands,
    /// subtracts in float, and narrows back to <c>T</c>.
    /// </summary>
    public static void Subtract<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            if (typeof(T) == typeof(BFloat16))
            {
                NarrowFloatKernels.Subtract(MemoryMarshal.Cast<T, BFloat16>(x), MemoryMarshal.Cast<T, BFloat16>(y), MemoryMarshal.Cast<T, BFloat16>(destination));
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                NarrowFloatKernels.Subtract(MemoryMarshal.Cast<T, Half>(x), MemoryMarshal.Cast<T, Half>(y), MemoryMarshal.Cast<T, Half>(destination));
                return;
            }
        }

        TensorPrimitives.Subtract(x, y, destination);
    }

    /// <summary>
    /// Element-wise multiply. For narrow floats this widens both operands,
    /// multiplies in float, and narrows back to <c>T</c>.
    /// </summary>
    public static void Multiply<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            if (typeof(T) == typeof(BFloat16))
            {
                NarrowFloatKernels.Multiply(MemoryMarshal.Cast<T, BFloat16>(x), MemoryMarshal.Cast<T, BFloat16>(y), MemoryMarshal.Cast<T, BFloat16>(destination));
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                NarrowFloatKernels.Multiply(MemoryMarshal.Cast<T, Half>(x), MemoryMarshal.Cast<T, Half>(y), MemoryMarshal.Cast<T, Half>(destination));
                return;
            }
        }

        TensorPrimitives.Multiply(x, y, destination);
    }

    /// <summary>
    /// Element-wise divide. For narrow floats this widens both operands, divides
    /// in float, and narrows back to <c>T</c>.
    /// </summary>
    public static void Divide<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        where T : struct, INumber<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            if (typeof(T) == typeof(BFloat16))
            {
                NarrowFloatKernels.Divide(MemoryMarshal.Cast<T, BFloat16>(x), MemoryMarshal.Cast<T, BFloat16>(y), MemoryMarshal.Cast<T, BFloat16>(destination));
                return;
            }
            if (typeof(T) == typeof(Half))
            {
                NarrowFloatKernels.Divide(MemoryMarshal.Cast<T, Half>(x), MemoryMarshal.Cast<T, Half>(y), MemoryMarshal.Cast<T, Half>(destination));
                return;
            }
        }

        TensorPrimitives.Divide(x, y, destination);
    }
}
