using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.Primitives;

/// <summary>
/// Dispatch surface for the narrow-float SIMD layer. Widens <c>Half</c>/
/// <c>BFloat16</c> to <c>float</c>, runs the genuinely-SIMD float
/// <c>TensorPrimitives</c> kernels, and narrows back. <c>float</c>/<c>double</c>
/// pass straight through to <c>TensorPrimitives</c> with no conversion.
/// </summary>
/// <remarks>
/// Phase 0 delivers the dispatch contract and selection logic only — the widen
/// kernel bodies are stubbed to fall back to the scalar <c>TensorPrimitives&lt;T&gt;</c>
/// call, so runtime behavior is unchanged while <see cref="NivaraPrimitives.UseWidenSimd"/>
/// is off. The stubbed branches are the implementation targets for Phase 1,
/// seeded by the validated kernels in the Nivara.SimdProbe probe.
/// </remarks>
public static class WidenPrimitives
{
    /// <summary>
    /// Length gate: widening is only worthwhile above this threshold — the probe
    /// shows narrow-float dot is slower for tiny vectors (n &lt; 128), so small
    /// buffers stay scalar.
    /// </summary>
    public static bool ShouldWiden<T>(int length) where T : struct, IFloatingPointIeee754<T>
        => ShouldWiden(typeof(T), length);

    /// <summary>
    /// Length gate operating on a runtime <paramref name="type"/>, for callers
    /// without a <c>struct</c>/<c>IFloatingPointIeee754</c> constraint.
    /// </summary>
    public static bool ShouldWiden(Type type, int length)
        => NivaraPrimitives.UseWidenSimd
           && Vector.IsHardwareAccelerated
           && IsNarrowFloat(type)
           && length >= Vector<byte>.Count * 4;

    /// <summary>
    /// Returns <c>true</c> for the narrow 16-bit float types this layer widens.
    /// </summary>
    public static bool IsNarrowFloat<T>() where T : struct, IFloatingPointIeee754<T>
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
    /// narrows the result back to <c>T</c>.
    /// </summary>
    public static T Dot<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            // Phase 0 stub: widen-compute-narrow kernel lands in Phase 1.
            // Falls through to the scalar TensorPrimitives<T> backend unchanged.
        }

        return TensorPrimitives.Dot(x, y);
    }

    /// <summary>
    /// Element-wise add. For narrow floats this widens both operands, adds in
    /// float, and narrows back to <c>T</c>.
    /// </summary>
    public static void Add<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            // Phase 0 stub: widen-compute-narrow kernel lands in Phase 1.
        }

        TensorPrimitives.Add(x, y, destination);
    }

    /// <summary>
    /// Element-wise multiply. For narrow floats this widens both operands,
    /// multiplies in float, and narrows back to <c>T</c>.
    /// </summary>
    public static void Multiply<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y, Span<T> destination)
        where T : struct, IFloatingPointIeee754<T>
    {
        if (ShouldWiden<T>(x.Length))
        {
            // Phase 0 stub: widen-compute-narrow kernel lands in Phase 1.
        }

        TensorPrimitives.Multiply(x, y, destination);
    }
}
