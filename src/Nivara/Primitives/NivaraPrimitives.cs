using System.Numerics;

namespace Nivara.Primitives;

/// <summary>
/// Global toggles and settings for the narrow-float SIMD layer
/// (<see cref="WidenPrimitives"/>).
/// </summary>
public static class NivaraPrimitives
{
    static readonly bool appContextWidenSimd = AppContext.TryGetSwitch("Nivara.Primitives.WidenSimd", out var enabled)
        && enabled;

    static bool useWidenSimd;

    /// <summary>
    /// Enables or disables the widen-compute-narrow SIMD path for
    /// <c>Half</c>/<c>BFloat16</c>.
    /// </summary>
    /// <remarks>
    /// Defaults to <c>false</c>. When <c>true</c>, the widen path only takes
    /// effect for types the hardware supports and over the length threshold —
    /// see <see cref="WidenPrimitives.ShouldWiden{T}(int)"/>. Off by default so
    /// existing behavior is bit-identical.
    /// </remarks>
    public static bool UseWidenSimd
    {
        get => appContextWidenSimd || useWidenSimd;
        set
        {
            if (value && !Vector.IsHardwareAccelerated)
                return;
            useWidenSimd = value;
        }
    }
}
