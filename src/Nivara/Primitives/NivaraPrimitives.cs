using System.Numerics;

namespace Nivara.Primitives;

/// <summary>
/// Global toggles and settings for the narrow-float SIMD layer
/// (<see cref="WidenPrimitives"/>).
/// </summary>
public static class NivaraPrimitives
{
    /// <summary>
    /// Hidden diagnostic opt-out for the widen-compute-narrow SIMD path. Defaults to
    /// <c>false</c> (widen enabled). Flip it via
    /// <c>AppContext.SetSwitch("Nivara.Primitives.DisableWidenSimd", true)</c> (or the
    /// environment) to force the scalar fallback for diagnostics and A/B comparison.
    /// </summary>
    public static bool DisableWidenSimd
        => AppContext.TryGetSwitch("Nivara.Primitives.DisableWidenSimd", out var disabled) && disabled;
}