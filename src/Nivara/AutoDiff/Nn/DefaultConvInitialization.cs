using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// The built-in weight draw shared by the convolution layers' no-initializer path: values uniform
/// in <c>[-√(2/fanIn), √(2/fanIn)]</c> drawn from a seed-42 stream. Centralised so the draw is
/// defined once; the fixed seed keeps identically-shaped layers initializing identically. Pass an
/// <c>IInitializer&lt;T&gt;</c> to a layer's constructor to override it.
/// </summary>
internal static class DefaultConvInitialization
{
    /// <summary>
    /// Fills <paramref name="destination"/> with the default convolution weight draw for the given
    /// <paramref name="fanIn"/>. Deterministic: the same shape always yields the same values.
    /// </summary>
    /// <param name="destination">The weight buffer to fill</param>
    /// <param name="fanIn">The fan-in used to scale the uniform bound</param>
    public static void Fill<T>(Span<T> destination, int fanIn) where T : struct, IFloatingPointIeee754<T>
    {
        var bound = T.CreateChecked(Math.Sqrt(2.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < destination.Length; i++)
            destination[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(bound));
    }
}
