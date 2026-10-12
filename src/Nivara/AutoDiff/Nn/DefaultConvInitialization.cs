using System.Numerics;

namespace Nivara.AutoDiff.Nn;

/// <summary>
/// The built-in weight draw shared by the convolution layers' no-initializer path: the He-uniform
/// draw, values uniform in <c>[-√(6/fanIn), √(6/fanIn)]</c> drawn from a seed-42 stream. Its
/// variance (<c>2/fanIn</c>) matches He-normal <c>N(0, √(2/fanIn))</c> and the library's
/// <c>KaimingUniformInitializer&lt;T&gt;</c>, so linear and convolution defaults share one family.
/// <c>fanIn</c> is the weight's second shape dimension (the matrix column count) — the same value
/// <c>KaimingUniformInitializer&lt;T&gt;</c> and PyTorch's <c>_calculate_fan_in_and_fan_out</c> compute.
/// For <c>ConvTranspose2d</c> that is <c>outChannels × kernel area</c>, deliberately matching the
/// initializer convention rather than the input fan-in. Centralised so the draw is defined once; the
/// fixed seed keeps identically-shaped layers initializing identically. Pass an
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
        var bound = T.CreateChecked(Math.Sqrt(6.0 / fanIn));
        var rng = new Random(42);
        for (int i = 0; i < destination.Length; i++)
            destination[i] = T.CreateChecked((rng.NextDouble() * 2.0 - 1.0) * double.CreateChecked(bound));
    }
}
