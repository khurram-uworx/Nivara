namespace Nivara;

/// <summary>
/// An immutable pair of registered BLAS providers. Held in a single
/// <see cref="ProviderRegistry.Current" /> slot so a reader can never observe a
/// half-registration (GEMM set, GEMV still null).
/// </summary>
internal sealed class ProviderRegistration(IGemmProvider? gemm, IGemvProvider? gemv)
{
    internal IGemmProvider? Gemm { get; } = gemm;

    internal IGemvProvider? Gemv { get; } = gemv;
}

/// <summary>
/// Single atomic slot for the registered BLAS providers. Replaces the two
/// independent mutable statics; reads are lock-free, writes are a single
/// <see cref="Volatile" /> swap of an immutable value.
/// </summary>
internal static class ProviderRegistry
{
    internal static volatile ProviderRegistration? Current;
}