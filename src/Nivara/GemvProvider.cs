namespace Nivara;

/// <summary>
/// A provider of the single-row matrix-vector product for <see cref="float" /> and
/// <see cref="double" />. Core consults the registered provider inside
/// <c>TensorsHelper.MultiplyCore</c> when <c>bTransposed &amp;&amp; aRows == 1</c>.
/// When no provider is registered, or the element type is not float or double, the
/// managed <c>Dot</c> loop runs unchanged.
/// </summary>
/// <remarks>
/// This is not <see cref="IGemmProvider" />. That contract stays dense GEMM with
/// <c>aRows &gt; 1</c>. Half, BFloat16, and a zero inner dimension stay on the managed path.
/// </remarks>
public interface IGemvProvider
{
    /// <summary>
    /// y[j] = dot(x, A[j, :]). <paramref name="a" /> is row-major <c>[n x k]</c>.
    /// </summary>
    void Gemv(ReadOnlySpan<float> x, ReadOnlySpan<float> a, Span<float> y, int n, int k);

    /// <summary>
    /// y[j] = dot(x, A[j, :]). See the float overload.
    /// </summary>
    void Gemv(ReadOnlySpan<double> x, ReadOnlySpan<double> a, Span<double> y, int n, int k);
}

/// <summary>
/// Holds the currently registered <see cref="IGemvProvider" />, if any. Extensions
/// sets this when <c>UseBlas</c> is called; core reads it inside
/// <c>TensorsHelper.MultiplyCore</c>. Core never instantiates a provider itself.
/// Backed by the single <see cref="ProviderRegistry" /> slot so a reader can never
/// observe a half-registration.
/// </summary>
internal static class GemvProviders
{
    internal static IGemvProvider? Current => ProviderRegistry.Current?.Gemv;
}
