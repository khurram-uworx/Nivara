namespace Nivara;

/// <summary>
/// A provider of dense matrix multiplication for <see cref="float" /> and
/// <see cref="double" />. Core consults the registered provider (if any) inside
/// <c>TensorsHelper.MultiplyCore</c>; when no provider is registered, or the element
/// type is not float or double, the managed path runs unchanged and bit-identically.
/// </summary>
/// <remarks>
/// Half, BFloat16, null-masked multiply, integer types, the single-row GEMV fast path
/// and transpose materialization are deliberately outside this contract and always use
/// the managed path. A native BLAS has no concept of null masks or non-float/double
/// types, and <c>Tensor.MatrixMultiply</c> (when it ships) is expected to replace the
/// managed path without touching this interface.
/// </remarks>
public interface IGemmProvider
{
    /// <summary>
    /// C = A · (B or Bᵀ) for <see cref="float" />. <paramref name="bTransposed" />
    /// indicates whether <paramref name="b" /> is stored transposed (<c>[N x K]</c>)
    /// or not (<c>[K x N]</c>), matching <c>TensorsHelper.MultiplyCore</c>.
    /// </summary>
    void MatMul(
        ReadOnlySpan<float> a, ReadOnlySpan<float> b, Span<float> c,
        int m, int k, int n, bool bTransposed);

    /// <summary>
    /// C = A · (B or Bᵀ) for <see cref="double" />. See the float overload.
    /// </summary>
    void MatMul(
        ReadOnlySpan<double> a, ReadOnlySpan<double> b, Span<double> c,
        int m, int k, int n, bool bTransposed);
}

/// <summary>
/// Holds the currently registered <see cref="IGemmProvider" />, if any. Extensions
/// sets this when <c>UseBlas</c> is called; core reads it inside
/// <c>TensorsHelper.MultiplyCore</c>. Core never instantiates a provider itself.
/// </summary>
internal static class GemmProviders
{
    internal static IGemmProvider? Current;
}
