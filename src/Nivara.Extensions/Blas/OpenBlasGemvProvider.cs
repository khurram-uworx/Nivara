using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Nivara.Extensions.Blas;

/// <summary>
/// <see cref="IGemvProvider" /> backed by <c>cblas_sgemv</c> / <c>cblas_dgemv</c>.
/// One call, no row-block split: M is 1, so the GEMM <c>Parallel.For</c> does not apply.
/// The OpenBLAS pool size is the same knob as <see cref="OpenBlasGemmProvider" />.
/// </summary>
public sealed class OpenBlasGemvProvider : IGemvProvider
{
    readonly int openBlasThreads;

    public OpenBlasGemvProvider(int openBlasThreads = 1)
    {
        this.openBlasThreads = openBlasThreads < 1 ? 1 : openBlasThreads;
        if (!OpenBlasInterop.TryInitialize(out string detail))
            throw new InvalidOperationException($"OpenBLAS is not available: {detail}");
    }

    public void Gemv(ReadOnlySpan<float> x, ReadOnlySpan<float> a, Span<float> y, int n, int k)
    {
        if (n <= 0 || k <= 0)
            return;

        OpenBlasInterop.SetNumThreads(openBlasThreads);
        ref float aRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(a));
        ref float xRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(x));
        ref float yRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(y));
        OpenBlasInterop.Sgemv(
            OpenBlasInterop.RowMajor, OpenBlasInterop.NoTrans,
            n, k, 1f, ref aRef, k, ref xRef, 1, 0f, ref yRef, 1);
    }

    public void Gemv(ReadOnlySpan<double> x, ReadOnlySpan<double> a, Span<double> y, int n, int k)
    {
        if (n <= 0 || k <= 0)
            return;

        OpenBlasInterop.SetNumThreads(openBlasThreads);
        ref double aRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(a));
        ref double xRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(x));
        ref double yRef = ref Unsafe.AsRef(in MemoryMarshal.GetReference(y));
        OpenBlasInterop.Dgemv(
            OpenBlasInterop.RowMajor, OpenBlasInterop.NoTrans,
            n, k, 1.0, ref aRef, k, ref xRef, 1, 0.0, ref yRef, 1);
    }
}
