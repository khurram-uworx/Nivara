#if BLAS_PROBE
using System.Runtime.InteropServices;

namespace Nivara.PerformanceTests;

/// <summary>
/// Hand-written CBLAS binding for the official OpenBLAS release, used only by the
/// <c>--cpu-gemm</c> probe ([docs/BLAS-MKL.md](../../docs/BLAS-MKL.md) §6).
///
/// Seven declarations, not a generated binding. Running a P/Invoke generator over
/// <c>cblas.h</c> produces the whole BLAS + LAPACK surface, which is the opposite of what this
/// probe is for; the surface the probe needs is the table below.
///
/// Only the LP64 ABI is declared (<c>int</c> dimensions). The official Windows release reports
/// <c>DYNAMIC_ARCH</c> and no <c>USE64BITINT</c> in <c>openblas_get_config()</c>, which is LP64.
/// An ILP64 build (the scipy/numpy convention, <c>cblas_sgemm64_</c>, <c>long long</c>
/// dimensions) would silently misread every dimension, so <see cref="TryInitialize"/> asserts
/// the ABI instead of discovering it from a wrong answer.
///
/// No <c>NativeLibrary.SetDllImportResolver</c> and no <c>DefaultDllImportSearchPaths</c>:
/// with no search-path flags the runtime probes the assembly directory and then falls back to
/// the OS, so a <c>libopenblas.dll</c> copied next to the executable resolves. Specifying
/// <c>DllImportSearchPath.AssemblyDirectory</c> would both drop that fallback (.NET 10+) and
/// trip CA5393.
///
/// <c>LibraryImportAttribute</c> has no <c>CallingConvention</c> property — source-generated
/// P/Invoke always uses the platform default. On x64 Windows there is one convention, so this
/// is correct here; on x86 the Cdecl/StdCall distinction would matter and this binding would
/// need <c>DllImport</c> with <c>CallingConvention.Cdecl</c> instead. The probe is x64-only.
/// </summary>
internal static partial class OpenBlasInterop
{
    const string Library = "libopenblas";

    internal const int RowMajor = 101;
    internal const int NoTrans = 111;
    internal const int Trans = 112;

    /// <summary>The build's self-reported configuration, e.g. <c>DYNAMIC_ARCH ... MAX_THREADS=64</c>.</summary>
    internal static string Config => ReadAnsi(GetConfig());

    /// <summary>The kernel OpenBLAS actually dispatched to, e.g. <c>SkylakeX</c> or <c>Haswell</c>.</summary>
    internal static string CoreName => ReadAnsi(GetCoreName());

    /// <summary>
    /// Confirms the library loads and reports the LP64 ABI. Returns <c>false</c> with a reason
    /// rather than throwing, so a machine without the DLL prints "not available" and skips the
    /// legs — never a silent pass, never a failed build.
    /// </summary>
    internal static bool TryInitialize(out string detail)
    {
        try
        {
            string config = Config;
            if (config.Contains("USE64BITINT", StringComparison.Ordinal))
            {
                detail = $"ILP64 build ({config}) — int64 dimensions, not the LP64 ABI this binding declares";
                return false;
            }

            detail = config;
            return true;
        }
        catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException or BadImageFormatException)
        {
            detail = ex.Message;
            return false;
        }
    }

    /// <summary>
    /// C = alpha·op(A)·op(B) + beta·C, row-major. <paramref name="a"/> is A as [M x K]
    /// (<c>lda = K</c>); <paramref name="b"/> is [N x K] when <c>transB = Trans</c>
    /// (<c>ldb = K</c>) or [K x N] when <c>transB = NoTrans</c> (<c>ldb = N</c>).
    /// </summary>
    [LibraryImport(Library, EntryPoint = "cblas_sgemm")]
    internal static partial void Sgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        float alpha, ref float a, int lda,
        ref float b, int ldb,
        float beta, ref float c, int ldc);

    [LibraryImport(Library, EntryPoint = "openblas_get_config")]
    private static partial IntPtr GetConfig();

    [LibraryImport(Library, EntryPoint = "openblas_get_corename")]
    private static partial IntPtr GetCoreName();

    /// <summary>
    /// OpenBLAS's own pool size. Set to 1 when Nivara's <c>Parallel.For</c> already owns the
    /// parallelism — two pools on one machine oversubscribe. Unlike MKL, this is safe to call
    /// before the library has initialised.
    /// </summary>
    [LibraryImport(Library, EntryPoint = "openblas_set_num_threads")]
    internal static partial void SetNumThreads(int threads);

    [LibraryImport(Library, EntryPoint = "openblas_get_num_threads")]
    internal static partial int GetNumThreads();

    static string ReadAnsi(IntPtr value) =>
        value == IntPtr.Zero ? "(null)" : Marshal.PtrToStringAnsi(value) ?? "(null)";
}
#endif
