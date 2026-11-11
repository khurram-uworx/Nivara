using System.Runtime.InteropServices;

namespace Nivara.PerformanceTests;

/// <summary>
/// Hand-written CBLAS binding for the official OpenBLAS release, used by the
/// <c>--cpu-gemm</c> and <c>--cpu-gemv</c> probes ([docs/BLAS-MKL.md](../../docs/BLAS-MKL.md) §6).
///
/// The declarations below, not a generated binding. Running a P/Invoke generator over
/// <c>cblas.h</c> produces the whole BLAS + LAPACK surface, which is the opposite of what this
/// probe is for; the surface the probe needs is the table below.
///
/// Only the LP64 ABI is declared (<c>int</c> dimensions). The official Windows release reports
/// <c>DYNAMIC_ARCH</c> and no <c>USE64BITINT</c> in <c>openblas_get_config()</c>, which is LP64.
/// An ILP64 build (the scipy/numpy convention, <c>cblas_sgemm64_</c>, <c>long long</c>
/// dimensions) would silently misread every dimension, so <see cref="TryInitialize" /> asserts
/// the ABI instead of discovering it from a wrong answer.
///
/// The DLL is resolved at runtime from a single canonical location — <c>lib/libopenblas.dll</c>
/// at the repository root (i.e. <c>../../lib/libopenblas.dll</c> relative to the executable).
/// If it is missing, <see cref="TryInitialize" /> returns <c>false</c> with a message naming
/// that path and the download URL, and the BLAS legs are skipped. No build-time gating, no
/// environment variable, no search path.
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

    /// <summary>The canonical location of the DLL, relative to the repository root.</summary>
    internal const string CanonicalRelativePath = "lib/libopenblas.dll";

    /// <summary>The full path the probe looked for, populated by <see cref="TryInitialize" />.</summary>
    internal static string ResolvedPath { get; private set; } = string.Empty;

    /// <summary>The build's self-reported configuration, e.g. <c>DYNAMIC_ARCH ... MAX_THREADS=64</c>.</summary>
    internal static string Config => ReadAnsi(GetConfig());

    /// <summary>The kernel OpenBLAS actually dispatched to, e.g. <c>SkylakeX</c> or <c>Haswell</c>.</summary>
    internal static string CoreName => ReadAnsi(GetCoreName());

    static OpenBlasInterop()
    {
        // Resolve libopenblas from the canonical repo-root location instead of relying on the
        // default search path (assembly dir / PATH). SetDllImportResolver must run before any
        // P/Invoke into this type, so it lives in the static constructor.
        NativeLibrary.SetDllImportResolver(typeof(OpenBlasInterop).Assembly, (libraryName, assembly, searchPath) =>
        {
            if (libraryName != Library)
                return IntPtr.Zero;

            foreach (string candidate in CandidatePaths())
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
                    return handle;
            }

            return IntPtr.Zero;
        });
    }

    /// <summary>
    /// Candidate locations for the DLL, checked in order. The first that exists and loads wins.
    /// </summary>
    static IEnumerable<string> CandidatePaths()
    {
        // Relative to the executable: tests/.../bin/Release/net11.0 -> repo root lib/.
        yield return Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "lib", "libopenblas.dll"));
        // Relative to the current working directory (repo root when invoked via dotnet run).
        yield return Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), "lib", "libopenblas.dll"));
        // Next to the executable, as a last resort.
        yield return Path.Combine(AppContext.BaseDirectory, "libopenblas.dll");
    }

    /// <summary>
    /// Confirms the library loads and reports the LP64 ABI. Returns <c>false</c> with a reason
    /// rather than throwing, so a machine without the DLL prints "not available" and skips the
    /// legs — never a silent pass, never a failed build.
    /// </summary>
    internal static bool TryInitialize(out string detail)
    {
        string? found = null;
        foreach (string candidate in CandidatePaths())
        {
            if (File.Exists(candidate))
            {
                found = candidate;
                break;
            }
        }

        ResolvedPath = found ?? Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), CanonicalRelativePath));

        if (found is null)
        {
            detail = $"libopenblas.dll not found at {ResolvedPath}. " +
                     "Download OpenBLAS 0.3.34 (x64) from " +
                     "https://github.com/xianyi/OpenBLAS/releases/tag/v0.3.34 " +
                     "(asset OpenBLAS-0.3.34-x64.zip), extract libopenblas.dll to " +
                     "lib/libopenblas.dll at the repository root, then re-run.";
            return false;
        }

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
            detail = $"{ex.GetType().Name}: {ex.Message} (looked at {found})";
            return false;
        }
    }

    /// <summary>
    /// C = alpha·op(A)·op(B) + beta·C, row-major. <paramref name="a" /> is A as [M x K]
    /// (<c>lda = K</c>); <paramref name="b" /> is [N x K] when <c>transB = Trans</c>
    /// (<c>ldb = K</c>) or [K x N] when <c>transB = NoTrans</c> (<c>ldb = N</c>).
    /// </summary>
    [LibraryImport(Library, EntryPoint = "cblas_sgemm")]
    internal static partial void Sgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        float alpha, ref float a, int lda,
        ref float b, int ldb,
        float beta, ref float c, int ldc);

    /// <summary>
    /// y = alpha·op(A)·x + beta·y, row-major. For the Nivara single-row path, A is
    /// <c>[N x K]</c>, <c>trans = NoTrans</c>, <c>M = N</c>, <c>N = K</c>, <c>lda = K</c>.
    /// </summary>
    [LibraryImport(Library, EntryPoint = "cblas_sgemv")]
    internal static partial void Sgemv(
        int order, int transA,
        int m, int n,
        float alpha, ref float a, int lda,
        ref float x, int incX,
        float beta, ref float y, int incY);

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
