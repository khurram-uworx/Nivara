using System.Runtime.InteropServices;

namespace Nivara.Extensions.Blas;

/// <summary>
/// Hand-written CBLAS binding for the official OpenBLAS release, used by the GEMM and GEMV providers and by the
/// <c>--cpu-gemm</c> and <c>--cpu-gemv</c> probes ([docs/BLAS-MKL.md](../../docs/BLAS-MKL.md) §6).
///
/// Seven declarations, not a generated binding. Running a P/Invoke generator over
/// <c>cblas.h</c> produces the whole BLAS + LAPACK surface, which is the opposite of what this
/// probe is for; the surface the probe needs is the table below.
///
/// Only the LP64 ABI is declared (<c>int</c> dimensions). The official Windows release reports
/// <c>DYNAMIC_ARCH</c> and no <c>USE64BITINT</c> in <c>openblas_get_config()</c>, which is LP64.
/// An ILP64 build (the scipy/numpy convention, <c>cblas_sgemm64_</c>, <c>long long</c>
/// dimensions) would silently misread every dimension, so <see cref="TryInitialize" /> asserts
/// the ABI instead of discovering it from a wrong answer.
///
/// The DLL is resolved at runtime from the first of three candidate locations that exists and
/// loads: <c>lib/libopenblas.dll</c> relative to the executable, relative to the current
/// working directory, or next to the executable. An explicit path supplied via
/// <see cref="NivaraBlas.UseBlas(string, int, int)" /> takes priority on the first load.
/// If none is found, <see cref="TryInitialize" /> returns <c>false</c> with a message naming
/// the canonical path and the download URL, and the BLAS legs are skipped. No build-time
/// gating, no environment variable, no <c>OPENBLAS_DLL</c>.
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

    /// <summary>
    /// An explicit path supplied by the caller (via <see cref="NivaraBlas.UseBlas(string, int, int)" />).
    /// When set, the resolver tries this first.
    /// </summary>
    internal static string? ExplicitPath;

    /// <summary>The file the resolver actually loaded, recorded on the first successful
    /// resolution. Null until the library has loaded. The runtime caches the first
    /// resolution per assembly load context, so a later, different <see cref="ExplicitPath" />
    /// cannot take effect; <see cref="TryInitialize" /> reports this.</summary>
    internal static string? LoadedPath;

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

            // Explicit path takes priority.
            if (!string.IsNullOrEmpty(ExplicitPath) && File.Exists(ExplicitPath) &&
                NativeLibrary.TryLoad(ExplicitPath!, out IntPtr explicitHandle))
            {
                LoadedPath ??= ExplicitPath;
                return explicitHandle;
            }

            foreach (string candidate in CandidatePaths())
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
                {
                    LoadedPath ??= candidate;
                    return handle;
                }
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
        // Explicit path: validate before any P/Invoke so this is order-independent.
        if (!string.IsNullOrEmpty(ExplicitPath))
        {
            string explicitFull = Path.GetFullPath(ExplicitPath!);
            if (!File.Exists(explicitFull))
            {
                ResolvedPath = explicitFull;
                detail = $"libopenblas.dll not found at {explicitFull} (explicit path).";
                return false;
            }

            // The resolver is only consulted on the first load in a process; if a
            // different file already won, the request cannot be honoured.
            if (LoadedPath is { } loaded && !string.Equals(Path.GetFullPath(loaded), explicitFull, StringComparison.OrdinalIgnoreCase))
            {
                ResolvedPath = loaded;
                detail = $"OpenBLAS is already loaded from {loaded}; the explicit path {explicitFull} " +
                         "is ignored for this process (the runtime caches the first resolution). " +
                         "Restart the process to use a different build.";
                return false;
            }

            ResolvedPath = LoadedPath ?? explicitFull;
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
                detail = $"{ex.GetType().Name}: {ex.Message} (looked at {explicitFull})";
                return false;
            }
        }

        // Already loaded from a previous registration: report it, do not re-scan.
        if (LoadedPath is { } alreadyLoaded)
        {
            ResolvedPath = alreadyLoaded;
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
                detail = $"{ex.GetType().Name}: {ex.Message} (looked at {alreadyLoaded})";
                return false;
            }
        }

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

    [LibraryImport(Library, EntryPoint = "cblas_dgemm")]
    internal static partial void Dgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        double alpha, ref double a, int lda,
        ref double b, int ldb,
        double beta, ref double c, int ldc);

    /// <summary>
    /// y = alpha·op(A)·x + beta·y, row-major. For the Nivara single-row path, A is
    /// [N x K], trans = NoTrans, M = N, N = K, lda = K.
    /// </summary>
    [LibraryImport(Library, EntryPoint = "cblas_sgemv")]
    internal static partial void Sgemv(
        int order, int transA,
        int m, int n,
        float alpha, ref float a, int lda,
        ref float x, int incX,
        float beta, ref float y, int incY);

    [LibraryImport(Library, EntryPoint = "cblas_dgemv")]
    internal static partial void Dgemv(
        int order, int transA,
        int m, int n,
        double alpha, ref double a, int lda,
        ref double x, int incX,
        double beta, ref double y, int incY);

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
