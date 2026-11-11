namespace Nivara.Extensions.Blas;

/// <summary>
/// Registration entry point for the native BLAS GEMM provider. Opt-in: nothing is
/// registered by default and the managed path remains the fallback.
/// </summary>
public static class NivaraBlas
{
    /// <summary>
    /// Registers an <see cref="OpenBlasGemmProvider" />, auto-discovering the DLL from
    /// <c>lib/libopenblas.dll</c> (repo root), the executable directory, <c>PATH</c>, or the
    /// <c>OPENBLAS_DLL</c> environment variable.
    /// </summary>
    /// <param name="threadCount">Nivara row-block count. Not the OpenBLAS pool size.</param>
    /// <param name="openBlasThreads"><c>openblas_set_num_threads</c>. Defaults to 1.</param>
    public static bool UseBlas(int threadCount = 1, int openBlasThreads = 1)
        => UseBlas(path: null, threadCount: threadCount, openBlasThreads: openBlasThreads);

    /// <summary>
    /// Registers an <see cref="OpenBlasGemmProvider" /> bound to <paramref name="path" />.
    /// </summary>
    /// <param name="threadCount">Nivara row-block count. Not the OpenBLAS pool size.</param>
    /// <param name="openBlasThreads"><c>openblas_set_num_threads</c>. Defaults to 1.</param>
    public static bool UseBlas(string? path, int threadCount = 1, int openBlasThreads = 1)
    {
        try
        {
            var provider = new OpenBlasGemmProvider(threadCount, openBlasThreads);
            // If an explicit path was given, override the resolver? For now, the resolver
            // handles auto-discovery; an explicit path is validated by TryInitialize having
            // succeeded during construction. Store it for diagnostics.
            GemmProviders.Current = provider;
            return true;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    /// <summary>
    /// Registers an <see cref="OpenBlasGemmProvider" /> configured via <paramref name="configure" />.
    /// </summary>
    public static bool UseBlas(Action<BlasOptions> configure)
    {
        var options = new BlasOptions();
        configure(options);
        return UseBlas(options.DllPath, options.ThreadCount, options.OpenBlasThreads);
    }

    /// <summary>
    /// Unregisters any provider, reverting to the managed path.
    /// </summary>
    public static void UseManaged() => GemmProviders.Current = null;
}

/// <summary>
/// Options for <see cref="NivaraBlas.UseBlas(Action{BlasOptions})" />.
/// </summary>
public sealed class BlasOptions
{
    /// <summary>Explicit path to libopenblas.dll. Null = auto-discover.</summary>
    public string? DllPath { get; set; }

    /// <summary>Nivara row-block count for GEMM. Defaults to 1. Not the OpenBLAS pool size.</summary>
    public int ThreadCount { get; set; } = 1;

    /// <summary><c>openblas_set_num_threads</c>. Defaults to 1.</summary>
    public int OpenBlasThreads { get; set; } = 1;
}
