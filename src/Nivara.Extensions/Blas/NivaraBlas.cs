using Nivara;
using System.Runtime.CompilerServices;

namespace Nivara.Extensions.Blas;

/// <summary>
/// Registration entry point for the native BLAS providers. Opt-in: nothing is
/// registered by default and the managed path remains the fallback.
/// Registration is process-wide: <see cref="MultiplyCore{T}"/> has no execution
/// context, so a registration applies to every caller in the process. Use
/// <see cref="EnterBlas(string?, int, int)" /> for a scoped registration that
/// restores the previous state on dispose, and <see cref="UseManaged" /> to revert.
/// </summary>
public static class NivaraBlas
{
    static readonly object Gate = new();
    static int? previousOpenBlasThreads;
    static string? lastDiagnostic;

    /// <summary>
    /// The diagnostic produced by the most recent registration attempt: the
    /// <c>TryInitialize</c> detail on failure (including the offending path) or
    /// the resolved OpenBLAS configuration on success.
    /// </summary>
    public static string? LastDiagnostic => lastDiagnostic;

    /// <summary>
    /// Registers an <see cref="OpenBlasGemmProvider" />, auto-discovering the DLL from
    /// <c>lib/libopenblas.dll</c> at the repository root, the executable directory, or next
    /// to the executable.
    /// </summary>
    /// <param name="threadCount">Nivara row-block count. Not the OpenBLAS pool size.</param>
    /// <param name="openBlasThreads"><c>openblas_set_num_threads</c>. Applied once, at
    /// registration; process-wide. Defaults to 1.</param>
    public static bool UseBlas(int threadCount = 1, int openBlasThreads = 1)
        => UseBlas(path: null, threadCount: threadCount, openBlasThreads: openBlasThreads);

    /// <summary>
    /// Registers an <see cref="OpenBlasGemmProvider" /> bound to <paramref name="path" />.
    /// The path is honoured on the process's first OpenBLAS load; a later registration
    /// that names a different file is rejected (the runtime caches the first resolution).
    /// </summary>
    /// <param name="threadCount">Nivara row-block count. Not the OpenBLAS pool size.</param>
    /// <param name="openBlasThreads"><c>openblas_set_num_threads</c>. Applied once, at
    /// registration; process-wide. Defaults to 1.</param>
    public static bool UseBlas(string? path, int threadCount = 1, int openBlasThreads = 1)
        => TryRegister(path, threadCount, openBlasThreads, out _);

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
    /// Registers an <see cref="OpenBlasGemmProvider" /> scoped to the returned
    /// <see cref="IDisposable" />: disposing restores the registration (and the OpenBLAS
    /// thread count) that was in effect before. Returns <c>null</c> when registration
    /// fails; <see cref="LastDiagnostic" /> explains why.
    /// </summary>
    public static IDisposable? EnterBlas(string? path = null, int threadCount = 1, int openBlasThreads = 1)
    {
        if (!TryRegister(path, threadCount, openBlasThreads, out var scope))
            return null;
        return scope;
    }

    /// <summary>
    /// Registers an <see cref="OpenBlasGemmProvider" /> configured via <paramref name="configure" />,
    /// scoped to the returned <see cref="IDisposable" />.
    /// </summary>
    public static IDisposable? EnterBlas(Action<BlasOptions> configure)
    {
        var options = new BlasOptions();
        configure(options);
        return EnterBlas(options.DllPath, options.ThreadCount, options.OpenBlasThreads);
    }

    /// <summary>
    /// Unregisters any provider, reverting to the managed path, and restores the OpenBLAS
    /// thread count that was in effect before registration.
    /// </summary>
    public static void UseManaged()
    {
        lock (Gate)
        {
            ProviderRegistry.Current = null;
            if (previousOpenBlasThreads is int threads)
            {
                try { OpenBlasInterop.SetNumThreads(threads); }
                catch (DllNotFoundException) { /* library was never loaded */ }
                previousOpenBlasThreads = null;
            }
        }
    }

    static bool TryRegister(string? path, int threadCount, int openBlasThreads, out IDisposable? scope)
    {
        scope = null;
        lock (Gate)
        {
            try
            {
                // Must be set before any P/Invoke into OpenBlasInterop.
                OpenBlasInterop.ExplicitPath = path;

                if (!OpenBlasInterop.TryInitialize(out string detail))
                {
                    lastDiagnostic = detail;
                    ProviderRegistry.Current = null;
                    return false;
                }

                // Capture the pool size before the providers overwrite it, so unregister
                // can restore it.
                int previousThreads = OpenBlasInterop.GetNumThreads();

                var gemm = new OpenBlasGemmProvider(threadCount, openBlasThreads);
                var gemv = new OpenBlasGemvProvider(openBlasThreads);

                var previousRegistration = ProviderRegistry.Current;
                ProviderRegistry.Current = new ProviderRegistration(gemm, gemv);
                previousOpenBlasThreads = previousThreads;
                lastDiagnostic = OpenBlasInterop.Config;

                var capturedPrevious = previousRegistration;
                var capturedPreviousThreads = previousThreads;
                scope = new RegistrationScope(capturedPrevious, capturedPreviousThreads);
                return true;
            }
            catch (InvalidOperationException ex)
            {
                lastDiagnostic = ex.Message;
                ProviderRegistry.Current = null;
                return false;
            }
        }
    }

    sealed class RegistrationScope(ProviderRegistration? previousRegistration, int previousThreads) : IDisposable
    {
        int disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;
            lock (Gate)
            {
                ProviderRegistry.Current = previousRegistration;
                try { OpenBlasInterop.SetNumThreads(previousThreads); }
                catch (DllNotFoundException) { /* library was never loaded */ }
            }
        }
    }
}

/// <summary>
/// Options for <see cref="NivaraBlas.UseBlas(Action{BlasOptions})" />.
/// </summary>
public sealed class BlasOptions
{
    /// <summary>Explicit path to libopenblas.dll. Null = auto-discover. Honoured on the
    /// process's first OpenBLAS load; a later different path is rejected.</summary>
    public string? DllPath { get; set; }

    /// <summary>Nivara row-block count for GEMM. Defaults to 1. Not the OpenBLAS pool size.</summary>
    public int ThreadCount { get; set; } = 1;

    /// <summary><c>openblas_set_num_threads</c>. Defaults to 1. Process-wide: applied once,
    /// at registration, and restored by <see cref="NivaraBlas.UseManaged" /> or by disposing
    /// the scope returned from <see cref="NivaraBlas.EnterBlas(string?, int, int)" />.</summary>
    public int OpenBlasThreads { get; set; } = 1;
}