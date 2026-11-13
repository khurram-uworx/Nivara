using Nivara.AutoDiff.Operations;
using Nivara.Extensions.Blas;
using Nivara.Tensors;
using NUnit.Framework;
using System.IO;
using System.Numerics;

namespace Nivara.Tests.BlasSmoke;

/// <summary>
/// OpenBLAS smoke tests. The fixture touches the native library, which the runtime
/// caches on the first load, so tests that assert on <see cref="OpenBlasInterop.LoadedPath" />
/// must run first; hence the <see cref="OrderAttribute" /> values. The fixture is not
/// parallelised.
/// </summary>
[NonParallelizable]
public class BlasSmokeTests
{
    static bool DllPresent() =>
        System.IO.File.Exists("../../../../../lib/libopenblas.dll") ||
        System.IO.File.Exists("lib/libopenblas.dll");

    static string RepoDllPath()
    {
        if (System.IO.File.Exists("../../../../../lib/libopenblas.dll"))
            return Path.GetFullPath("../../../../../lib/libopenblas.dll");
        return Path.GetFullPath("lib/libopenblas.dll");
    }

    [SetUp]
    public void SetUp() => NivaraBlas.UseManaged();

    [TearDown]
    public void TearDown() => NivaraBlas.UseManaged();

    [Test]
    public void UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        // A copy outside every CandidatePaths() entry, so auto-discovery cannot
        // accidentally satisfy the request.
        string dir = Path.Combine(Path.GetTempPath(), "nivara-blas-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        string copy = Path.Combine(dir, "libopenblas.dll");
        File.Copy(RepoDllPath(), copy, overwrite: true);

        try
        {
            NivaraBlas.UseManaged();
            Assert.That(NivaraBlas.UseBlas(copy), Is.True, NivaraBlas.LastDiagnostic);
            Assert.That(OpenBlasInterop.LoadedPath, Is.EqualTo(copy));
            Assert.That(OpenBlasInterop.ResolvedPath, Is.EqualTo(copy));

            // MatMul parity against the managed path.
            int m = 64, k = 128, n = 64;
            var a = new float[m * k];
            var bt = new float[n * k];
            var r = new Random(7);
            for (int i = 0; i < a.Length; i++) a[i] = (float)r.NextDouble();
            for (int i = 0; i < bt.Length; i++) bt[i] = (float)r.NextDouble();

            var managed = new float[m * n];
            GradKernels.MatMulTransposedB(a, bt, managed, m, k, n);

            var blas = new float[m * n];
            GradKernels.MatMulTransposedB(a, bt, blas, m, k, n);

            double maxAbs = 0;
            for (int i = 0; i < managed.Length; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs(managed[i] - blas[i]));
            Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-3), $"maxAbs = {maxAbs:E3}");

            // A later, different (but existing) explicit path cannot take effect and
            // names both the loaded file and the requested one.
            string secondDir = Path.Combine(Path.GetTempPath(), "nivara-blas-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(secondDir);
            string second = Path.Combine(secondDir, "libopenblas.dll");
            File.Copy(RepoDllPath(), second, overwrite: true);
            try
            {
                Assert.That(NivaraBlas.UseBlas(second), Is.False);
                Assert.That(NivaraBlas.LastDiagnostic, Does.Contain(OpenBlasInterop.LoadedPath!));
                Assert.That(NivaraBlas.LastDiagnostic, Does.Contain(second));
            }
            finally
            {
                try { File.Delete(second); } catch (Exception) { }
                try { Directory.Delete(secondDir); } catch (Exception) { }
            }
        }
        finally
        {
            // The module stays mapped for the process lifetime, so the file may be
            // locked; best-effort cleanup.
            try { File.Delete(copy); } catch (Exception) { }
            try { Directory.Delete(dir); } catch (Exception) { }
        }
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseBlas_NonexistentExplicitPath_ReturnsFalse_NamingPath()
    {
        string bogus = Path.Combine(Path.GetTempPath(), "nivara-blas-missing-" + Guid.NewGuid().ToString("N"), "libopenblas.dll");
        NivaraBlas.UseManaged();
        Assert.That(NivaraBlas.UseBlas(bogus), Is.False);
        Assert.That(NivaraBlas.LastDiagnostic, Does.Contain(bogus));
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseBlas_RegistersProvider_AndMatMulMatchesManaged()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        int m = 64, k = 128, n = 64;
        var a = new float[m * k];
        var bt = new float[n * k];
        var r = new Random(7);
        for (int i = 0; i < a.Length; i++) a[i] = (float)r.NextDouble();
        for (int i = 0; i < bt.Length; i++) bt[i] = (float)r.NextDouble();

        NivaraBlas.UseManaged();
        var managed = new float[m * n];
        GradKernels.MatMulTransposedB(a, bt, managed, m, k, n);

        Assert.That(NivaraBlas.UseBlas(), Is.True);
        var blas = new float[m * n];
        GradKernels.MatMulTransposedB(a, bt, blas, m, k, n);

        double maxAbs = 0;
        for (int i = 0; i < managed.Length; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(managed[i] - blas[i]));
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-3), $"maxAbs = {maxAbs:E3}");
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseBlas_OpenBlasThreads_AppliedAtRegistration()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        NivaraBlas.UseManaged();
        Assert.That(NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 4), Is.True);
        // No MatMul in between: the count is set at registration.
        Assert.That(OpenBlasInterop.GetNumThreads(), Is.EqualTo(4));
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void MatMul_DoesNotChangeOpenBlasThreads()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        NivaraBlas.UseManaged();
        Assert.That(NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 4), Is.True);
        OpenBlasInterop.SetNumThreads(1);
        var a = new float[8 * 8];
        var b = new float[8 * 8];
        var result = new float[8 * 8];
        GradKernels.MatMulTransposedB(a, b, result, 8, 8, 8);
        Assert.That(OpenBlasInterop.GetNumThreads(), Is.EqualTo(1));
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void Gemv_DoesNotChangeOpenBlasThreads()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        NivaraBlas.UseManaged();
        Assert.That(NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 4), Is.True);
        OpenBlasInterop.SetNumThreads(1);
        var a = new float[8];
        var b = new float[8 * 8];
        var result = new float[8];
        TensorsHelper.MultiplyCore(a.AsSpan(), b.AsSpan(), result, 1, 8, 8, bTransposed: true);
        Assert.That(OpenBlasInterop.GetNumThreads(), Is.EqualTo(1));
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseManaged_FallsBackToManagedPath()
    {
        NivaraBlas.UseManaged();
        Assert.That(GemmProviders.Current, Is.Null);
        Assert.That(GemvProviders.Current, Is.Null);
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseManaged_RestoresPreviousOpenBlasThreads()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        NivaraBlas.UseManaged();
        OpenBlasInterop.SetNumThreads(2);
        Assert.That(NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 4), Is.True);
        NivaraBlas.UseManaged();
        Assert.That(OpenBlasInterop.GetNumThreads(), Is.EqualTo(2));
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void EnterBlas_Dispose_RestoresPreviousRegistration()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        NivaraBlas.UseManaged();
        Assert.That(NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 2), Is.True);
        var outer = GemmProviders.Current;
        using (NivaraBlas.EnterBlas(threadCount: 1, openBlasThreads: 3))
        {
            Assert.That(GemmProviders.Current, Is.Not.Null);
            Assert.That(GemmProviders.Current, Is.Not.SameAs(outer));
        }
        Assert.That(GemmProviders.Current, Is.SameAs(outer));
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseBlas_SingleRowFloatAndDouble_MatchManaged()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        try
        {
            AssertSingleRow<float>(64, 48);
            AssertSingleRow<double>(32, 40);
        }
        finally
        {
            NivaraBlas.UseManaged();
        }
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseBlas_SingleRowHalf_StaysOnManagedPath()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        int k = 16, n = 12;
        var a = new Half[k];
        var b = new Half[n * k];
        var random = new Random(7);
        for (int i = 0; i < a.Length; i++) a[i] = (Half)random.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (Half)random.NextDouble();

        NivaraBlas.UseManaged();
        var managed = new Half[n];
        TensorsHelper.MultiplyCore(a.AsSpan(), b.AsSpan(), managed, 1, k, n, bTransposed: true);

        try
        {
            Assert.That(NivaraBlas.UseBlas(openBlasThreads: 4), Is.True);
            var blas = new Half[n];
            TensorsHelper.MultiplyCore(a.AsSpan(), b.AsSpan(), blas, 1, k, n, bTransposed: true);

            for (int i = 0; i < n; i++)
                Assert.That(blas[i], Is.EqualTo(managed[i]));
        }
        finally
        {
            NivaraBlas.UseManaged();
        }
    }

    [Test]
    [DependsOnTest(nameof(UseBlas_ExplicitPath_LoadsThatFile_AndLaterDifferentPathIsRejected))]
    public void UseBlas_ConcurrentMatMulAndToggle_DoesNotThrow()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        var a = new float[16 * 16];
        var b = new float[16 * 16];
        var r = new Random(1);
        for (int i = 0; i < a.Length; i++) a[i] = (float)r.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)r.NextDouble();

        NivaraBlas.UseManaged();
        Assert.That(NivaraBlas.UseBlas(), Is.True);

        var exceptions = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        Parallel.For(0, 64, i =>
        {
            try
            {
                if (i % 4 == 0)
                {
                    NivaraBlas.UseManaged();
                    NivaraBlas.UseBlas();
                }
                else
                {
                    var result = new float[16 * 16];
                    GradKernels.MatMulTransposedB(a, b, result, 16, 16, 16);
                }
            }
            catch (Exception ex)
            {
                exceptions.Add(ex);
            }
        });

        Assert.That(exceptions, Is.Empty, string.Join("; ", exceptions.Select(e => e.Message)));
    }

    static void AssertSingleRow<T>(int k, int n)
        where T : struct, IFloatingPointIeee754<T>
    {
        var a = new T[k];
        var b = new T[n * k];
        var random = new Random(7);
        for (int i = 0; i < a.Length; i++) a[i] = T.CreateChecked(random.NextDouble());
        for (int i = 0; i < b.Length; i++) b[i] = T.CreateChecked(random.NextDouble());

        NivaraBlas.UseManaged();
        var managed = new T[n];
        GradKernels.MatMulTransposedB(a, b, managed, 1, k, n);

        Assert.That(NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 4), Is.True);
        Assert.That(GemvProviders.Current, Is.Not.Null);
        var blas = new T[n];
        GradKernels.MatMulTransposedB(a, b, blas, 1, k, n);
        Assert.That(OpenBlasInterop.GetNumThreads(), Is.EqualTo(4));

        double maxAbs = 0;
        for (int i = 0; i < n; i++)
            maxAbs = Math.Max(maxAbs, Math.Abs(double.CreateChecked(managed[i]) - double.CreateChecked(blas[i])));
        Assert.That(maxAbs, Is.LessThanOrEqualTo(1e-3), $"{typeof(T).Name} maxAbs = {maxAbs:E3}");
    }
}
