using Nivara.AutoDiff.Operations;
using Nivara.Extensions.Blas;
using NUnit.Framework;

namespace Nivara.Tests.BlasSmoke;

public class BlasSmokeTests
{
    static bool DllPresent() =>
        System.IO.File.Exists("../../../../../lib/libopenblas.dll") ||
        System.IO.File.Exists("lib/libopenblas.dll");

    [Test]
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

        NivaraBlas.UseManaged();
    }

    [Test]
    public void UseBlas_OpenBlasThreads_IsWhatGetNumThreadsReports()
    {
        if (!DllPresent())
            Assert.Ignore("libopenblas.dll not present; skipping BLAS smoke test.");

        var registered = false;
        try
        {
            registered = NivaraBlas.UseBlas(threadCount: 1, openBlasThreads: 4);
            Assert.That(registered, Is.True);
            var a = new float[8 * 8];
            var b = new float[8 * 8];
            var result = new float[8 * 8];
            GradKernels.MatMulTransposedB(a, b, result, 8, 8, 8);
            Assert.That(OpenBlasInterop.GetNumThreads(), Is.EqualTo(4));
        }
        finally
        {
            NivaraBlas.UseManaged();
            if (registered)
                OpenBlasInterop.SetNumThreads(1);
        }
    }

    [Test]
    public void UseManaged_FallsBackToManagedPath()
    {
        NivaraBlas.UseManaged();
        Assert.That(GemmProviders.Current, Is.Null);
    }
}
