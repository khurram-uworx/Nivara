using Nivara.AutoDiff.Operations;
using Nivara.Extensions.Blas;
using Nivara.Tensors;
using NUnit.Framework;
using System.Numerics;

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
        Assert.That(GemvProviders.Current, Is.Null);
    }

    [Test]
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
            OpenBlasInterop.SetNumThreads(1);
        }
    }

    [Test]
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
            OpenBlasInterop.SetNumThreads(1);
        }
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
