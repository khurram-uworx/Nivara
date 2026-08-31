using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Nivara.SimdProbe;

/// <summary>Validates each SIMD kernel against its scalar correctness baseline.</summary>
internal static class Correctness
{
    public static int RunAll()
    {
        int failures = 0;
        Console.WriteLine("=== Correctness ===");
        Console.WriteLine($"Architecture: {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"Vector128.IsHardwareAccelerated: {Vector128.IsHardwareAccelerated}");
        Console.WriteLine();

        failures += Check("DotBf16", () =>
        {
            var a = Bf16Vec(256);
            var b = Bf16Vec(384);
            float scalar = DotScalarBf16Reference(a, b);
            float simdResult = NarrowSimdKernels.DotBf16(a, b);
            return NearlyEqual(simdResult, scalar, 1e-3f);
        });

        failures += Check("DotHalf", () =>
        {
            var a = RandomHalf(256);
            var b = RandomHalf(384);
            float scalar = DotScalarHalfReference(a, b);
            float simdResult = NarrowSimdKernels.DotHalf(a, b);
            return NearlyEqual(simdResult, scalar, 1e-3f);
        });

        failures += Check("AddBf16", () =>
        {
            var a = RandomBf16(256);
            var b = RandomBf16(256);
            var dst = new BFloat16[a.Length];
            NarrowSimdKernels.AddBf16(a, b, dst);
            for (int i = 0; i < a.Length; i++)
                if (!NearlyEqual((float)dst[i], (float)a[i] + (float)b[i], 1e-2f)) return false;
            return true;
        });

        failures += Check("MultiplyBf16", () =>
        {
            var a = RandomBf16(256);
            var b = RandomBf16(256);
            var dst = new BFloat16[a.Length];
            NarrowSimdKernels.MultiplyBf16(a, b, dst);
            for (int i = 0; i < a.Length; i++)
                if (!NearlyEqual((float)dst[i], (float)a[i] * (float)b[i], 1e-2f)) return false;
            return true;
        });

        failures += Check("RmsNormBf16", () =>
        {
            var a = RandomBf16(8 * 64);
            var dst = new BFloat16[a.Length];
            NarrowSimdKernels.RmsNormBf16(a, dst, 8, 64, 1e-6f);
            return true; // structural check only; numeric verified in mismatch below
        });

        // RmsNorm numeric check
        {
            var a = RandomBf16(8 * 64);
            var simdDst = new BFloat16[a.Length];
            NarrowSimdKernels.RmsNormBf16(a, simdDst, 8, 64, 1e-6f);
            bool ok = true;
            for (int i = 0; i < 8; i++)
            {
                float sumSq = 0;
                for (int j = 0; j < 64; j++) { float f = (float)a[i * 64 + j]; sumSq += f * f; }
                float inv = 1f / MathF.Sqrt(sumSq / 64 + 1e-6f);
                for (int j = 0; j < 64; j++)
                {
                    float expect = (float)a[i * 64 + j] * inv;
                    if (!NearlyEqual((float)simdDst[i * 64 + j], expect, 1e-2f)) { ok = false; break; }
                }
            }
            if (!ok) { Console.WriteLine("    FAIL rmsnorm numeric"); failures++; }
        }

        Console.WriteLine();
        Console.WriteLine(failures == 0 ? "All correctness checks PASSED." : $"{failures} correctness check(s) FAILED.");
        return failures;
    }

    static int Check(string name, Func<bool> body)
    {
        bool ok = false;
        try { ok = body(); }
        catch (Exception ex) { Console.WriteLine($"  FAIL {name}: {ex.GetType().Name}: {ex.Message}"); return 1; }
        Console.WriteLine($"  {(ok ? "PASS" : "FAIL")} {name}");
        return ok ? 0 : 1;
    }

    static bool NearlyEqual(float a, float b, float tol) => Math.Abs(a - b) <= tol;

    static BFloat16[] RandomBf16(int n)
    {
        var rng = new Random(1234);
        var arr = new BFloat16[n];
        for (int i = 0; i < n; i++) arr[i] = (BFloat16)((float)rng.NextDouble() * 2 - 1);
        return arr;
    }

    static Half[] RandomHalf(int n)
    {
        var rng = new Random(4321);
        var arr = new Half[n];
        for (int i = 0; i < n; i++) arr[i] = (Half)((float)rng.NextDouble() * 2 - 1);
        return arr;
    }

    static BFloat16[] Bf16Vec(int seed) => Enumerable.Range(0, seed).Select(i => (BFloat16)(i * 1e-4f - 0.01f)).ToArray();

    static float DotScalarBf16Reference(ReadOnlySpan<BFloat16> a, ReadOnlySpan<BFloat16> b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += (float)a[i] * (float)b[i];
        return s;
    }

    static float DotScalarHalfReference(ReadOnlySpan<Half> a, ReadOnlySpan<Half> b)
    {
        float s = 0;
        for (int i = 0; i < a.Length; i++) s += (float)a[i] * (float)b[i];
        return s;
    }
}
