using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;

namespace Nivara.SimdProbe;

/// <summary>Benchmarks SIMD kernels against the scalar BCL path.</summary>
internal static class Benchmark
{
    public static int RunAll()
    {
        Console.WriteLine("=== Benchmarks ===");
        Console.WriteLine();

        // Dot products across a range of lengths.
        int[] sizes = [128, 384, 768, 1536, 3072];
        foreach (int n in sizes)
        {
            var bf16 = RandomBf16(n);
            var hf = RandomHalf(n);
            var bf16b = RandomBf16(n);
            var hfb = RandomHalf(n);

            // Scalar baseline: TensorPrimitives.Dot on the narrow type (BCL scalar fallback).
            ulong scalarBf16 = TimeIt(() => TensorPrimitives.Dot<BFloat16>(bf16, bf16b));
            ulong scalarHalf = TimeIt(() => TensorPrimitives.Dot<Half>(hf, hfb));

            // SIMD kernel.
            ulong simdBf16 = TimeIt(() => NarrowSimdKernels.DotBf16(bf16, bf16b));
            ulong simdHalf = TimeIt(() => NarrowSimdKernels.DotHalf(hf, hfb));

            string row = $"  n={n,5} | BF16 scalar={scalarBf16,6} ns simd={simdBf16,6} ns ({Speedup(scalarBf16, simdBf16)}) | Half scalar={scalarHalf,6} ns simd={simdHalf,6} ns ({Speedup(scalarHalf, simdHalf)})";
            Console.WriteLine(row);
        }

        Console.WriteLine();

        // Element-wise ops.
        {
            var a = RandomBf16(3072);
            var b = RandomBf16(3072);
            var dst = new BFloat16[a.Length];
            var dst2 = new BFloat16[a.Length];

            ulong scalarAdd = TimeIt(() => TensorPrimitives.Add<BFloat16>(a, b, dst2));
            ulong simdAdd = TimeIt(() => NarrowSimdKernels.AddBf16(a, b, dst));
            Console.WriteLine($"  AddBf16 (n=3072): scalar={scalarAdd,6} ns simd={simdAdd,6} ns ({Speedup(scalarAdd, simdAdd)})");

            ulong scalarMul = TimeIt(() => TensorPrimitives.Multiply<BFloat16>(a, b, dst2));
            ulong simdMul = TimeIt(() => NarrowSimdKernels.MultiplyBf16(a, b, dst));
            Console.WriteLine($"  MulBf16 (n=3072): scalar={scalarMul,6} ns simd={simdMul,6} ns ({Speedup(scalarMul, simdMul)})");

            ulong scalarF32 = TimeIt(() => TensorPrimitives.Add(ToFloatSpan(a), ToFloatSpan(b), new float[a.Length]));
            Console.WriteLine($"  Add<F32> (n=3072): {scalarF32,6} ns (reference F32 speed)");
        }

        return 0;
    }

    static string Speedup(ulong scalar, ulong simd)
        => scalar >= simd && simd > 0 ? $"{scalar / (double)simd:F1}x" : "slower";

    static ulong TimeIt(Action action)
    {
        // warm
        action();
        int reps = 2000;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < reps; i++) action();
        sw.Stop();
        return (ulong)(sw.Elapsed.TotalNanoseconds / reps);
    }

    static float[] ToFloatSpan(BFloat16[] arr)
    {
        var f = new float[arr.Length];
        for (int i = 0; i < arr.Length; i++) f[i] = (float)arr[i];
        return f;
    }

    static BFloat16[] RandomBf16(int n)
    {
        var rng = new Random(99);
        var arr = new BFloat16[n];
        for (int i = 0; i < n; i++) arr[i] = (BFloat16)((float)rng.NextDouble() * 2 - 1);
        return arr;
    }

    static Half[] RandomHalf(int n)
    {
        var rng = new Random(199);
        var arr = new Half[n];
        for (int i = 0; i < n; i++) arr[i] = (Half)((float)rng.NextDouble() * 2 - 1);
        return arr;
    }
}
